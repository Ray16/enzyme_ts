"""Batched CI-NEB reaction-path + TS search on the UMA PES.

Two-stage nudged elastic band:
  (1) plain improved-tangent NEB to converge the band toward the MEP, then
  (2) climbing-image (CI) to drive the highest image onto the saddle.
The ENTIRE band (all images incl. endpoints) is evaluated in ONE batched UMA
predict per optimizer step via uma_batch.BatchedNEB -> amortizes launch overhead.

The climbing image is returned as an exact-saddle GUESS; run_cineb_to_ts then
refines it with the validated Sella + full-Hessian + mode-descent workflow, so the
final TS is exact and validated (one imaginary mode, connected endpoints) -- CI-NEB
only supplies the guess and a cross-check barrier from the band.

Endpoints are held fixed by ASE's NEB (only interior images move); any FixAtoms
constraint on the endpoints (e.g. frozen Calpha) is carried by every image.
"""
from __future__ import annotations
import numpy as np
from ase.optimize import FIRE
from ase.constraints import FixAtoms, FixInternals
from ..engine import calculator as uma_helper, batch as uma_batch
from ..optim import relax
from . import scan as relaxed_scan

KCAL = uma_helper.KCAL_MOL_PER_EV


def seed_band(reactant, product, seed_bonds, charge, spin, model, nimg,
              frozen=None, fmax=0.1, steps=120):
    """Build the initial NEB band by a RELAXED SCAN along the reaction
    coordinate(s), instead of geometric interpolation (idpp/linear).

    Why: geometric interpolation is the fragile step. idpp can hang (observed:
    it never returns for a linear molecule like HCN), and linear interpolation
    drives atoms straight THROUGH each other on a bond swap, producing clash
    spikes of thousands of kcal/mol (observed on HCN<->HNC: +105 eV mid-band).
    A relaxed scan instead drives the reaction-coordinate distances from their
    reactant to product values and relaxes all other DOF at each step, so every
    image is a physical, near-MEP, clash-free point -- the same scan->string
    seeding the ML/MM-toolkit paper (Ohmura 2025) uses ahead of its GSM search.

    seed_bonds: list of (i, j) atom-index pairs defining the reaction coordinate
    (each held at a scanned distance; j's fragment is the one moved to set it).
    Endpoint distances are read from reactant/product. Returns nimg+2 chained
    Atoms (each interior image relaxed starting from the previous one)."""
    seed_bonds = [tuple(b) for b in seed_bonds]
    dR = [reactant.get_distance(i, j) for (i, j) in seed_bonds]
    dP = [product.get_distance(i, j) for (i, j) in seed_bonds]
    frozen = list(frozen) if frozen is not None else []
    band = [reactant.copy()]
    prev = reactant
    for t in np.linspace(0, 1, nimg + 2)[1:-1]:
        img = prev.copy()
        img.info.update(charge=int(charge), spin=int(spin))
        targets = [(1 - t) * dR[b] + t * dP[b] for b in range(len(seed_bonds))]
        for (i, j), d in zip(seed_bonds, targets):
            relaxed_scan.set_distance_rigid(img, i, j, d)
        img.calc = uma_helper.get_calculator("omol", model)
        cons = ([FixAtoms(indices=frozen)] if frozen else []) + \
               [FixInternals(bonds=[[float(d), [i, j]] for (i, j), d in zip(seed_bonds, targets)])]
        img.set_constraint(cons)
        try:
            relax(img, fmax=fmax, steps=steps, logfile=None, maxstep=0.1)
        except Exception as e:
            # a single image relax failing (e.g. xTB SCF blow-up after etemp
            # retries) must NOT kill the whole run -- keep the distance-set
            # (already near-physical) geometry; the NEB refines the band anyway.
            print(f"[seed] WARNING: image t={t:.2f} relax failed ({type(e).__name__}: {e}); "
                  f"using unrelaxed seeded geometry")
        img.set_constraint(FixAtoms(indices=frozen) if frozen else [])
        band.append(img.copy())
        prev = img
    band.append(product.copy())
    d0 = [f"{b}:{dR[k]:.2f}->{dP[k]:.2f}" for k, b in enumerate(seed_bonds)]
    print(f"[seed] relaxed-scan band: {len(band)} images along {len(seed_bonds)} coord(s) {d0}")
    return band


def run_cineb(reactant, product, charge, spin, model, nimg=9, fmax=0.10,
              climb_fmax=0.05, steps=400, interp="idpp", label="cineb",
              maxstep=0.2, climb_maxstep=0.05, initial_band=None):
    """Return dict(images, energies[eV], profile[kcal/mol rel reactant], i_ts, barrier, ts_guess).

    maxstep / climb_maxstep cap the FIRE optimizer's per-step atomic displacement
    (Angstrom) in the no-climb and climbing stages respectively. The climb stage
    is capped tighter by default: forcing the highest image uphill can drive a
    large step into an atomic clash and diverge (observed on MPD with a tight
    free region -- energy ran away ~1000 kcal/mol over ~35 uncapped FIRE steps).
    A small climb step keeps the saddle ascent on the band instead of overshooting.
    """
    if initial_band is not None:
        # a pre-built band (e.g. from seed_band's relaxed scan) -- use it directly,
        # skipping the fragile geometric interpolation entirely.
        images = [im.copy() for im in initial_band]
    else:
        images = [reactant.copy()] + [reactant.copy() for _ in range(nimg)] + [product.copy()]
    for im in images:
        im.info["charge"] = int(charge); im.info["spin"] = int(spin)
    neb = uma_batch.BatchedNEB(images, charge, spin, model,
                               climb=False, method="improvedtangent")
    if initial_band is None:
        # geometric interpolation of the interior images (no calculator needed)
        try:
            neb.interpolate(method=interp, apply_constraint=True)
        except TypeError:
            neb.interpolate(method=interp)

    opt = FIRE(neb, logfile="-", trajectory=f"{label}.traj", maxstep=maxstep)
    conv1 = opt.run(fmax=fmax, steps=steps)          # stage 1: converge band (no climb)
    neb.climb = True
    opt.maxstep = climb_maxstep                      # cap the uphill climb step -> no clash/divergence
    conv2 = opt.run(fmax=climb_fmax, steps=steps)    # stage 2: climbing image -> saddle

    E, _ = uma_batch.batched_ef(images, charge, spin, model)
    E = np.asarray(E)
    prof = (E - E[0]) * KCAL

    # --- band-quality guard -------------------------------------------------
    # The plain argmax can land on a CLASH-SPIKE image: an interpolated geometry
    # with atoms crashing (energy hundreds-to-thousands of kcal/mol above the
    # reactant), which is not a saddle at all -- handing it to Sella wastes the
    # whole refinement (observed: a 616 kcal/mol PETase k=2 band guess). No real
    # enzymatic elementary step has a barrier anywhere near CLASH_KCAL, so treat
    # any image above it as an artifact and pick the highest PHYSICAL interior
    # image as the TS guess instead. The pathological flag lets the caller retry
    # with more images.
    CLASH_KCAL = 150.0
    interior = list(range(1, len(prof) - 1))
    clash = prof > CLASH_KCAL
    non_clash = [i for i in interior if not clash[i]]
    i_argmax = int(np.argmax(prof))
    pathological = False
    if non_clash:
        i_ts = max(non_clash, key=lambda i: prof[i])
        if clash[i_argmax]:
            print(f"[cineb] WARNING: band max (i={i_argmax}, {prof[i_argmax]:.0f} kcal/mol) is a CLASH "
                  f"spike; using highest physical image i={i_ts} ({prof[i_ts]:.1f} kcal/mol) as TS guess. "
                  f"{int(clash.sum())} clash image(s) in band -- consider more --nimg.")
    else:
        # every interior image is clashed -> band is unusable as-is
        i_ts = max(interior, key=lambda i: prof[i]) if interior else i_argmax
        pathological = True
        print(f"[cineb] WARNING: ALL interior images exceed {CLASH_KCAL:.0f} kcal/mol -- band is "
              f"pathological (interpolation clash). Caller should retry with more images.")

    ts_guess = images[i_ts].copy(); ts_guess.info.update(charge=int(charge), spin=int(spin))
    print(f"[cineb] band {len(images)} images | converged(no-climb)={bool(conv1)} climb={bool(conv2)}")
    print(f"[cineb] profile (kcal/mol): " + " ".join(f"{x:+.1f}" for x in prof))
    print(f"[cineb] TS guess i={i_ts}  barrier = {prof[i_ts]:.1f} kcal/mol  "
          f"(raw band max {prof[i_argmax]:.1f} at i={i_argmax})")
    return dict(images=images, energies=E, profile=prof, i_ts=i_ts,
                barrier=float(prof[i_ts]), ts_guess=ts_guess,
                band_max=float(prof[i_argmax]), n_clash=int(clash.sum()),
                pathological=bool(pathological))


def run_cineb_to_ts(reactant, product, charge, spin, model, results_prefix,
                    nimg=9, frozen=None, seed_bonds=None, **kw):
    """Full path: CI-NEB band -> climbing-image guess -> Sella exact saddle + Hessian + endpoints.
    Returns the ts_sella diag dict, augmented with the NEB band barrier/profile.

    seed_bonds (recommended): reaction-coordinate atom pairs. When given, the
    initial band is built by a RELAXED SCAN (seed_band) instead of geometric
    interpolation -- clash-free and robust (idpp can hang, linear clashes). This
    is the preferred path; interpolation is only the fallback when seed_bonds is
    None."""
    from . import saddle as ts_sella
    if seed_bonds is not None:
        band = seed_band(reactant, product, seed_bonds, charge, spin, model, nimg, frozen=frozen)
        r = run_cineb(reactant, product, charge, spin, model, nimg=nimg,
                      label=f"{results_prefix}_neb", initial_band=band, **kw)
    else:
        r = run_cineb(reactant, product, charge, spin, model, nimg=nimg,
                      label=f"{results_prefix}_neb", **kw)
        # Interpolated band only: if pathological (all interior images clash), a
        # denser band takes smaller geometric steps and usually removes the clash
        # (MPD lesson: nimg 9 -> 21). Retry up to twice with ~1.8x images. (A
        # seeded band is already clash-free, so no retry needed there.)
        retries = 0
        while r.get("pathological") and retries < 2:
            nimg = int(nimg * 1.8)
            retries += 1
            print(f"[cineb] pathological band -> retry {retries} with nimg={nimg}")
            r = run_cineb(reactant, product, charge, spin, model, nimg=nimg,
                          label=f"{results_prefix}_neb", **kw)
    ts_guess = r["ts_guess"]
    if frozen is not None:
        from ase.constraints import FixAtoms
        ts_guess.set_constraint(FixAtoms(indices=list(frozen)))

    # Diagnostic: check the RAW climbing image's own vibrational character
    # BEFORE Sella touches it. This distinguishes two very different bugs:
    # (a) the NEB band itself isn't well-localized (raw guess doesn't have
    #     genuine 1-imaginary-mode saddle character -> need more images /
    #     tighter climb convergence), vs (b) the guess IS already saddle-like
    # but Sella's optimization trajectory drifts off it into a nearby minimum
    #     (need different Sella settings, not a better band).
    from ase.io import write
    write(f"{results_prefix}_neb_climbing_image_raw.xyz", ts_guess)
    raw_freqs, raw_n_imag, raw_imag, _, _ = ts_sella.confirm(ts_guess, charge, spin, model)
    print(f"[cineb] RAW climbing image (pre-Sella) vibrational check: "
          f"n_imag={raw_n_imag}  imag={[f'{complex(x).imag:.0f}i' for x in raw_imag]}")

    diag = ts_sella.run_full(ts_guess, charge, spin, model,
                             label="CI-NEB -> Sella TS", results_prefix=results_prefix)
    diag["neb_barrier"] = r["barrier"]
    diag["neb_i_ts"] = r["i_ts"]
    diag["neb_profile"] = r["profile"].tolist()
    diag["neb_band_max"] = r.get("band_max")
    diag["neb_n_clash"] = r.get("n_clash")
    diag["neb_pathological"] = r.get("pathological")
    diag["neb_raw_guess_n_imag"] = raw_n_imag
    diag["neb_raw_guess_imag_cm"] = [float(complex(x).imag) for x in raw_imag]
    return diag
