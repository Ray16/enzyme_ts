"""Rigorous UMA TS workflow: Sella saddle search + batched Hessian + IRC.

Pipeline:
  TS guess --Sella(order=1)--> true first-order saddle
           --batched Hessian--> confirm exactly ONE imaginary mode
           --IRC fwd/rev------> reactant & product CONNECTED to the TS
           --LBFGS------------> relax IRC endpoints to minima

Because R and P are produced by IRC from the same saddle, they are on one
connected path in a consistent conformational family (valid energy diagram).
"""
from __future__ import annotations
import numpy as np
from ase.io import write
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.formula import Formula
from sella import Sella, IRC
from ..engine import calculator as uma_helper, batch as uma_batch
from ..optim import relax

KCAL = uma_helper.KCAL_MOL_PER_EV
THRESH_CM = 50.0   # cm^-1: modes with |imag| below this are numerical near-zero
                   # (floppy/trans-rot residue), not genuine reaction saddle modes


def identify_fragments(atoms, scale=1.2):
    """Detect bonds from 3D geometry (scaled covalent radii) and return the list
    of disconnected molecular fragments as Hill formulas. Used to VERIFY that an
    IRC endpoint is the chemically expected species, not a random minimum."""
    cut = [c * scale for c in natural_cutoffs(atoms)]
    nl = NeighborList(cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms); seen = [False] * n; frags = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]; comp = []
        while stack:
            i = stack.pop()
            if seen[i]:
                continue
            seen[i] = True; comp.append(i)
            stack.extend(nl.get_neighbors(i)[0])
        frags.append(comp)
    formulas = sorted(str(Formula(atoms[c].get_chemical_formula())) for c in frags)
    return formulas


def attach(atoms, q, s, model):
    atoms.info["charge"] = int(q); atoms.info["spin"] = int(s)
    atoms.calc = uma_helper.get_calculator("omol", model)
    return atoms


def minimize(atoms, q, s, model, fmax=0.02, steps=500, traj=None):
    attach(atoms, q, s, model)
    relax(atoms, fmax=fmax, steps=steps, trajectory=traj)
    return atoms, atoms.get_potential_energy()


def _free_rmsd(pos_a, pos_b, free):
    """RMSD over the free (non-frozen) atoms only -- the subspace Sella moves."""
    if not free:
        free = list(range(len(pos_a)))
    d = np.asarray(pos_a)[free] - np.asarray(pos_b)[free]
    return float(np.sqrt((d * d).sum() / len(free)))


def _run_sella(ts, traj, fmax, steps, delta0=None):
    kw = dict(order=1, internal=False, logfile="-", trajectory=traj,
              refine_initial_hessian=True)
    if delta0 is not None:
        kw["delta0"] = delta0
    Sella(ts, **kw).run(fmax=fmax, steps=steps)
    return ts.get_potential_energy()


def saddle(ts_guess, q, s, model, fmax=0.03, steps=400, traj="sella.traj",
           e_ref=None, max_drift_kcal=60.0, delta0=None):
    """Refine a TS guess to a true first-order saddle with Sella, GUARDED against
    the saddle search wandering out of the guess's reaction basin.

    Observed failure (gas-phase HCN<->HNC regression on UMA): an unconstrained
    Cartesian Sella search left the correct ~48 kcal/mol isomerization saddle and
    rolled ~250 kcal/mol away to a spurious C-N *fragmentation* saddle (2 large
    imaginary modes, dissociated 'C'+'HN' endpoints). Sella's own
    ``allow_fragments=False`` does not catch this in Cartesian mode, and its trust
    radius adapts upward, so neither alone is a reliable bound across systems that
    range from 3-atom gas phase to 200-atom frozen-boundary clusters.

    The GSM highest-energy image handed in here is already a good MEP TS
    approximation, so a "refinement" that shifts the energy by more than
    ``max_drift_kcal`` has almost certainly diverged. On detecting that, we (1)
    retry once from the guess with a much tighter initial trust radius to keep the
    search local, and (2) if it still diverges, keep the guess geometry itself
    (the downstream Hessian + mode-flatten + mode-descent then characterize it).
    ``ts_guess.info['sella_fallback']`` records which path was taken. This keeps
    the generic component self-protecting rather than relying on per-system tuning.
    """
    attach(ts_guess, q, s, model)
    free, _ = _free_indices(ts_guess)
    p_ref = ts_guess.get_positions().copy()
    if e_ref is None:
        e_ref = ts_guess.get_potential_energy()

    # refine_initial_hessian -> exact Hessian at the guess so Sella climbs the
    # correct (reaction) mode instead of wandering to a spurious saddle. Pass
    # delta0 explicitly for a WEAK imaginary mode (shallow curvature, e.g. a broad
    # electrostatically-stabilized barrier): Sella's approximate Hessian updates
    # over many steps can lose track of a barely-negative mode and start plain
    # minimizing (observed: CM k=2, 70i initial mode, geometry drifted steadily
    # toward the reactant over 3000+ steps despite fmax looking like it was
    # converging cleanly) -- a tight initial trust radius keeps steps small enough
    # to stay locked onto that shallow mode instead of stepping off it.
    e_new = _run_sella(ts_guess, traj, fmax, steps, delta0=delta0)
    drift = abs(e_new - e_ref) * KCAL
    ts_guess.info["sella_fallback"] = "none"
    if drift <= max_drift_kcal:
        return ts_guess, e_new

    rmsd = _free_rmsd(ts_guess.get_positions(), p_ref, free)
    print(f"[saddle] WARNING: Sella drifted {drift:.0f} kcal/mol (free-atom RMSD "
          f"{rmsd:.2f} A) from the TS guess -- exceeds {max_drift_kcal:.0f} kcal/mol, "
          f"treating as a runaway to a spurious saddle. Retrying from the guess with "
          f"a tight trust radius (delta0=0.01) ...")
    ts_guess.set_positions(p_ref)
    e_new = _run_sella(ts_guess, f"{traj}", fmax, steps, delta0=0.01)
    drift = abs(e_new - e_ref) * KCAL
    if drift <= max_drift_kcal:
        ts_guess.info["sella_fallback"] = "tight_trust"
        print(f"[saddle] tight-trust retry converged near the guess "
              f"(drift {drift:.1f} kcal/mol).")
        return ts_guess, e_new

    print(f"[saddle] WARNING: tight-trust retry still drifted {drift:.0f} kcal/mol; "
          f"keeping the (unrefined) GSM TS guess as the saddle -- it is a better MEP "
          f"TS estimate than a diverged Sella saddle. The Hessian/mode-flatten stage "
          f"will characterize it; treat n_imag with caution.")
    ts_guess.set_positions(p_ref)
    ts_guess.info["sella_fallback"] = "guess"
    return ts_guess, ts_guess.get_potential_energy()


def _free_indices(atoms):
    from ase.constraints import FixAtoms
    frozen = set()
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            frozen.update(int(i) for i in c.index)
    return [i for i in range(len(atoms)) if i not in frozen], sorted(frozen)


def confirm(ts, q, s, model, partial=True):
    """Vibrational check at the saddle. For a FROZEN-boundary cluster, default to Partial
    Hessian Vibrational Analysis (PHVA) over the free atoms only -- physically the correct
    subspace (frozen Calpha are not DOF) and ~12x cheaper / far lighter on GPU memory.
    Returns (freqs, n_imag, imag, evec, evec_is_cartesian)."""
    free, frozen = _free_indices(ts)
    if partial and frozen and free:
        freqs, evecs_free, imag = uma_batch.batched_partial_hessian(ts, free, q, s, model)
        # lowest (imaginary) mode -> mass-unweighted cartesian displacement over ALL atoms
        m_free = ts.get_masses()[free]
        mode_free = evecs_free[:, 0].reshape(-1, 3) / np.sqrt(m_free)[:, None]
        evec_cart = np.zeros((len(ts), 3)); evec_cart[free] = mode_free
        nrm = np.linalg.norm(evec_cart)
        if nrm > 0:
            evec_cart /= nrm
        return freqs, len(imag), imag, evec_cart, True
    freqs, n_imag, imag, evecs = uma_batch.batched_hessian(ts, q, s, model)
    return freqs, n_imag, imag, evecs, False


def mode_descent(ts, q, s, model, evec_cart, step=0.4, fmax=0.02, tag="ep"):
    """Step off the saddle along +/- the imaginary mode, then minimize each side.
    Robust replacement for IRC (which stops at the saddle since |grad|~0 there).

    The +mode and -mode minimizations are INDEPENDENT but are run SEQUENTIALLY: the
    cached UMA calculator is a single GPU model whose MoLE (mixture-of-linear-experts)
    routing buffers are NOT thread-safe, so two overlapping forward passes from a
    thread pool corrupt each other's state -- observed as 'result shape
    torch.Size([10460, 640]) does not match input shape torch.Size([10492, 768])' in
    mole.py when the fwd/rev geometries have different neighbour-list sizes (it
    silently "worked" only for 3-atom HCN, whose two sides share an identical graph
    size). The tiny wall-clock cost of two back-to-back UMA minimizations is far
    cheaper than a corrupted, non-reproducible saddle characterization."""
    p0 = ts.get_positions()
    fwd0 = ts.copy(); fwd0.set_positions(p0 + step * evec_cart)
    rev0 = ts.copy(); rev0.set_positions(p0 - step * evec_cart)
    fwd, eF = minimize(fwd0, q, s, model, traj=f"{tag}_fwd.traj")
    rev, eR = minimize(rev0, q, s, model, traj=f"{tag}_rev.traj")
    return fwd, eF, rev, eR


def _energy_at(atoms, positions, q, s, model):
    a = atoms.copy(); a.set_positions(positions)
    attach(a, q, s, model)
    return a.get_potential_energy()


def _cart_mode(atoms, free, evecs_free, j):
    """Cartesian, unit-normalized displacement for free-atom eigenvector column j
    (un-mass-weight, then map onto the full atom array; frozen atoms get 0)."""
    m_free = atoms.get_masses()[free]
    mode_free = evecs_free[:, j].reshape(-1, 3) / np.sqrt(m_free)[:, None]
    v = np.zeros((len(atoms), 3)); v[free] = mode_free
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def mode_flatten(ts, q, s, model, results_prefix, max_iter=6, amp=0.35, thresh=THRESH_CM):
    """Remove SPURIOUS imaginary modes from a saddle (Ohmura et al., chemRxiv
    2025-jft1k, PHG-Dimer stage-4 'mode-flattening loop').

    A genuine reaction TS has exactly ONE imaginary mode. A cluster saddle search
    often converges with several (extra small imaginary modes from floppy free-
    region groups -- e.g. a loosely anchored water/sidechain -- competing with the
    real reaction coordinate; this ended every MPD run). We keep the LARGEST
    imaginary mode (the reaction coordinate) and, for every OTHER imaginary mode,
    displace the geometry along it (mass-scaled, sign chosen to LOWER the energy)
    to fall off that spurious ridge, then re-refine the saddle with Sella. Repeat
    until one imaginary mode remains or max_iter is hit.

    Returns (ts, ok, n_imag). Uses the partial (free-atom) Hessian, consistent
    with confirm()."""
    free, frozen = _free_indices(ts)
    for it in range(max_iter):
        freqs, evecs_free, _imag = uma_batch.batched_partial_hessian(ts, free, q, s, model)
        imag_idx = [i for i, f in enumerate(freqs) if np.imag(f) > thresh]
        n_imag = len(imag_idx)
        print(f"[flatten] iter {it}: n_imag={n_imag} "
              f"{[f'{np.imag(freqs[i]):.0f}i' for i in imag_idx]}")
        if n_imag <= 1:
            return ts, True, n_imag
        keep = max(imag_idx, key=lambda i: np.imag(freqs[i]))   # reaction mode
        spurious = [i for i in imag_idx if i != keep]
        p0 = ts.get_positions()
        disp = np.zeros_like(p0)
        for j in spurious:
            v = _cart_mode(ts, free, evecs_free, j)
            ep = _energy_at(ts, p0 + amp * v, q, s, model)
            em = _energy_at(ts, p0 - amp * v, q, s, model)
            disp += amp * v if ep < em else -amp * v
        ts.set_positions(p0 + disp)
        ts, _ = saddle(ts, q, s, model, traj=f"{results_prefix}_flatten{it}.traj")
    # final assessment
    freqs, _, _ = uma_batch.batched_partial_hessian(ts, free, q, s, model)
    n_imag = int(sum(1 for f in freqs if np.imag(f) > thresh))
    print(f"[flatten] reached max_iter={max_iter}: n_imag={n_imag}")
    return ts, n_imag <= 1, n_imag


def run_full(ts_guess, q, s, model, label, results_prefix,
             expected_reactant=None, expected_product=None, flatten=True, delta0=None):
    print(f"\n===== {label}: Sella saddle search =====")
    ts, eTS = saddle(ts_guess, q, s, model, traj=f"{results_prefix}_sella.traj", delta0=delta0)
    write(f"{results_prefix}_ts.xyz", ts)
    freqs, n_imag_raw, imag, evecs, evec_is_cart = confirm(ts, q, s, model)
    # ignore numerical near-zero modes (trans/rot/floppy not projected out);
    # a genuine reaction TS has one LARGE imaginary frequency.
    imag_real = [x for x in imag if np.imag(x) > THRESH_CM]
    n_imag = len(imag_real)
    print(f"[freq] imaginary(all)={[f'{np.imag(x):.0f}i' for x in imag]}")
    print(f"[freq] imaginary(>{THRESH_CM:.0f}cm^-1) = {n_imag}: {[f'{np.imag(x):.0f}i' for x in imag_real]}")

    # Mode-flattening: if the saddle has spurious extra imaginary modes, remove
    # them (Ohmura et al. PHG-Dimer stage 4) and re-characterize before trusting
    # the result. This is the fix for the multi-imaginary-mode failure that ended
    # every MPD run.
    n_flat_iters = 0
    if flatten and n_imag > 1:
        print(f"\n===== {label}: mode-flattening ({n_imag} imaginary modes -> target 1) =====")
        ts, _ok, _ni = mode_flatten(ts, q, s, model, results_prefix)
        n_flat_iters = 1
        write(f"{results_prefix}_ts.xyz", ts)
        eTS = ts.get_potential_energy()
        freqs, n_imag_raw, imag, evecs, evec_is_cart = confirm(ts, q, s, model)
        imag_real = [x for x in imag if np.imag(x) > THRESH_CM]
        n_imag = len(imag_real)
        print(f"[freq] after flatten: imaginary(>{THRESH_CM:.0f}cm^-1) = {n_imag}: "
              f"{[f'{np.imag(x):.0f}i' for x in imag_real]}")
    ts_ok = (n_imag == 1)

    print(f"\n===== {label}: mode-following descent to connected endpoints =====")
    # cartesian displacement of the imaginary mode (un-mass-weight the eigenvector)
    if evec_is_cart:
        evec_cart = evecs                      # PHVA already returned cartesian, mapped, normalized
    else:
        mode = evecs[:, 0].reshape(-1, 3)
        masses = ts.get_masses()
        evec_cart = mode / np.sqrt(masses)[:, None]
        evec_cart /= np.linalg.norm(evec_cart)
    fwd, eF, rev, eR = mode_descent(ts, q, s, model, evec_cart, tag=results_prefix)
    # reactant/product = the two connected minima, labeled by energy
    if eR <= eF:
        react, eReact, prod, eProd = rev, eR, fwd, eF
    else:
        react, eReact, prod, eProd = fwd, eF, rev, eR
    write(f"{results_prefix}_reactant.xyz", react)
    write(f"{results_prefix}_product.xyz", prod)

    # ---- IRC endpoint identity check (verify correct chemical species) ----
    react_frags = identify_fragments(react)
    prod_frags = identify_fragments(prod)
    print("\n==== IRC endpoint identity ====")
    print(f"  reactant fragments: {react_frags}")
    print(f"  product  fragments: {prod_frags}")
    react_ok = prod_ok = None
    if expected_reactant is not None:
        react_ok = (sorted(react_frags) == sorted(expected_reactant))
        print(f"  reactant matches expected {expected_reactant}: {react_ok}")
    if expected_product is not None:
        prod_ok = (sorted(prod_frags) == sorted(expected_product))
        print(f"  product  matches expected {expected_product}: {prod_ok}")

    fwd_barrier = (eTS - eReact) * KCAL
    rev_barrier = (eTS - eProd) * KCAL
    rxn = (eProd - eReact) * KCAL
    print("\n==== reaction energy diagram (kcal/mol, rel. reactant) ====")
    print(f"  reactant : 0.00")
    print(f"  TS       : {fwd_barrier:.2f}")
    print(f"  product  : {rxn:.2f}")
    print(f"  (reverse barrier {rev_barrier:.2f})")
    print(f"  TS valid (1 imag freq): {ts_ok}")

    diag = dict(reactant=0.0, ts=float(fwd_barrier), product=float(rxn),
                reverse_barrier=float(rev_barrier), n_imag=int(n_imag),
                n_imag_raw=int(n_imag_raw), ts_valid=bool(ts_ok),
                imag_cm=[float(np.imag(x)) for x in imag_real],
                imag_cm_all=[float(np.imag(x)) for x in imag],
                mode_flatten_applied=bool(n_flat_iters),
                sella_fallback=ts.info.get("sella_fallback", "none"),
                reactant_fragments=react_frags, product_fragments=prod_frags,
                reactant_id_ok=react_ok, product_id_ok=prod_ok,
                e_reactant_eV=float(eReact), e_ts_eV=float(eTS), e_product_eV=float(eProd))
    return diag
