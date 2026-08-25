"""End-to-end enzyme TS / Gibbs pipeline: prep -> carve -> scan -> TS -> qRRHO.

    from enzyme_ts import run
    from enzyme_ts.systems.kemp_hg3 import SYSTEM
    result = run(SYSTEM, k=1)        # TSResult(dG_act=..., dG_rxn=..., ts_valid=...)

`run` is idempotent about prep: it builds real_noep.pdb only if absent.
"""
from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
import numpy as np
from ase.constraints import FixAtoms
from ase.io import write, read as _read

from . import config
from .optim import relax
from .engine import calculator as calc_mod
from .pes import scan as scan_mod, saddle, search
from .build import substrate as sub_mod, protein as prot_mod, carve as carve_mod
from . import thermo

KCAL = calc_mod.KCAL_MOL_PER_EV


@dataclass
class TSResult:
    tag: str
    k: int
    charge: int
    dG_act: float
    dG_rxn: float
    dE_act: float
    n_imag_ts: int
    ts_valid: bool
    n_free: int
    n_total: int
    exp_dG: float | None
    reactant_geom: dict
    product_geom: dict
    imag_cm: list


def _workdir(system):
    base = system.workdir or config.workroot()
    d = os.path.join(base, "build", system.tag)
    os.makedirs(d, exist_ok=True)
    return d


def prepare(system, outdir):
    """Model substrate onto TSA + protonate protein + merge -> real_noep.pdb."""
    src = system.pdb if os.path.isabs(system.pdb) else os.path.join(config.workroot(), system.pdb)
    _, placed = sub_mod.model_substrate(src, system.chains[0], system.mechanism, outdir)
    prot = prot_mod.prep_protein(src, system.chains, outdir,
                                 method=system.protonation_method, pH=system.pH,
                                 overrides=system.catalytic_states,
                                 reduce_flips=system.reduce_flips,
                                 drop_resn=system.drop_resn,
                                 tsa_resname=system.mechanism.tsa_resname)
    return prot_mod.merge(prot, placed, outdir)


def _coord_geom(atoms, mech, idx):
    g = {}
    for c in mech.coords:
        if c[0] in ("break", "form"):
            g[f"{c[1]}-{c[2]}"] = float(atoms.get_distance(idx[c[1]], idx[c[2]]))
        elif c[0] == "proton":
            g[f"{c[1]}-{c[2]}"] = float(atoms.get_distance(idx[c[1]], idx[c[2]]))
            g[f"{c[2]}..{c[3]}"] = float(atoms.get_distance(idx[c[2]], idx[c[3]]))
    return g


def run(system, k=1, method="scan_ts", fmax=0.05, model=None, do_prep=None):
    model = model or config.DEFAULT_MODEL
    calc_mod.set_engine(model)   # validates the model's engine is runnable here
    outdir = _workdir(system)
    real = os.path.join(outdir, "real_noep.pdb")
    if do_prep is None:
        do_prep = not os.path.exists(real)
    if do_prep:
        real = prepare(system, outdir)

    mech = system.mechanism
    out_prefix = os.path.join(outdir, f"dg_k{k}")
    b = carve_mod.carve_active_site(system, real, k, out_prefix)
    q, s, frozen, idx = b["charge"], b["spin"], b["frozen"], b["idx"]

    def C(): return calc_mod.get_calculator(config.DEFAULT_TASK, model)

    dkind, da, db, d_end, step = mech.drive
    ia, ib = idx[da], idx[db]

    # reactant
    reactant = b["atoms"].copy(); reactant.info.update(charge=int(q), spin=int(s))
    reactant.calc = C(); reactant.set_constraint(FixAtoms(indices=frozen))
    relax(reactant, fmax=fmax, steps=400, trajectory=f"{out_prefix}_reactant_relax.traj")
    write(f"{out_prefix}_reactant_seed.xyz", reactant)
    d0 = reactant.get_distance(ia, ib)

    # relaxed scan of the driving coordinate (proton/other coords relay freely)
    if dkind == "break":
        dists = sorted({round(float(x), 3) for x in np.arange(d0, d_end + 1e-6, step)})
    else:
        dists = sorted({round(float(x), 3) for x in np.arange(d0, d_end - 1e-6, -step)}, reverse=True)
    energies, geoms = scan_mod.scan(reactant, q, s, model, ia, ib, dists, frozen=frozen)
    i_top = int(np.argmax(energies))
    print(f"[run] {system.tag} k={k}: scan max {(energies[i_top]-energies[0])*KCAL:+.1f} kcal/mol at {dists[i_top]:.2f} A")

    # product
    product = geoms[-1].copy(); product.info.update(charge=int(q), spin=int(s))
    product.set_constraint(FixAtoms(indices=frozen)); product.calc = C()
    relax(product, fmax=fmax, steps=400, trajectory=f"{out_prefix}_product_relax.traj")

    # TS search
    seed_bonds = [(ia, ib)] + [(idx[c[1]], idx[c[2]]) for c in mech.coords
                               if c[0] in ("break", "form") and (c[1], c[2]) != (da, db)]
    diag = search.find_ts(method, reactant=reactant, product=product, scan_geoms=geoms,
                          scan_energies=energies, i_top=i_top, q=q, s=s, model=model,
                          frozen=frozen, out_prefix=out_prefix, label=f"{system.tag} k={k}",
                          seed_bonds=seed_bonds)

    # assign reactant/product minima by geometry (saddle labels by energy, often swapped)
    rp, pp = f"{out_prefix}_reactant.xyz", f"{out_prefix}_product.xyz"
    a1, a2 = _read(rp), _read(pp)
    d1, d2 = a1.get_distance(ia, ib), a2.get_distance(ia, ib)
    react_is_1 = (abs(d1 - d0) < abs(d2 - d0)) if dkind == "break" else (d1 > d2)
    react_path, prod_path = (rp, pp) if react_is_1 else (pp, rp)

    def G(path):
        a = _read(path); a.info.update(charge=int(q), spin=int(s))
        a.calc = C(); a.set_constraint(FixAtoms(indices=frozen))
        e = a.get_potential_energy()
        fr, *_ = saddle.confirm(a, q, s, model)
        return thermo.gibbs(a, q, s, model, T=config.DEFAULT_T, e_elec=e, freqs_cm=fr, method="qrrho"), e
    (gR, eR), (gT, eT), (gP, eP) = G(react_path), G(f"{out_prefix}_ts.xyz"), G(prod_path)

    res = TSResult(
        tag=system.tag, k=k, charge=int(q),
        dG_act=(gT["G"] - gR["G"]) * KCAL, dG_rxn=(gP["G"] - gR["G"]) * KCAL,
        dE_act=(eT - eR) * KCAL, n_imag_ts=gT["n_imag"], ts_valid=bool(gT["n_imag"] == 1),
        n_free=b["n_free"], n_total=b["n_total"], exp_dG=system.exp_dG,
        reactant_geom=_coord_geom(_read(react_path), mech, idx),
        product_geom=_coord_geom(_read(prod_path), mech, idx),
        imag_cm=diag.get("imag_cm", []))
    json.dump(asdict(res), open(os.path.join(outdir, f"result_k{k}.json"), "w"), indent=2)
    print("=" * 62)
    print(f"  {system.tag} k={k}: dG_act = {res.dG_act:.1f} (exp {res.exp_dG}), "
          f"dG_rxn = {res.dG_rxn:.1f}, TS valid = {res.ts_valid} (n_imag {res.n_imag_ts})")
    print("=" * 62)
    return res
