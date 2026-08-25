"""Unified transition-state search dispatch.

Three interchangeable strategies, all ending in Sella + batched-Hessian +
mode-descent (saddle.run_full / *_to_ts):
  "scan_ts" (default) refine the relaxed-scan energy maximum -- cheapest, matches
            standard QM-cluster practice; the scan point is already on the connected
            MEP because only the driven bond was held while everything else relaxed.
  "gsm"     grow a string between the endpoints (robust when the scan max is a poor
            guess); "neb" climbing-image NEB seeded by a two-coordinate scan.
"""
from __future__ import annotations
from ase.constraints import FixAtoms
from . import saddle


def find_ts(method, *, reactant, product, scan_geoms, scan_energies, i_top,
            q, s, model, frozen, out_prefix, label, seed_bonds=None, nimg=9):
    if method == "scan_ts":
        ts_guess = scan_geoms[i_top].copy()
        ts_guess.info.update(charge=int(q), spin=int(s))
        ts_guess.set_constraint(FixAtoms(indices=frozen))
        return saddle.run_full(ts_guess, q, s, model, label=label, results_prefix=out_prefix)
    if method == "gsm":
        from . import gsm
        return gsm.run_gsm_to_ts(reactant, product, q, s, model,
                                 results_prefix=out_prefix, frozen=frozen)
    if method == "neb":
        from . import neb
        return neb.run_cineb_to_ts(reactant, product, q, s, model, results_prefix=out_prefix,
                                   nimg=nimg, frozen=frozen, seed_bonds=seed_bonds)
    raise ValueError(f"unknown TS-search method {method!r}")
