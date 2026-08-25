"""Relaxed coordinate scan to build a reaction energy profile robustly.

Drive a forming/breaking bond distance in steps; at each step freeze that one
distance and fully relax all other DOF with UMA. The resulting energy-vs-coord
curve IS the reaction energy diagram; its maximum is the TS estimate.
"""
from __future__ import annotations
import numpy as np
from ase.constraints import FixInternals, FixAtoms
from ase.neighborlist import natural_cutoffs, NeighborList
from ..engine import calculator as uma_helper
from ..optim import relax

KCAL = uma_helper.KCAL_MOL_PER_EV


def _fragment_of(atoms, j, scale=1.2):
    cut = [c * scale for c in natural_cutoffs(atoms)]
    nl = NeighborList(cut, self_interaction=False, bothways=True); nl.update(atoms)
    seen = set(); stack = [j]
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k); stack.extend(nl.get_neighbors(k)[0])
    return sorted(seen)


def set_distance_rigid(atoms, i, j, d):
    """Translate the fragment containing j so that |r_i - r_j| = d."""
    frag = _fragment_of(atoms, j)
    if i in frag:                       # i and j already bonded: move only j
        frag = [j]
    p = atoms.get_positions()
    axis = p[j] - p[i]; L = np.linalg.norm(axis); axis = axis / L
    shift = (d - L) * axis
    p[frag] += shift
    atoms.set_positions(p)


def scan(atoms, q, s, model, i, j, dists, fmax=0.05, steps=300, frozen=None):
    """Relaxed scan of the distance (i,j) over ``dists``. At each step only that one
    distance is held (FixInternals); every other DOF relaxes, so coupled coordinates
    (e.g. a proton relay riding along with a forming bond) follow the connected MEP
    naturally -- unlike a rigid endpoint build, which can teleport into a
    disconnected, spurious high-energy basin. Pass ``frozen`` (a list of atom
    indices) for a boundary-frozen cluster; those atoms are held via FixAtoms in
    ADDITION to the scanned distance (omitting them lets the frozen shell drift,
    which corrupts a carved-cluster scan)."""
    a = atoms.copy()
    a.info["charge"] = int(q); a.info["spin"] = int(s)
    frozen = list(frozen) if frozen is not None else []
    energies = []; geoms = []
    for d in dists:
        set_distance_rigid(a, i, j, d)
        a.info["charge"] = int(q); a.info["spin"] = int(s)
        a.calc = uma_helper.get_calculator("omol", model)
        cons = [FixInternals(bonds=[[float(d), [i, j]]])]
        if frozen:
            cons.append(FixAtoms(indices=frozen))
        a.set_constraint(cons)
        relax(a, fmax=fmax, steps=steps, logfile=None)
        e = a.get_potential_energy()
        a.set_constraint(FixAtoms(indices=frozen) if frozen else [])
        energies.append(e); geoms.append(a.copy())
        print(f"  d({i},{j})={d:.2f} A  E={e:.4f} eV  ({(e-energies[0])*KCAL:+.2f} kcal/mol)")
    return np.array(energies), geoms
