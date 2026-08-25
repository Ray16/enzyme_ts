"""Active-site region selection via Voronoi-graph contact shells.

Wraps quantumPDB's Voronoi-tessellation cluster-sphere algorithm (Kastner,
Luo, Ho, Reinhardt, Keys & Kulik, "QuantumPDB: A Workflow for High-Throughput
Quantum Cluster Model Generation from Protein Structures", J. Chem. Inf.
Model. 2026, 66, 6011-6026; github.com/davidkastner/quantumPDB, MIT license,
vendored read-only at tools/quantumPDB/) to decide WHICH protein residues
belong in the active-site region, and at what shell (graph-distance from the
center), instead of a single Euclidean-distance cutoff.

Why this over a plain sphere: a residue can sit just outside an arbitrary
Euclidean radius while still being in direct van-der-Waals contact with the
active site (common in non-spherical pockets -- the QuantumPDB paper cites
acetylcholinesterase's "gorge" and carbonic anhydrase's "cone" as cases where
a sphere misses or over-includes residues). Voronoi ridges between atoms
define an actual contact graph; shell 1 = residues touching the center,
shell 2 = residues touching shell 1, etc. -- topology, not radius, beyond the
first shell.

This module intentionally reuses ONLY the shell-topology part of
quantumPDB (qp.cluster.spheres: voronoi / get_center_residues /
get_next_neighbors). Charge assignment, capping, and the frozen/free split
stay in carve_cluster.py, which is Amber-protonation-state aware (ASH/GLH/
HIP/HIE/HID/CYX/CYM/LYN reflect explicit protonation via resname);
quantumPDB's own compute_charge() assumes a different convention (fixed
resname + explicit named-hydrogen presence) that would silently mis-assign
charge on Amber-prepped inputs, so it is not used here.

Usage:
    shells = select_shells("real_noep.pdb", center_resnames="MG",
                            sphere_count=3, merge_cutoff=8.0)
    # shells[0] = center residues, shells[1] = first coordination shell, ...
    protein_resids = union_upto(shells, k=2)   # (chain, resid) tuples
"""
from __future__ import annotations
import os
import sys

# quantumPDB provides the Voronoi active-site selection (qp.cluster.spheres). Its
# location is configurable: $ENZYME_TS_QUANTUMPDB, else the repo's tools/quantumPDB.
from ..config import quantumpdb_path as _qpdb_path
_QPDB = _qpdb_path()
if _QPDB and _QPDB not in sys.path:
    sys.path.insert(0, _QPDB)

from Bio.PDB import PDBParser
from Bio.PDB import Polypeptide
from qp.cluster.spheres import CenterResidue, voronoi, get_center_residues, get_next_neighbors


def _res_key(res):
    full_id = res.get_full_id()
    chain = full_id[2]
    resi = full_id[3][1]
    return (chain, resi)


def select_shells(pdb_path, center_resnames, sphere_count=3, first_sphere_radius=4.0,
                   merge_cutoff=8.0, ligand_resnames=(), include_ligands=2,
                   smooth_method="box_plot", **smooth_params):
    """Return (protein_shells, ligand_shells, metal_id) for the merged center(s).

    protein_shells / ligand_shells: list of sets of (chain, resid) tuples,
    index 0 = center residue(s) themselves, index i = the i-th coordination
    shell outward in the Voronoi contact graph (index 0..sphere_count).
    ligand_shells holds non-polypeptide residues (waters, cofactors) pulled
    in at each shell; the caller's own reaction ligands (e.g. ATP/substrate)
    should still be forced into the carve via carve_cluster.carve()'s
    explicit `ligands=` argument regardless of what shell they land in here.

    If more than one center is found (e.g. two Mg not within merge_cutoff of
    each other), only the largest merged center cluster is returned --
    inspect quantumPDB's get_center_residues() directly if you need all of
    them.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_path)
    model = structure[0]

    center_def = CenterResidue(center_resnames)
    neighbors = voronoi(model, center_def, list(ligand_resnames), smooth_method,
                        "/tmp", **smooth_params)
    centers = get_center_residues(model, center_def, merge_cutoff)
    if not centers:
        raise ValueError(f"No residues matching center definition {center_resnames!r} found in {pdb_path}")
    center = max(centers, key=len)   # the merged cluster with the most center residues

    metal_id, _seen, spheres = get_next_neighbors(
        center, neighbors, sphere_count, list(ligand_resnames),
        first_sphere_radius=first_sphere_radius, smooth_method=smooth_method,
        include_ligands=include_ligands, **smooth_params)

    protein_shells, ligand_shells = [], []
    for sphere in spheres:
        prot, lig = set(), set()
        for res in sphere:
            key = _res_key(res)
            if Polypeptide.is_aa(res):
                prot.add(key)
            else:
                lig.add(key)
        protein_shells.append(prot)
        ligand_shells.append(lig)
    return protein_shells, ligand_shells, metal_id


def union_upto(shells, k):
    """Union of shells[0..k] inclusive (residues within k contact-hops of center)."""
    out = set()
    for s in shells[:k + 1]:
        out |= s
    return out


def shell_sizes(protein_shells, ligand_shells):
    return [(len(p), len(l)) for p, l in zip(protein_shells, ligand_shells)]
