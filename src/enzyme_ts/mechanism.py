"""Declarative mechanism / system specification.

This is the ONLY part that must be reasoned per enzyme: what the substrate is, how
it maps onto the bound TS-analog, and which atoms form/break/transfer. Everything
downstream (prep, carve, TS search, Gibbs) is generic and driven by this spec.

Atom SELECTORS resolve a named role to a concrete atom:
  lig(smarts, i)                 the i-th atom of the ligand's first SMARTS match
  lig_H(role)                    the H bonded to an already-resolved ligand atom
  prot_near(resns, names, ref)   nearest protein atom (resns/names CSV) to role `ref`
  prot_resid(resid, name)        an explicit protein atom (fragile vs renumbering)

A COORDINATE is one elementary change, referencing roles:
  bond("form"/"break", a, b)     a forming or breaking bond
  proton(donor, h, acceptor)     a proton transfer donor-H ... acceptor
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ---- atom selectors (return plain tuples so specs stay serializable) ----
def lig(smarts: str, i: int): return ("lig", smarts, i)
def lig_H(role: str):         return ("lig_H", role)
def prot_near(resns: str, names: str, ref_role: str): return ("prot_near", resns, names, ref_role)
def prot_resid(resid: int, name: str): return ("prot_resid", resid, name)


# ---- reaction coordinates ----
def bond(kind: str, a: str, b: str):
    assert kind in ("form", "break")
    return (kind, a, b)
def proton(donor: str, h: str, acceptor: str):
    return ("proton", donor, h, acceptor)


@dataclass
class Mechanism:
    """The reasoned chemistry of one enzyme step."""
    smiles: str                              # real substrate (neutral/charged) as SMILES
    charge: int                              # substrate formal charge
    tsa_resname: str                         # bound TS-analog (or substrate) residue name to align onto
    scaffold: list[tuple]                    # [(lig_smarts, idx, tsa_atom_name), ...] Kabsch anchors
    atoms: dict[str, tuple]                  # role -> selector
    coords: list[tuple]                      # elementary changes (bond/proton)
    drive: tuple                             # (kind, roleA, roleB, d_end, step): scan coordinate
    resname: str = "SUB"


@dataclass
class System:
    """A concrete enzyme + structure + mechanism, ready to run."""
    tag: str
    pdb: str                                 # crystal PDB path (with bound TSA) or fetched id
    chains: list[str]
    mechanism: Mechanism
    exp_dG: float | None = None              # experimental Eyring barrier for validation
    protonation_method: str = "pypka"        # "pypka" (PB/MC determinant) | "tleap" (defaults)
    pH: float = 7.0
    reduce_flips: bool = True                 # run `reduce` for Asn/Gln/His flips first
    catalytic_states: dict[int, str] = field(default_factory=dict)  # mechanism-forced {resid: RESNAME}
    drop_resn: list[str] = field(default_factory=list)          # het residues to strip
    sphere_count: int = 4
    free_radius: float = 3.5
    workdir: str | None = None               # base dir; defaults to config.workroot()


# ---- resolvers (used by build.substrate for ligand, build.cluster for protein) ----
def resolve_ligand_atom(mol, sel, resolved: dict):
    """Resolve a ligand-atom selector against an RDKit mol; returns an atom index."""
    from rdkit import Chem
    if sel[0] == "lig":
        _, smarts, i = sel
        m = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        if not m:
            raise ValueError(f"SMARTS {smarts!r} did not match the ligand")
        return m[0][i]
    if sel[0] == "lig_H":
        base = resolved[sel[1]]
        return next(nb.GetIdx() for nb in mol.GetAtomWithIdx(base).GetNeighbors()
                    if nb.GetSymbol() == "H")
    return None                              # not a ligand selector (protein: resolved later)


def resolve_protein_atom(sel, atoms, ref_xyz_by_role):
    """Resolve a protein-atom selector against a pdb atom-dict list."""
    import numpy as np
    if sel[0] == "prot_near":
        _, resns, names, ref_role = sel
        resns, names = set(resns.split(",")), set(names.split(","))
        ref = ref_xyz_by_role[ref_role]
        cand = [a for a in atoms if a["res"] in resns and a["name"] in names]
        if not cand:
            raise ValueError(f"no protein atom matching {resns}/{names}")
        return min(cand, key=lambda a: float(np.linalg.norm(a["xyz"] - ref)))
    if sel[0] == "prot_resid":
        return next(a for a in atoms if a["resi"] == sel[1] and a["name"] == sel[2])
    raise ValueError(f"unknown protein selector {sel!r}")
