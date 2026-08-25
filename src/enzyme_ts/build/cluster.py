"""Carve an active-site QM cluster from an AMBER-protonated PDB.

- Select whole ligand/ion residues + a chosen set of protein residues.
- Cap broken peptide bonds with H link atoms (replace the excluded backbone
  neighbor C/N by an H at ~1.0 A along the original bond).
- Assign integer total charge from AMBER protonation-state resnames.
- Mark frozen atoms = everything beyond `free_radius` of the reaction center,
  so the pocket scaffold holds its crystallographic/NAC geometry.

Pure stdlib + numpy; no external structure libs.
"""
from __future__ import annotations
import numpy as np

RESCHG = {"ARG": +1, "LYS": +1, "HIP": +1, "ASP": -1, "GLU": -1,
          "HIE": 0, "HID": 0, "HIS": 0, "ASN": 0, "GLN": 0, "SER": 0,
          "THR": 0, "TYR": 0, "CYS": 0, "CYX": 0, "GLY": 0, "ALA": 0,
          "VAL": 0, "LEU": 0, "ILE": 0, "PRO": 0, "PHE": 0, "TRP": 0,
          "MET": 0, "WAT": 0, "HOH": 0,
          # protonated/deprotonated Amber variants -- easy to encounter at a
          # catalytic acid/base and silently wrong (defaulting to 0) if missing:
          "ASH": 0, "GLH": 0,      # neutral (protonated) Asp/Glu
          "CYM": -1,               # deprotonated Cys thiolate
          "LYN": 0}                # neutral (deprotonated) Lys
# Residue names not in RESCHG (nonstandard/unrecognized) silently default to
# neutral via RESCHG.get(..., 0) in carve() -- fine for genuinely neutral
# residues, a silent bug for anything else. Any run on a new system should
# cross-check the returned `charge` against an independent source (e.g. the
# propka/pypka output used upstream) rather than trust this table blindly.
BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA"}


from ase.data import atomic_numbers
_TWO = ("Mg", "Cl", "Na", "Fe", "Zn", "Mn", "Ca", "Br")


def _element(line):
    raw = line[76:78].strip()
    if raw and raw.title() in atomic_numbers:
        return raw.title()
    nm = line[12:16].strip()
    if nm[:2].title() in _TWO:
        return nm[:2].title()
    for ch in nm:
        if ch.isalpha():
            return ch.upper()
    return nm[0]


def read_pdb(path):
    A = []
    for l in open(path):
        if l[:6] in ("ATOM  ", "HETATM"):
            A.append(dict(name=l[12:16].strip(), res=l[17:20].strip(),
                          chain=l[21], resi=int(l[22:26]),
                          xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]),
                          elem=_element(l)))
    return A


def carve(pdb, ligands, protein_resids, lig_charges, reaction_center,
          free_radius=3.5, out_prefix="cluster"):
    A = read_pdb(pdb)
    sel_keys = set(protein_resids)                       # (chain,resi)
    # collect selected atoms
    keep = []
    for i, a in enumerate(A):
        if a["res"] in ligands:
            keep.append(i)
        elif (a["chain"], a["resi"]) in sel_keys:
            keep.append(i)
    keepset = set(keep)

    # peptide-bond caps for protein residues
    caps = []  # (element, xyz)
    m1_atoms = []  # global indices (into A) of the MM-side atom immediately across
                   # each cut -- the "M1" atom in ONIOM boundary nomenclature. Its
                   # charge must be excluded/zeroed from any external point-charge
                   # embedding field (it sits ~1 A from the cap H and would otherwise
                   # over-polarize the region through the link atom).
    by_res = {}
    for i, a in enumerate(A):
        by_res.setdefault((a["chain"], a["resi"]), {})[a["name"]] = (i, a["xyz"])
    for (ch, ri) in protein_resids:
        res = by_res.get((ch, ri), {})
        # N-side: bond N(ri)-C(ri-1); cap if previous residue not selected
        if "N" in res and (ch, ri - 1) not in sel_keys:
            prev = by_res.get((ch, ri - 1), {})
            if "C" in prev:
                nN = res["N"][1]; cC = prev["C"][1]
                v = cC - nN; v /= np.linalg.norm(v)
                caps.append(("H", nN + 1.01 * v))
                m1_atoms.append(prev["C"][0])
        # C-side: bond C(ri)-N(ri+1); cap if next residue not selected
        if "C" in res and (ch, ri + 1) not in sel_keys:
            nxt = by_res.get((ch, ri + 1), {})
            if "N" in nxt:
                cC = res["C"][1]; nN = nxt["N"][1]
                v = nN - cC; v /= np.linalg.norm(v)
                caps.append(("H", cC + 1.09 * v))
                m1_atoms.append(nxt["N"][0])

    # build atom arrays
    elems = [A[i]["elem"] for i in keep] + [c[0] for c in caps]
    xyz = [A[i]["xyz"] for i in keep] + [c[1] for c in caps]
    xyz = np.array(xyz)
    # local index map for original atoms
    orig_local = {gi: li for li, gi in enumerate(keep)}

    # charge: sum per-residue once (ligands/ions from lig_charges, protein from RESCHG)
    q = 0
    seen_res = set()
    for i in keep:
        a = A[i]; key = (a["chain"], a["resi"], a["res"])
        if key in seen_res:
            continue
        seen_res.add(key)
        if a["res"] in lig_charges:
            q += lig_charges[a["res"]]
        else:
            q += RESCHG.get(a["res"], 0)

    # frozen mask: free if within free_radius of any reaction-center atom
    rc = np.array(reaction_center)
    free = []
    for li, gi in enumerate(keep):
        d = np.min(np.linalg.norm(A[gi]["xyz"] - rc, axis=1))
        free.append(d <= free_radius)
    free += [False] * len(caps)        # caps always frozen
    free = np.array(free)
    frozen_idx = list(np.where(~free)[0])

    return dict(elements=elems, xyz=xyz, charge=int(q), frozen=frozen_idx,
                n_free=int(free.sum()), n_total=len(elems),
                orig_local=orig_local, keep=keep, ncaps=len(caps),
                m1_atoms=m1_atoms)


def find_local(A, keep, res, name):
    """local index of a given (resname, atomname) atom in the carved cluster."""
    for li, gi in enumerate(keep):
        if A[gi]["res"] == res and A[gi]["name"] == name:
            return li
    return None
