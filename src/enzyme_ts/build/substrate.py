"""Model the real substrate onto a bound TS-analog (or substrate) by scaffold Kabsch.

The bound analog marks a near-attack/TS pose in the crystal; superimposing the real
substrate's shared scaffold onto it yields a physically meaningful reactant pose
without docking. Generalizes the per-system prep_*.py modeling step.
"""
from __future__ import annotations
import os, json
import numpy as np
from .. import mechanism as M


def _parse_pdb(path):
    A = []
    for l in open(path):
        if l[:6] in ("ATOM  ", "HETATM"):
            A.append(dict(name=l[12:16].strip(), res=l[17:20].strip(), ch=l[21],
                          xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return A


def _kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, P.mean(0), Q.mean(0)


def model_substrate(src_pdb, chain, mech: "M.Mechanism", outdir):
    """Build the substrate, align onto the TSA, write substrate_placed.pdb +
    reacting_atoms.json. Returns (rmsd, placed_pdb_path)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    A = _parse_pdb(src_pdb)
    tsa = {a["name"]: a for a in A if a["res"] == mech.tsa_resname and a["ch"] == chain}
    if not tsa:
        raise SystemExit(f"[substrate] no TSA {mech.tsa_resname} in chain {chain} of {src_pdb}")

    mol = Chem.MolFromSmiles(mech.smiles)
    if mol is None:
        raise SystemExit(f"[substrate] bad SMILES {mech.smiles!r}")
    # scaffold anchor atoms (before AddHs so SMARTS indices are stable)
    anchor_idx = [mol.GetSubstructMatches(Chem.MolFromSmarts(sm))[0][i]
                  for sm, i, _ in mech.scaffold]
    tsa_names = [t for _, _, t in mech.scaffold]

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    allpos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    P = allpos[anchor_idx]
    Q = np.array([tsa[n]["xyz"] for n in tsa_names])
    R, pc, qc = _kabsch(P, Q)
    newpos = (allpos - pc) @ R.T + qc
    rmsd = float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum() / len(P)))

    def aname(i): return f"{mol.GetAtomWithIdx(i).GetSymbol()}{i:02d}"
    placed = os.path.join(outdir, "substrate_placed.pdb")
    with open(placed, "w") as fh:
        for i in range(mol.GetNumAtoms()):
            x, y, z = newpos[i]
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            fh.write(f"HETATM{i+1:>5} {aname(i):<4} {mech.resname} B 301    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {sym:>2}\n")
        fh.write("END\n")

    # resolve ligand-atom roles now (names); leave protein roles as selectors for carve
    react, resolved = {}, {}
    for role, sel in mech.atoms.items():
        j = M.resolve_ligand_atom(mol, sel, resolved)
        if j is not None:
            resolved[role] = j
            react[role] = aname(j)
        else:
            react[role] = list(sel)          # protein selector, resolved against merged pdb later
    json.dump({"reacting_atoms": react, "smiles": mech.smiles, "charge": mech.charge,
               "resname": mech.resname, "scaffold_rmsd": rmsd},
              open(os.path.join(outdir, "reacting_atoms.json"), "w"), indent=2)
    print(f"[substrate] {mech.resname} aligned onto {mech.tsa_resname} (scaffold RMSD {rmsd:.2f} A)")
    return rmsd, placed
