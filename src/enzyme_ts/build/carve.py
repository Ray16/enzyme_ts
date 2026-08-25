"""Carve the active-site cluster and resolve the mechanism's reacting atoms to
cluster indices. Combines Voronoi shell selection (regions) + capping/charge
(cluster) + the mechanism spec (ligand roles already named; protein roles resolved
structurally here, so it is robust to tleap's 1..N renumbering)."""
from __future__ import annotations
import os, json
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from . import cluster as CC
from . import regions as RS
from .. import mechanism as M


def carve_active_site(system, real_pdb, k, out_prefix):
    outdir = os.path.dirname(real_pdb)
    A = CC.read_pdb(real_pdb)
    rj = json.load(open(os.path.join(outdir, "reacting_atoms.json")))
    resn = rj.get("resname", "SUB")
    react = rj["reacting_atoms"]

    # ligand roles (given as atom names) -> atom dicts
    atomsD = {role: next(a for a in A if a["res"] == resn and a["name"] == v)
              for role, v in react.items() if isinstance(v, str)}
    ref_xyz = {role: a["xyz"] for role, a in atomsD.items()}
    # protein roles (selectors) -> atom dicts, resolved structurally
    for role, v in react.items():
        if not isinstance(v, str):
            atomsD[role] = M.resolve_protein_atom(tuple(v), A, ref_xyz)

    # force catalytic protein residues in at any shell depth
    force = {(a["chain"], a["resi"]) for role, a in atomsD.items() if a["res"] != resn}

    prot_shells, lig_shells, center_id = RS.select_shells(
        real_pdb, center_resnames=resn, sphere_count=system.sphere_count, merge_cutoff=8.0)
    resids = set(RS.union_upto(prot_shells, k) | RS.union_upto(lig_shells, k)) | force
    sizes = RS.shell_sizes(prot_shells, lig_shells)

    rc = [atomsD[r]["xyz"] for r in atomsD]
    res = CC.carve(real_pdb, ligands=[resn], protein_resids=sorted(resids),
                   lig_charges={resn: rj["charge"]}, reaction_center=rc,
                   free_radius=system.free_radius, out_prefix=out_prefix)

    def idx_of(a):
        return int(np.argmin(np.linalg.norm(np.array(res["xyz"]) - a["xyz"], axis=1)))
    idx = {role: idx_of(a) for role, a in atomsD.items()}

    atoms = Atoms(symbols=res["elements"], positions=res["xyz"])
    atoms.set_constraint(FixAtoms(indices=res["frozen"]))
    atoms.info["charge"] = int(res["charge"])
    atoms.info["spin"] = 1
    n_elec = int(sum(atoms.get_atomic_numbers())) - atoms.info["charge"]
    if n_elec % 2 != 0:
        raise ValueError(f"[carve] {n_elec} electrons ODD (charge {res['charge']}) -- "
                         "check protonation / ligand charge.")

    print(f"[carve] {system.tag} k={k}: {res['n_total']} atoms ({res['n_free']} free), "
          f"charge={res['charge']}, forced catalytic residues {sorted(force)}")
    return dict(atoms=atoms, charge=int(res["charge"]), spin=1, idx=idx,
                frozen=res["frozen"], n_free=res["n_free"], n_total=res["n_total"],
                shell_sizes=sizes)
