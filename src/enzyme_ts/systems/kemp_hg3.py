"""Kemp eliminase HG3 (parent) -- example System spec.

Reasoned chemistry: one concerted E2-like TS -- Asp127 general base abstracts the
benzisoxazole C3-H, concerted with N2-O1 ring cleavage -> 2-cyano-nitrophenolate.
Substrate 5-nitro-1,2-benzisoxazole modeled onto the bound TS-analog 6-nitro-
benzotriazole (6NT). Scaffold anchors map the reacting 5-ring + fusion carbons:
  benzisoxazole C3<->N3 (near base), N2<->N2, O1<->N1 (near oxyanion), C3a<->C3A, C7a<->C7A.
Drive the breaking N-O bond; the C3-H proton relays to the Asp carboxylate.

Experimental: kcat 0.68/s (Privett PNAS 2012) -> dG_act 17.7 kcal/mol (Eyring, 298 K).
Structure: 5RGA = clean HG3 with bound 6NT (verified 0 seq diff vs 7K4P).
"""
from enzyme_ts import Mechanism, System, lig, lig_H, prot_near, bond, proton

MECH = Mechanism(
    smiles="[O-][N+](=O)c1ccc2onc(c2c1)",     # 5-nitro-1,2-benzisoxazole (neutral)
    charge=0,
    tsa_resname="6NT",
    scaffold=[
        ("[o]:[n]:[c]", 2, "N3"),             # C3  (acidic CH, near base)   <-> N3
        ("[o]:[n]:[c]", 1, "N2"),             # N2  (-> nitrile)             <-> N2
        ("[o]:[n]:[c]", 0, "N1"),             # O1  (-> phenolate)           <-> N1
        ("[o]:[n]:[c]:[c]", 3, "C3A"),        # C3a (fusion C bonded to C3)  <-> C3A
        ("[c]:[o]:[n]", 0, "C7A"),            # C7a (fusion C bonded to O1)  <-> C7A
    ],
    atoms={
        "C3":   lig("[o]:[n]:[c]", 2),
        "N2":   lig("[o]:[n]:[c]", 1),
        "O1":   lig("[o]:[n]:[c]", 0),
        "H3":   lig_H("C3"),
        "base": prot_near("ASP,GLU", "OD1,OD2,OE1,OE2", "H3"),   # Asp127 carboxylate
    },
    coords=[bond("break", "N2", "O1"), proton("C3", "H3", "base")],
    drive=("break", "N2", "O1", 2.60, 0.10),  # scan N-O from d0 to 2.60 A, step 0.10
)

SYSTEM = System(
    tag="HG3_pkg",
    pdb="/nfs/lambda_stor_01/homes/rzhu/uma_enzyme_ts/KempEliminase/build/5rga.pdb",
    chains=["A"],
    mechanism=MECH,
    exp_dG=17.7,
    drop_resn=["SO4", "1PE", "PG4", "GOL"],
    workdir="/nfs/lambda_stor_01/homes/rzhu/uma_enzyme_ts/KempEliminase",
)
