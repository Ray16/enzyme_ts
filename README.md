# enzyme_ts

Config-driven **enzyme transition-state & Gibbs-barrier modeling** on frozen-boundary
ML-potential (UMA / OMol25) active-site clusters.

Given a crystal structure with a bound transition-state analog (or substrate) and a
short declarative mechanism, the package:

1. **models the real substrate** onto the bound analog by scaffold Kabsch alignment,
2. **protonates** the protein (pdb4amber + tleap/ff14SB) and merges the substrate,
3. **carves** a Voronoi active-site cluster (catalytic residues found structurally),
4. builds reactant/product **endpoints by a relaxed scan** of the driving coordinate,
5. locates the **transition state** (Sella + batched Hessian + mode-descent),
6. reports **quasi-RRHO Gibbs** ΔG‡ / ΔG_rxn with a mechanism-consistency check.

## The only per-enzyme input: a `System`

Everything mechanical is generic. You write the *reasoned* chemistry — what the
substrate is, how it maps onto the bound analog, and which atoms form / break /
transfer:

```python
from enzyme_ts import Mechanism, System, lig, lig_H, prot_near, bond, proton, run

MECH = Mechanism(
    smiles="[O-][N+](=O)c1ccc2onc(c2c1)",          # real substrate
    charge=0, tsa_resname="6NT",                    # bound analog to align onto
    scaffold=[("[o]:[n]:[c]", 2, "N3"),            # (ligand SMARTS, idx, TSA atom)
              ("[o]:[n]:[c]", 1, "N2"),
              ("[o]:[n]:[c]", 0, "N1"),
              ("[o]:[n]:[c]:[c]", 3, "C3A"),
              ("[c]:[o]:[n]", 0, "C7A")],
    atoms={"C3": lig("[o]:[n]:[c]", 2), "N2": lig("[o]:[n]:[c]", 1),
           "O1": lig("[o]:[n]:[c]", 0), "H3": lig_H("C3"),
           "base": prot_near("ASP,GLU", "OD1,OD2,OE1,OE2", "H3")},
    coords=[bond("break", "N2", "O1"), proton("C3", "H3", "base")],
    drive=("break", "N2", "O1", 2.60, 0.10),        # scan N-O to 2.60 A, step 0.10
)
SYSTEM = System(tag="HG3", pdb="5rga.pdb", chains=["A"], mechanism=MECH, exp_dG=17.7)

result = run(SYSTEM, k=1)     # test; then k=2 for production
print(result.dG_act, result.exp_dG, result.ts_valid)
```

Selectors: `lig(smarts, i)`, `lig_H(role)`, `prot_near(resns, names, ref_role)`,
`prot_resid(resid, name)`. Coordinates: `bond("form"/"break", a, b)`,
`proton(donor, h, acceptor)`. Multi-atom mechanisms (e.g. serine-hydrolase
acylation: nucleophile attack + proton relay) are just more coords.

## Layout

```
engine/    calculator (cached UMA), batch inference
pes/       scan, saddle (Sella), gsm, neb, search (unified TS dispatch)
build/     substrate (model onto TSA), protein (tleap), regions (Voronoi), cluster, carve
mechanism  Mechanism/System dataclasses + atom resolvers   pipeline  run()
thermo     quasi-RRHO Gibbs      plot  ΔG level diagram      config  tool paths
systems/   worked examples (kemp_hg3, ...)
```

## Requirements

pip: `numpy scipy ase rdkit sella torch fairchem-core biopython` (+ `matplotlib`,
`pysisyphus` optional). External, configured via `ENZYME_TS_*` env vars (see
`config.py`): **AmberTools** (tleap/pdb4amber) in a conda env, and a **quantumPDB**
checkout for Voronoi selection. UMA weights are fetched by fairchem on first use.

## Method notes

Frozen-boundary whole-cluster ML potential (Himo-style): the entire carved region is
UMA, boundary atoms fixed. `k` sets the number of frozen shells around a fixed free
reactive core. Validate small (`k=1`) first, then `k=2` for production. Thermo is
quasi-RRHO with no translation/rotation (pinned cluster).
