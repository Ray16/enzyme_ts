"""enzyme_ts -- config-driven enzyme transition-state & Gibbs-barrier modeling.

Frozen-boundary ML-potential (UMA/OMol) active-site cluster pipeline:
model the real substrate onto a bound TS-analog, protonate, carve a Voronoi
active-site cluster, build endpoints by a relaxed scan, locate the saddle
(Sella + batched Hessian + mode-descent), and report quasi-RRHO Gibbs barriers.

The only per-enzyme input is a declarative `System` (see enzyme_ts.mechanism and
the examples in enzyme_ts.systems):

    from enzyme_ts import run
    from enzyme_ts.systems.kemp_hg3 import SYSTEM
    result = run(SYSTEM, k=1)
    print(result.dG_act, result.exp_dG)
"""
from .mechanism import (Mechanism, System, lig, lig_H, prot_near, prot_resid,
                        bond, proton)
from .pipeline import run, prepare, TSResult

__all__ = ["run", "prepare", "TSResult", "Mechanism", "System",
           "lig", "lig_H", "prot_near", "prot_resid", "bond", "proton"]
__version__ = "0.1.0"
