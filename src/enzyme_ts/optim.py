"""Local geometry-optimizer policy for minimizations on an ML-potential PES.

LBFGS (not BFGS) with a capped max step is the project default for every
minimization (reactant / product / relaxed-scan / IRC endpoints):

  * LBFGS -- limited-memory quasi-Newton. At our cluster sizes (~50-300 atoms)
    the memory argument is moot; the reason to prefer it over full-Hessian BFGS
    is robustness on a slightly rough ML PES -- its short curvature memory
    recovers from a bad step instead of corrupting one dense accumulated
    Hessian, and it takes larger, cheaper steps into the basin.
  * maxstep (default 0.2 A) -- an UNCAPPED quasi-Newton step on a fresh ML PES is
    the main cause of blow-ups; capping it matters more than the LBFGS-vs-BFGS
    choice. Tighten to ~0.1 A for delicate / interpolated geometries.
  * linesearch=True -> LBFGSLineSearch: buys robustness against near-convergence
    oscillation at the cost of extra force evals per step.

This is ONLY for minimizations. The saddle search is Sella; the NEB band
optimizer stays FIRE (very hard to break on a noisy band, and also the
recommended fallback if LBFGS misbehaves). Engine-agnostic: it only touches the
ASE calculator already attached to `atoms`, so it works identically for UMA and
MACE-POLAR-1.
"""
from __future__ import annotations
from ase.optimize import LBFGS, LBFGSLineSearch

DEFAULT_MAXSTEP = 0.2


def relax(atoms, *, fmax=0.05, steps=400, maxstep=DEFAULT_MAXSTEP,
          logfile="-", trajectory=None, linesearch=False):
    """LBFGS minimization with a capped step. Returns the (in-place) Atoms.

    Mirrors the previous `BFGS(atoms, ...).run(fmax, steps)` call pattern so the
    swap is one-for-one at every call site.
    """
    if linesearch:
        opt = LBFGSLineSearch(atoms, logfile=logfile, trajectory=trajectory)
    else:
        opt = LBFGS(atoms, logfile=logfile, trajectory=trajectory, maxstep=maxstep)
    opt.run(fmax=fmax, steps=steps)
    return atoms
