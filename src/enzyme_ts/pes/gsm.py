"""Growing String Method (GSM) TS search via pysisyphus -- a robust alternative
to CI-NEB.

Why: CI-NEB proved both slow and HANG-PRONE on our systems (a scan-seeded band
still took 485+ oscillating climbing-image FIRE steps and then wedged). GSM
grows the string from the two endpoints INWARD, adding nodes near the MEP, so it
sidesteps bad-interpolation clashes and the pathological climbing-image
convergence. The paper we benchmark against (Ohmura 2025) also uses GSM, not NEB.

Same contract as cineb.run_cineb_to_ts: GSM supplies the TS guess, then
ts_sella.run_full refines it (Sella + mode-flatten + Hessian + mode-descent).

pysisyphus works in Bohr/Hartree; our ASE engine (UMA or xTB via uma_helper)
works in Ang/eV -- converted at the calculator boundary. Frozen cluster atoms
are passed through Geometry(freeze_atoms=...).
"""
from __future__ import annotations
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write
from ase.units import Hartree, Bohr
from ..engine import calculator as uma_helper

KCAL_PER_HARTREE = Hartree * uma_helper.KCAL_MOL_PER_EV   # eV/Hartree * kcal/mol/eV


def _imports():
    from pysisyphus.Geometry import Geometry
    from pysisyphus.cos.GrowingString import GrowingString
    from pysisyphus.optimizers.StringOptimizer import StringOptimizer
    from pysisyphus.calculators.Calculator import Calculator
    return Geometry, GrowingString, StringOptimizer, Calculator


def _calc_class(Calculator):
    class _ASEBacked(Calculator):
        """pysisyphus Calculator backed by our ASE engine (uma_helper)."""
        def __init__(self, charge, spin, model, **kw):
            super().__init__(**kw)
            self._q = int(charge); self._s = int(spin); self._model = model

        def _eval(self, atoms, coords):
            pos = np.asarray(coords, float).reshape(-1, 3) * Bohr      # Bohr -> Ang
            a = Atoms(symbols=list(atoms), positions=pos)
            a.info["charge"] = self._q; a.info["spin"] = self._s
            a.calc = uma_helper.get_calculator("omol", self._model)
            return a.get_potential_energy(), a.get_forces()           # eV, eV/Ang

        def get_forces(self, atoms, coords):
            e, f = self._eval(atoms, coords)
            return {"energy": e / Hartree,
                    "forces": (f * Bohr / Hartree).ravel()}           # -> Hartree, Hartree/Bohr

        def get_energy(self, atoms, coords):
            e, _ = self._eval(atoms, coords)
            return {"energy": e / Hartree}
    return _ASEBacked


def run_gsm_to_ts(reactant, product, charge, spin, model, results_prefix,
                  frozen=None, max_nodes=12, max_cycles=80, climb=True, **kw):
    """Grow a string reactant<->product, take the highest-energy node as the TS
    guess, and refine with ts_sella.run_full. Returns the ts_sella diag dict
    augmented with the GSM barrier/profile."""
    Geometry, GrowingString, StringOptimizer, Calculator = _imports()
    CalcCls = _calc_class(Calculator)
    frozen = list(frozen) if frozen is not None else []
    symbols = reactant.get_chemical_symbols()

    def geom(a):
        return Geometry(symbols, a.get_positions().ravel() / Bohr,     # Ang -> Bohr
                        coord_type="cart", freeze_atoms=frozen)

    # One shared calculator instance: attach to the endpoints AND hand it back
    # from calc_getter for every grown node (mlmm_toolkit path_search.py pattern;
    # a per-image fresh calc leaves grown nodes with calculator=None).
    shared = CalcCls(charge, spin, model)
    r_geom, p_geom = geom(reactant), geom(product)
    r_geom.set_calculator(shared); p_geom.set_calculator(shared)

    gs = GrowingString([r_geom, p_geom], calc_getter=(lambda: shared),
                       max_nodes=max_nodes, climb=climb)
    opt = StringOptimizer(gs, max_cycles=max_cycles, dump=False)
    print(f"[gsm] growing string: max_nodes={max_nodes} climb={climb} "
          f"(engine={uma_helper.engine_for_model(model)}, model={model})")
    opt.run()

    energies = np.array([img.energy for img in gs.images], float)
    prof = (energies - energies[0]) * KCAL_PER_HARTREE
    i_hei = int(np.argmax(prof))
    ts_pos = np.asarray(gs.images[i_hei].cart_coords, float).reshape(-1, 3) * Bohr
    ts_guess = Atoms(symbols=symbols, positions=ts_pos)
    ts_guess.info.update(charge=int(charge), spin=int(spin))
    if frozen:
        ts_guess.set_constraint(FixAtoms(indices=frozen))
    write(f"{results_prefix}_gsm_ts_guess.xyz", ts_guess)

    print(f"[gsm] {len(gs.images)} nodes | barrier = {prof[i_hei]:.1f} kcal/mol at node {i_hei}")
    print("[gsm] profile (kcal/mol): " + " ".join(f"{x:+.1f}" for x in prof))

    from . import saddle as ts_sella
    diag = ts_sella.run_full(ts_guess, charge, spin, model,
                             label="GSM -> Sella TS", results_prefix=results_prefix)
    diag["gsm_barrier"] = float(prof[i_hei])
    diag["gsm_nodes"] = int(len(gs.images))
    diag["gsm_profile"] = prof.tolist()
    diag["gsm_i_hei"] = i_hei
    return diag
