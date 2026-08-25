"""MACE-POLAR-1 engine adapter (polarisable electrostatic MACE foundation model).

MACE-POLAR-1 (arXiv 2602.19411) augments MACE with a non-self-consistent
polarisable field and enforces global charge/spin via learnable Fukui
equilibration -- so it takes total charge Q and spin S as explicit inputs, and
is trained on OMol25 at wB97M-V (the same data family as UMA/OMol25). API and
weights verified against the official docs:
  https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html
  weights (ASL license): github.com/ACEsuit/mace-foundations (release mace_polar_1)

Install (SEPARATE env -- mace-torch pins e3nn==0.4.4, which conflicts with the
e3nn>=0.5 that fairchem/UMA needs, so MACE and UMA CANNOT share an env; engine
selection is therefore launch-time, i.e. run the pipeline under whichever env):
  pip install "mace-torch>=0.3.16"
  pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0

Charge/spin/field are passed through atoms.info -- EXACTLY the keys this pipeline
already uses (atoms.info['charge'], ['spin']) plus an optional 'external_field':
  atoms.info["charge"] = 0        # total charge
  atoms.info["spin"]   = 1        # total spin
  atoms.info["external_field"] = [0.0, 0.0, 0.0]

Model names: this pipeline's 'mace-polar-1[-s|-m|-l]' map to the loader's
'polar-1-{s,m,l}' (bare 'mace-polar-1' -> medium, the standard foundation model).
Weights auto-download to ~/.cache/mace/ on first use.

default_dtype: float64 by default (matches the official example; the polarisable
equilibration is safer in double precision). Override to float32 for speed on
V100/T4 via ENZYME_TS_MACE_DTYPE=float32 AFTER a float64-vs-float32 barrier check.

PERFORMANCE NOTE (see the parallelization follow-up): batched_ef is a SERIAL loop
for now -- the batched Hessian / BatchedNEB feed it hundreds/thousands of
displaced structures, so this is the wrong place to be serial. MACE is a GNN and
evaluates batched graphs natively; the real fix is a batched forward over the
displacement set (torch_geometric batch), not a per-structure ASE calc.
"""
from __future__ import annotations
import functools
import os
import numpy as np
from . import registry

_DTYPE = os.environ.get("ENZYME_TS_MACE_DTYPE", "float64")
_FIELD = [0.0, 0.0, 0.0]
_warned_serial = False

# enzyme_ts model name -> mace_polar loader checkpoint name
_MODEL_MAP = {
    "mace-polar-1": "polar-1-m",
    "mace-polar-1-s": "polar-1-s",
    "mace-polar-1-m": "polar-1-m",
    "mace-polar-1-l": "polar-1-l",
}


def _loader_name(model: str) -> str:
    key = str(model).lower()
    if key not in _MODEL_MAP:
        raise ValueError(
            f"unknown MACE-POLAR model {model!r}; expected one of {sorted(_MODEL_MAP)}.")
    return _MODEL_MAP[key]


@functools.lru_cache(maxsize=4)
def _load_calc(loader_name: str, device: str, dtype: str):
    """Load and cache the polarisable calculator (weights auto-download to
    ~/.cache/mace/ on first use)."""
    from mace.calculators import mace_polar
    return mace_polar(model=loader_name, device=device, default_dtype=dtype)


class MACEPolarEngine:
    name = "mace"

    def available(self) -> bool:
        """True iff the MACE-POLAR runtime is importable in this process. No
        checkpoint env needed -- the loader auto-downloads by model name."""
        try:
            from mace.calculators import mace_polar  # noqa: F401
            import graph_longrange  # noqa: F401  (provided by graph_electrostatics)
            return True
        except Exception:
            return False

    def _require(self):
        try:
            from mace.calculators import mace_polar  # noqa: F401
            import graph_longrange  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "MACE-POLAR-1 selected but its runtime is not importable here. "
                "Install in a DEDICATED env (it pins e3nn==0.4.4, incompatible with "
                "UMA/fairchem):\n"
                "  pip install 'mace-torch>=0.3.16'\n"
                "  pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0\n"
                "then run the pipeline under that env.") from e

    def get_calculator(self, task_name="omol", model="mace-polar-1",
                       device=None, inference_settings="default"):
        """ASE calculator for the configured MACE-POLAR-1 model. task_name /
        inference_settings are UMA concepts and are ignored."""
        self._require()
        dev = registry.resolve_device(device)
        return _load_calc(_loader_name(model), dev, _DTYPE)

    def _tag(self, atoms, charge, spin):
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(spin)
        atoms.info.setdefault("external_field", list(_FIELD))
        return atoms

    def batched_ef(self, atoms_list, charge, spin, model="mace-polar-1",
                   task_name="omol", device=None):
        """Energies (eV) and forces for a list of Atoms.

        SERIAL fallback (one single-point per structure) -- correct but slow; see
        the module header. Replace with a batched MACE forward for Hessian speed.
        """
        global _warned_serial
        if not _warned_serial:
            print("[mace-polar] NOTE: batched_ef is running SERIALLY (one single-point "
                  "per structure); batched Hessian/NEB will be slow until MACE's native "
                  "batched forward is wired in. See engine/mace_polar.py.")
            _warned_serial = True
        calc = self.get_calculator(task_name, model, device)
        E, F = [], []
        for at in atoms_list:
            a = self._tag(at.copy(), charge, spin)
            a.calc = calc
            E.append(float(a.get_potential_energy()))
            F.append(np.asarray(a.get_forces(), float).copy())
        return np.asarray(E), F
