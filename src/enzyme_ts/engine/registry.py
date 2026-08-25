"""Engine registry: route a model NAME to its compute backend.

The pipeline is no longer UMA-only. Each engine is an adapter (see engine/uma.py,
engine/mace_polar.py) exposing exactly two responsibilities:

    get_calculator(task_name, model, device, inference_settings) -> ase.Calculator
    batched_ef(atoms_list, charge, spin, model)                  -> (E_eV, [forces])

`batched_ef` is the load-bearing one: the batched Hessian, partial Hessian, and
BatchedNEB (engine/batch.py) all route through it. An engine that only exposes a
per-structure ASE calculator (no native batched predict) must STILL implement
batched_ef -- even if only as a serial loop, which is correct but slow and is
exactly the wrong place to go serial (thousands of force evals per Hessian).

Engine selection is inferred from the MODEL NAME, so every call site keeps
threading a single `model=` through and never names the engine explicitly:

    uma-*         -> "uma"   (FAIRChem / OMol25; charge/spin via atoms.info)
    mace-polar-*  -> "mace"  (MACE-POLAR-1 polarisable foundation model)

Adapters are imported lazily (fairchem and mace are both heavy and may not even
be installed in the same env), so importing this module costs nothing.
"""
from __future__ import annotations
import functools

# Longest/most-specific prefix first.
_MODEL_PREFIXES = (
    ("uma-", "uma"),
    ("mace-polar", "mace"),   # mace-polar-1, mace-polar-1-M, mace-polar-1-L
    ("mace-", "mace"),
)

KNOWN_ENGINES = ("uma", "mace")


def engine_for_model(model: str) -> str:
    """Infer the engine backend from a model name."""
    m = str(model).lower()
    for pref, eng in _MODEL_PREFIXES:
        if m.startswith(pref):
            return eng
    raise ValueError(
        f"cannot infer an engine from model name {model!r}; expected a "
        f"'uma-*' (FAIRChem/OMol) or 'mace-polar-*' (MACE-POLAR-1) model name.")


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=None)
def _adapter(engine: str):
    """Construct (and cache) an engine adapter. Construction is cheap; the heavy
    library import happens on first get_calculator/batched_ef call, not here."""
    if engine == "uma":
        from . import uma
        return uma.UMAEngine()
    if engine == "mace":
        from . import mace_polar
        return mace_polar.MACEPolarEngine()
    raise ValueError(f"unknown engine {engine!r} (known: {KNOWN_ENGINES}).")


def get_engine(engine: str):
    return _adapter(engine)


def adapter_for_model(model: str):
    return _adapter(engine_for_model(model))


def available(engine: str) -> bool:
    """True iff this engine's backend can actually run in this process/env."""
    try:
        return bool(_adapter(engine).available())
    except Exception:
        return False
