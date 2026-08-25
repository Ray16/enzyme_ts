"""Compute-engine facade: model-name -> backend dispatch (UMA or MACE-POLAR-1).

This module is the stable public surface the rest of the package imports as
`uma_helper` (historical name). It no longer contains the UMA implementation --
that moved to engine/uma.py; MACE-POLAR-1 is engine/mace_polar.py -- and simply
dispatches by model name via engine/registry.py:

    calc = get_calculator("omol", "uma-s-1p2")     # ASE calculator (single point)
    calc = get_calculator("omol", "mace-polar-1")  #   "" for MACE-POLAR-1
    KCAL_MOL_PER_EV                                 # unit conversion constant
    engine_for_model("mace-polar-1") == "mace"     # which backend a model uses

The batched force-eval path is engine/batch.py::batched_ef (also dispatched).
"""
from __future__ import annotations
import os
import torch
from ase.units import kcal, mol
from . import registry
from .registry import engine_for_model  # re-export (used by pes/gsm.py etc.)
from .. import config

# --- shared-node thread etiquette -------------------------------------------
# Each ML force eval has a non-trivial CPU-side cost (neighbor/graph build for
# ~hundreds of atoms, collation, autograd for forces). Left uncapped, torch/OMP
# grab ALL cores per job -> massive oversubscription on this shared box that
# thrashes and inflates wall-clock while the GPU sits idle. Cap to a modest
# budget (default 8). Applies to whichever engine is active (both use torch).
# Override via UMA_NUM_THREADS (kept for back-compat) or set it to 0 to disable.
_NT = int(os.environ.get("UMA_NUM_THREADS", os.environ.get("ENZYME_TS_NUM_THREADS", "8")))
if _NT > 0:
    try:
        torch.set_num_threads(_NT)
        os.environ.setdefault("OMP_NUM_THREADS", str(_NT))
        os.environ.setdefault("MKL_NUM_THREADS", str(_NT))
    except Exception:
        pass

# 1 eV expressed in kcal/mol (from ASE's own unit definitions). Engine-agnostic.
KCAL_MOL_PER_EV: float = 1.0 / (kcal / mol)

# The default model now comes from config (env ENZYME_TS_MODEL, else uma-s-1p2).
DEFAULT_MODEL = config.DEFAULT_MODEL

# Retained for the few call sites that print it; the actual engine is per-model.
ENGINE = "auto"


def set_engine(engine: str | None = None) -> None:
    """Validate that a requested engine's backend can actually run here, failing
    LOUDLY at run start rather than cryptically mid-pipeline. Pass an engine name
    ("uma"/"mace") or a model name (engine is inferred). None -> no-op."""
    if engine is None:
        return
    eng = engine if engine in registry.KNOWN_ENGINES else engine_for_model(engine)
    if not registry.available(eng):
        if eng == "mace":
            raise RuntimeError(
                "engine 'mace' (MACE-POLAR-1) is not runnable here: `mace` must be "
                "importable AND ENZYME_TS_MACE_POLAR_CKPT must point to a downloaded "
                "checkpoint. Install/point it, or select a uma-* model.")
        raise RuntimeError(f"engine {eng!r} is not available in this environment.")


def get_calculator(task_name: str = "omol",
                   model: str = DEFAULT_MODEL,
                   device: str | None = None,
                   inference_settings: str = "default"):
    """Return an ASE calculator for `model`, dispatched to its engine backend.
    Heavy weights are cached inside each adapter; the returned calculator is
    cheap and safe to create per call."""
    return registry.adapter_for_model(model).get_calculator(
        task_name, model, device, inference_settings)
