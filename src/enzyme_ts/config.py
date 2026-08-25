"""Central configuration for enzyme_ts: external-tool locations and defaults.

Nothing here is hardcoded to a user path; every location is an environment variable
with a sensible fallback, so the package is portable. Override via:
  ENZYME_TS_QUANTUMPDB   path to a quantumPDB checkout (Voronoi active-site selection)
  ENZYME_TS_AMBER_ENV    name of the conda env that provides tleap/pdb4amber (AmberTools)
  ENZYME_TS_CONDA_SH     path to conda's profile.d/conda.sh (for activating that env)
  ENZYME_TS_MODEL        default UMA model name (uma-s-1p2)
  ENZYME_TS_WORKROOT     base directory for build/<tag> working dirs (default: cwd)
"""
from __future__ import annotations
import os

DEFAULT_MODEL = os.environ.get("ENZYME_TS_MODEL", "uma-s-1p2")
DEFAULT_TASK = "omol"
DEFAULT_T = 298.15


def quantumpdb_path() -> str | None:
    """Location of the quantumPDB package (provides qp.cluster.spheres)."""
    p = os.environ.get("ENZYME_TS_QUANTUMPDB")
    if p:
        return p
    # legacy fallback: the tools/quantumPDB in the original workspace, if present
    legacy = "/nfs/lambda_stor_01/homes/rzhu/uma_enzyme_ts/tools/quantumPDB"
    return legacy if os.path.isdir(legacy) else None


# --- external tools -----------------------------------------------------------
# Single-env design: tleap/pdb4amber/reduce are expected ON PATH and pypKa is
# importable in the SAME env (see environment.yml -> `ts_finder`). Each is still
# overridable for a split-env setup, and the code falls back to a subprocess env
# only if a tool is genuinely absent in-process.

def tleap_bin() -> str:
    return os.environ.get("ENZYME_TS_TLEAP", "tleap")


def pdb4amber_bin() -> str:
    return os.environ.get("ENZYME_TS_PDB4AMBER", "pdb4amber")


def reduce_bin() -> str:
    return os.environ.get("ENZYME_TS_REDUCE", "reduce")


def pypka_python() -> str | None:
    """Fallback interpreter for pypKa IF it is not importable in the current env
    (split-env setups only). Empty by default -> use in-process `import pypka`."""
    return os.environ.get("ENZYME_TS_PYPKA_PY") or None


def amber_env() -> str | None:
    """Fallback conda env for AmberTools IF tleap is not on PATH (split-env only)."""
    return os.environ.get("ENZYME_TS_AMBER_ENV") or None


def conda_sh() -> str:
    return os.environ.get(
        "ENZYME_TS_CONDA_SH",
        "/nfs/lambda_stor_01/homes/rzhu/miniforge3/etc/profile.d/conda.sh")


def workroot() -> str:
    return os.environ.get("ENZYME_TS_WORKROOT", os.getcwd())
