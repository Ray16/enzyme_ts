"""UMA (FAIRChem / OMol25) engine adapter.

The heavy object is the MLIPPredictUnit (model weights on GPU); it is loaded ONCE
per (model, device, settings) and cached -- get_calculator is called on every
batched predict, so reloading would be catastrophic. The returned
FAIRChemCalculator is cheap and wraps the shared predictor.

Inference runs in full float32 with TF32 off ("default" settings): V100 (sm_70)
has no TF32 and no fast bf16, so "turbo" (tf32/compile/merge_mole) gives nothing
here and can hurt (torch.compile re-traces on the varying Hessian batch sizes).
The real speedup on this hardware is batching (batched_ef), not turbo mode.
"""
from __future__ import annotations
import functools
import numpy as np
from . import registry


@functools.lru_cache(maxsize=8)
def _get_predictor(model: str, device: str, inference_settings: str):
    """Load and cache the MLIPPredictUnit. 'default' => float32, tf32 off."""
    from fairchem.core import pretrained_mlip
    return pretrained_mlip.get_predict_unit(
        model, inference_settings=inference_settings, device=device)


class UMAEngine:
    name = "uma"

    def available(self) -> bool:
        try:
            import fairchem.core  # noqa: F401
            return True
        except Exception:
            return False

    def get_calculator(self, task_name="omol", model="uma-s-1p2",
                       device=None, inference_settings="default"):
        from fairchem.core import FAIRChemCalculator
        dev = registry.resolve_device(device)
        predictor = _get_predictor(model, dev, inference_settings)
        return FAIRChemCalculator(predictor, task_name=task_name)

    def batched_ef(self, atoms_list, charge, spin, model="uma-s-1p2",
                   task_name="omol", device=None):
        """Energies (eV) and forces for a list of Atoms in ONE batched predict."""
        from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
        calc = self.get_calculator(task_name, model, device)
        datas = []
        for at in atoms_list:
            a = at.copy()
            a.info["charge"] = int(charge)
            a.info["spin"] = int(spin)
            datas.append(calc.a2g(a))
        batch = atomicdata_list_to_batch(datas)
        pred = calc.predictor.predict(batch)
        E = pred["energy"].detach().cpu().numpy().reshape(-1)
        F = pred["forces"].detach().cpu().numpy()
        natoms = [len(a) for a in atoms_list]
        Fs = np.split(F, np.cumsum(natoms)[:-1])
        return E, [f.copy() for f in Fs]
