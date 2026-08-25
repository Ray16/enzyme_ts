"""Batched inference + a batched CI-NEB and batched Hessian.

The win: uma-s is tiny, so single-structure calls leave the GPU ~10-20% idle
(launch-overhead bound). Evaluating many structures (NEB band, finite-difference
Hessian displacements) in ONE predict() call amortizes that overhead.

batched_ef dispatches to the active engine (UMA: one native batched predict;
MACE-POLAR-1: serial fallback for now -- see engine/mace_polar.py). Everything
below (BatchedNEB, batched_hessian, batched_partial_hessian) sits on top of
batched_ef and is therefore engine-agnostic.
"""
from __future__ import annotations
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep import NEB
from . import calculator as uma_helper
from . import registry

DEFAULT_MODEL = uma_helper.DEFAULT_MODEL


def batched_ef(atoms_list, charge, spin, model=DEFAULT_MODEL):
    """Energies (eV, len N) and forces (list of (natoms,3)) for a list of Atoms,
    computed by the model's engine backend (one batched predict for UMA)."""
    return registry.adapter_for_model(model).batched_ef(atoms_list, charge, spin, model)


class BatchedNEB(NEB):
    """NEB that evaluates the whole band in one batched predict per step."""
    def __init__(self, images, charge, spin, model, **kw):
        super().__init__(images, allow_shared_calculator=True, **kw)
        self._cs = (charge, spin, model)

    def get_forces(self):
        charge, spin, model = self._cs
        imgs = self.images  # include endpoints (improvedtangent needs their E)
        E, F = batched_ef(imgs, charge, spin, model)
        for img, e, f in zip(imgs, E, F):
            spc = SinglePointCalculator(img, energy=float(e), forces=f)
            img.calc = spc
        return super().get_forces()


def batched_hessian(atoms, charge, spin, model="uma-s-1p2", delta=0.01):
    """Central-difference Hessian via batched force evals.
    Returns (freqs_cm complex, n_imag, imag_list_cm, eigvecs)."""
    import ase.units as u
    n = len(atoms)
    x0 = atoms.get_positions().ravel()
    ndof = 3 * n
    # build all 2*ndof displaced structures, evaluate in batches
    disp = []
    for i in range(ndof):
        for s in (+1, -1):
            a = atoms.copy()
            p = x0.copy(); p[i] += s * delta
            a.set_positions(p.reshape(-1, 3))
            disp.append(a)
    # chunk to keep memory bounded -- ADAPTIVE: the force autograd graph scales with
    # (batch x natoms); a fixed 64 OOMs on ~680-atom clusters (V100 32GB). Cap atoms/batch.
    import torch
    forces = []
    CH = max(1, min(64, 12000 // max(1, n)))
    for k in range(0, len(disp), CH):
        _, Fs = batched_ef(disp[k:k + CH], charge, spin, model)
        forces.extend(Fs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    H = np.zeros((ndof, ndof))
    for i in range(ndof):
        Fp = forces[2 * i].ravel()
        Fm = forces[2 * i + 1].ravel()
        H[i] = -(Fp - Fm) / (2 * delta)   # dF/dx = -d2E/dx2
    H = 0.5 * (H + H.T)
    masses = np.repeat(atoms.get_masses(), 3)
    Hm = H / np.sqrt(np.outer(masses, masses))
    evals, evecs = np.linalg.eigh(Hm)
    # eV/Ang^2/amu -> cm^-1
    conv = u._e * 1e20 / (u._amu)        # to (1/s^2)*... use ase convention
    # ASE: freq[cm-1] = sqrt(eval) * sqrt(_e/_amu)/Ang *1e10 /(2 pi c)
    s = np.sqrt(np.abs(evals).astype(complex))
    factor = np.sqrt(u._e / u._amu) * 1e10 / (2 * np.pi * u._c * 100)
    freqs = factor * np.sqrt(evals.astype(complex))
    n_imag = int(np.sum(np.imag(freqs) > 1e-6))
    imag = [complex(f) for f in freqs if np.imag(f) > 1e-6]
    return freqs, n_imag, imag, evecs


def batched_partial_hessian(atoms, free_idx, charge, spin, model="uma-s-1p2", delta=0.01):
    """Partial (mass-weighted) Hessian over only the free atoms — the relevant
    subspace for a frozen-scaffold cluster TS. Returns (freqs_cm, evecs, imag_cm)."""
    import ase.units as u
    free_idx = list(free_idx)
    x0 = atoms.get_positions()
    dofs = [(a, k) for a in free_idx for k in range(3)]
    disp = []
    for (a, k) in dofs:
        for s in (+1, -1):
            at = atoms.copy(); p = x0.copy(); p[a, k] += s * delta
            at.set_positions(p); disp.append(at)
    import torch
    forces = []
    CH = max(1, min(64, 12000 // max(1, len(atoms))))
    for c in range(0, len(disp), CH):
        _, Fs = batched_ef(disp[c:c + CH], charge, spin, model)
        forces.extend(Fs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    nd = len(dofs); H = np.zeros((nd, nd))
    for i, (a, k) in enumerate(dofs):
        Fp = forces[2 * i][free_idx].ravel()
        Fm = forces[2 * i + 1][free_idx].ravel()
        H[i] = -(Fp - Fm) / (2 * delta)
    H = 0.5 * (H + H.T)
    masses = np.repeat(atoms.get_masses()[free_idx], 3)
    Hm = H / np.sqrt(np.outer(masses, masses))
    evals, evecs = np.linalg.eigh(Hm)
    factor = np.sqrt(u._e / u._amu) * 1e10 / (2 * np.pi * u._c * 100)
    freqs = factor * np.sqrt(evals.astype(complex))
    imag = [complex(f) for f in freqs if np.imag(f) > 1e-6]
    return freqs, evecs, imag
