"""Harmonic (quasi-RRHO) Gibbs free-energy corrections from a partial Hessian.

For a FROZEN-boundary QM cluster the whole system does not translate or rotate
(the scaffold is pinned), so the correct statistical-mechanics model is purely
HARMONIC over the FREE-atom vibrational modes -- there are no translational or
rotational partition functions to add or project out. The Gibbs free energy of
the active-site sub-system is then

    G(T) = E_elec + ZPE + [H_vib(T) - H_vib(0)] - T * S_vib(T)

which is exactly ASE's HarmonicThermo.get_helmholtz_energy (the pV term is
negligible for a condensed-phase active site, so Helmholtz == Gibbs here).

Soft modes: harmonic vibrational entropy S ~ -k*ln(nu) diverges as nu -> 0, and
a frozen-cluster free region always has a few very soft modes (loosely anchored
waters / sidechain wags). Two treatments are provided (see `gibbs(method=...)`):
  - 'qrrho' (DEFAULT): Grimme quasi-RRHO -- ZPE and vibrational enthalpy stay
    harmonic, and only each mode's ENTROPY is interpolated between the harmonic-
    oscillator and free-rotor limits via a Chai-Head-Gordon damping weight
    w = 1/(1+(cutoff/nu)^4). Soft modes cross smoothly to the (finite) free-rotor
    entropy instead of being truncated. This is the community standard (Grimme,
    Chem. Eur. J. 2012; used by the Ohmura ML/MM toolkit's thermoanalysis module).
  - 'floor' (legacy): Cramer-Truhlar hard floor -- raise every real mode below
    `floor_cm` to `floor_cm` before the (fully harmonic) entropy sum.
Imaginary modes (the reaction coordinate at a TS, plus numerical near-zero
artifacts) are dropped in both.

Reuses the pipeline's own partial Hessian (ts_sella.confirm / uma_batch), so the
same PES engine (UMA or xTB via uma_helper.ENGINE) and the same free-atom
subspace are used consistently with the TS search.
"""
from __future__ import annotations
import numpy as np
from ase.units import invcm                      # 1 cm^-1 expressed in eV
from ase.thermochemistry import HarmonicThermo
from .engine import calculator as uma_helper
from .pes import saddle as ts_sella

DEFAULT_T = 298.15
FLOOR_CM = 100.0       # legacy hard-floor cutoff (cm^-1), Cramer-Truhlar/Shermo
                       # standard. NOTE: the qRRHO path (default) uses the SAME 100
                       # cm^-1 as the Chai-Head-Gordon damping cutoff but interpolates
                       # rather than truncates (see QRRHO_CUTOFF_CM below).


# ---- physical constants (SI, for the free-rotor limit) ----
_H = 6.62607015e-34         # J s
_KB = 1.380649e-23          # J/K
_C_CM = 2.99792458e10       # cm/s
_EV = 1.602176634e-19       # J/eV
_KB_EV = _KB / _EV          # eV/K
_BAV = 1.0e-44              # kg m^2, Grimme average moment of inertia
QRRHO_CUTOFF_CM = 100.0     # cm^-1, Chai-Head-Gordon damping cutoff (Grimme uses 100)
QRRHO_ALPHA = 4


def _real_freqs(freqs_cm):
    """Real vibrational wavenumbers (cm^-1), imaginary (TS reaction coord + numerical
    near-zero artifacts) dropped."""
    real = np.array([f.real for f in freqs_cm if abs(f.imag) < 1e-6 and f.real > 1e-6])
    n_imag = int(sum(1 for f in freqs_cm if f.imag > 1e-6))
    return real, n_imag


def real_vib_energies(freqs_cm, floor_cm=FLOOR_CM):
    """(legacy hard-floor) vib energies (eV) with modes < floor_cm raised to floor_cm
    (Cramer-Truhlar). Kept for method='floor'."""
    real, n_imag = _real_freqs(freqs_cm)
    n_raised = int(np.sum(real < floor_cm))
    raised = np.where(real < floor_cm, floor_cm, real)
    return raised * invcm, n_imag, n_raised


def _S_harmonic_mode(nu_cm, T):
    """Harmonic-oscillator vibrational entropy per mode (eV/K)."""
    x = (nu_cm * invcm) / (_KB_EV * T)           # h nu / kT
    x = np.maximum(x, 1e-12)
    return _KB_EV * (x / np.expm1(x) - np.log1p(-np.exp(-x)))


def _S_freerotor_mode(nu_cm, T):
    """Free-rotor entropy per mode (eV/K), Grimme 2012: a low mode treated as a free
    rotor with moment mu = h/(8 pi^2 nu), damped toward Bav so S stays finite as nu->0."""
    nu_hz = nu_cm * _C_CM
    mu = _H / (8.0 * np.pi**2 * nu_hz)           # kg m^2
    mu_eff = mu * _BAV / (mu + _BAV)
    arg = 8.0 * np.pi**3 * mu_eff * _KB * T / _H**2
    S_si = _KB * (0.5 + 0.5 * np.log(arg))       # J/K
    return S_si / _EV


def qrrho_entropy(freqs_cm, T, cutoff_cm=QRRHO_CUTOFF_CM, alpha=QRRHO_ALPHA):
    """Grimme quasi-RRHO vibrational entropy (eV/K): each mode's entropy is a
    Chai-Head-Gordon-weighted blend of harmonic-oscillator and free-rotor limits,
    w = 1/(1+(cutoff/nu)^alpha). Replaces the hard low-frequency floor -- soft modes
    smoothly cross over to the (finite) free-rotor entropy instead of being truncated."""
    real, _ = _real_freqs(freqs_cm)
    if len(real) == 0:
        return 0.0, 0
    w = 1.0 / (1.0 + (cutoff_cm / real) ** alpha)
    S = w * _S_harmonic_mode(real, T) + (1.0 - w) * _S_freerotor_mode(real, T)
    n_rotor = int(np.sum(w < 0.5))               # modes more free-rotor than HO
    return float(np.sum(S)), n_rotor


def gibbs(atoms, charge, spin, model, T=DEFAULT_T, floor_cm=FLOOR_CM,
          e_elec=None, freqs_cm=None, method="qrrho"):
    """Gibbs free energy G(T) [eV] of one stationary point (frozen-boundary cluster,
    harmonic over free atoms; no translation/rotation -- the scaffold is pinned).

    Soft modes: method='qrrho' (default) uses Grimme quasi-RRHO -- ZPE and vibrational
    enthalpy stay harmonic (they do not diverge at low nu), and only the ENTROPY of each
    mode is interpolated between the harmonic-oscillator and free-rotor limits (Chai-
    Head-Gordon damping). method='floor' is the legacy Cramer-Truhlar hard floor.

    Pass e_elec / freqs_cm to reuse the TS Hessian; else a partial Hessian is computed.
    """
    if e_elec is None:
        a = atoms.copy(); a.info.update(charge=int(charge), spin=int(spin))
        a.calc = uma_helper.get_calculator("omol", model)
        e_elec = a.get_potential_energy()
    if freqs_cm is None:
        freqs_cm, _n_imag, _imag, _evec, _cart = ts_sella.confirm(atoms, charge, spin, model)

    if method == "floor":
        vib_e, n_imag, n_raised = real_vib_energies(freqs_cm, floor_cm)
        th = HarmonicThermo(vib_energies=vib_e, potentialenergy=e_elec)
        G = th.get_helmholtz_energy(T, verbose=False)
        return dict(e_elec=float(e_elec), zpe=float(th.get_ZPE_correction()), G=float(G),
                    g_corr=float(G - e_elec), ts_correction=float(-T * th.get_entropy(T, verbose=False)),
                    S_vib=float(th.get_entropy(T, verbose=False)), T=float(T),
                    n_imag=int(n_imag), n_vib=int(len(vib_e)), n_raised=int(n_raised),
                    n_rotor=0, method="floor")

    # --- quasi-RRHO (default) ---
    real, n_imag = _real_freqs(freqs_cm)
    vib_e_real = real * invcm                                # eV, no floor
    x = vib_e_real / (_KB_EV * T)
    zpe = float(0.5 * np.sum(vib_e_real))                    # harmonic ZPE
    U_th = float(np.sum(vib_e_real / np.expm1(np.maximum(x, 1e-12))))  # thermal vib energy above ZPE
    S_vib, n_rotor = qrrho_entropy(freqs_cm, T)
    G = float(e_elec + zpe + U_th - T * S_vib)               # Helmholtz == Gibbs (pinned cluster)
    return dict(e_elec=float(e_elec), zpe=zpe, G=G, g_corr=float(G - e_elec),
                ts_correction=float(-T * S_vib), S_vib=float(S_vib), T=float(T),
                n_imag=int(n_imag), n_vib=int(len(real)), n_rotor=int(n_rotor),
                n_raised=0, method="qrrho")
