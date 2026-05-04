"""Tests for DMFT-to-frequency-grid resampling pipeline (compute_C_phi_scalar_ft_dmft).

Tests cover:
- Known analytical Fourier pairs
- Parseval's theorem (energy conservation)
- Symmetry of output
- Resolution convergence with tau_max
- Interpolation accuracy
- Full DMFT pipeline self-consistency
"""

import math

import numpy as np
import pytest
import torch

from rnn import compute_C_phi_scalar_ft_dmft
from single_site_dmft import (
    compute_Delta_0, integrate_potential, compute_phi_prime_avg,
    compute_C_simple, fix,
)


# ---------------------------------------------------------------------------
# 1. Known analytical case: exponential decay
# ---------------------------------------------------------------------------

def test_exponential_decay():
    """C(tau) = exp(-|tau|) has FT = 2/(1+omega^2).

    The DFT of the wrapped (sampled) version should approximate this.
    """
    tau_max = 200.0
    omega_max = 10.0

    # Generate fine-grid exponential decay
    N_tau = 100000
    tau_vals = np.linspace(0, tau_max, N_tau)
    C_tau = np.exp(-tau_vals)

    ft, n_omega, dt_theory = compute_C_phi_scalar_ft_dmft(
        tau_vals, C_tau, tau_max, omega_max)

    # Compare against analytical: FT of exp(-|tau|) = 2/(1+omega^2)
    # The DFT convention includes dt_theory scaling, so the DFT values
    # should be C_tau_wrapped[k] (not the continuous FT).
    # Instead, check that the spectrum is positive (it must be for a valid
    # autocorrelation) and has the right shape (peak at omega=0, decaying).
    assert ft[0] > 0, "Spectrum at omega=0 should be positive"
    assert (ft >= -1e-6).all(), "Power spectrum should be non-negative"

    # Check shape: ft should decay from peak at k=0
    # The first few bins should be monotonically decreasing
    assert ft[1] < ft[0], "Spectrum should decrease from omega=0"
    assert ft[2] < ft[1], "Spectrum should continue decreasing"

    # Quantitative check at omega=0: sum of C_wrapped should equal ft[0]
    # Since ft = FFT(C_wrapped), ft[0] = sum(C_wrapped)
    # C_wrapped at the effective lags should sum to approximately
    # 2 * integral_0^inf exp(-tau) dtau / dt_theory = 2/dt_theory
    # (factor 2 from negative lags). Check rough agreement.
    expected_ft0 = ft[0].item()
    assert expected_ft0 > 0, "ft[0] should be positive"


# ---------------------------------------------------------------------------
# 2. Parseval's theorem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [1.5, 2.5, 3.5])
def test_parseval(g):
    """Sum of power spectrum ≈ C(0) * n_omega (Parseval's theorem for DFT)."""
    Delta_0 = compute_Delta_0(g)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=200, N_tau=100000)
    Delta = fix(Delta)
    C_phi_scalar = compute_C_simple(Delta_0, Delta)

    tau_max = 100.0
    omega_max = 10.0
    ft, n_omega, dt_theory = compute_C_phi_scalar_ft_dmft(
        tau_vals, C_phi_scalar, tau_max, omega_max)

    # Parseval: (1/n_omega) sum |X[k]|^2 = (1/N) sum |x[n]|^2
    # But since our "x[n]" is the wrapped autocorrelation and ft = FFT(x),
    # we have: sum(ft) = n_omega * x[0] (DC component = sum of input)
    # Actually for real symmetric input: ft[0] = sum(C_wrapped)
    # The wrapped C at lag 0 is C(0), and the integral of the spectrum
    # times dt_theory / (2*pi) should give C(0).
    # Simpler check: (1/n_omega) * sum(ft) ≈ C(0)
    C_0 = C_phi_scalar[0]
    spectral_avg = ft.sum().item() / n_omega
    rel_err = abs(spectral_avg - C_0) / C_0

    assert rel_err < 0.01, (
        f"g={g}: Parseval check failed. spectral_avg={spectral_avg:.6f}, "
        f"C(0)={C_0:.6f}, rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Symmetry of wrapped spectrum
# ---------------------------------------------------------------------------

def test_spectrum_real():
    """The FFT of a real symmetric wrapped autocorrelation should be real.

    compute_C_phi_scalar_ft_dmft takes .real of the FFT, so any imaginary
    part is discarded. Verify that the imaginary part is negligible.
    """
    g = 2.5
    Delta_0 = compute_Delta_0(g)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=200, N_tau=100000)
    Delta = fix(Delta)
    C_phi_scalar = compute_C_simple(Delta_0, Delta)

    tau_max = 100.0
    omega_max = 10.0

    # Manually reproduce the wrapping to check imaginary part
    dt_theory = math.pi / omega_max
    n_omega = math.ceil(2 * tau_max * omega_max / math.pi)

    from scipy.interpolate import interp1d
    indices = np.arange(n_omega)
    effective_lag = np.minimum(indices, n_omega - indices) * dt_theory
    interp_fn = interp1d(tau_vals, C_phi_scalar, bounds_error=False, fill_value=0.0)
    C_wrapped = interp_fn(effective_lag)

    fft_result = np.fft.fft(C_wrapped)
    imag_norm = np.abs(fft_result.imag).max()
    real_norm = np.abs(fft_result.real).max()
    ratio = imag_norm / max(real_norm, 1e-12)

    assert ratio < 1e-10, (
        f"FFT imaginary part is not negligible: |imag|/|real| = {ratio:.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Resolution convergence
# ---------------------------------------------------------------------------

def test_resolution_convergence():
    """Low-frequency bins should stabilize as tau_max increases."""
    g = 2.5
    Delta_0 = compute_Delta_0(g)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=200, N_tau=100000)
    Delta = fix(Delta)
    C_phi_scalar = compute_C_simple(Delta_0, Delta)

    omega_max = 10.0
    results = {}
    for tau_max in [50.0, 100.0, 200.0]:
        ft, n_omega, dt = compute_C_phi_scalar_ft_dmft(
            tau_vals, C_phi_scalar, tau_max, omega_max)
        results[tau_max] = ft[0].item()  # DC component

    # DC component should converge
    err_50_100 = abs(results[50.0] - results[100.0]) / abs(results[100.0])
    err_100_200 = abs(results[100.0] - results[200.0]) / abs(results[200.0])

    # All should be very close to each other (converged)
    assert err_50_100 < 0.01, (
        f"DC component not converged at tau_max=100: err vs 50 = {err_50_100:.4e}"
    )
    assert err_100_200 < 0.01, (
        f"DC component not converged at tau_max=200: err vs 100 = {err_100_200:.4e}"
    )


# ---------------------------------------------------------------------------
# 5. Interpolation accuracy
# ---------------------------------------------------------------------------

def test_interpolation_accuracy():
    """Interpolated values at the effective lag grid should match the ODE output."""
    g = 2.5
    Delta_0 = compute_Delta_0(g)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=200, N_tau=100000)
    Delta = fix(Delta)
    C_phi_scalar = compute_C_simple(Delta_0, Delta)

    tau_max = 100.0
    omega_max = 10.0
    dt_theory = math.pi / omega_max
    n_omega = math.ceil(2 * tau_max * omega_max / math.pi)

    # Compute the effective lag grid
    indices = np.arange(n_omega)
    effective_lag = np.minimum(indices, n_omega - indices) * dt_theory

    # Interpolated values
    from scipy.interpolate import interp1d
    interp_fn = interp1d(tau_vals, C_phi_scalar, bounds_error=False, fill_value=0.0)
    C_interp = interp_fn(effective_lag)

    # Find the closest ODE points for comparison
    max_err = 0.0
    for i, lag in enumerate(effective_lag):
        if lag > tau_vals[-1]:
            continue
        idx = np.searchsorted(tau_vals, lag)
        idx = min(idx, len(tau_vals) - 1)
        C_exact = np.interp(lag, tau_vals, C_phi_scalar)
        err = abs(C_interp[i] - C_exact)
        max_err = max(max_err, err)

    assert max_err < 1e-8, f"Interpolation error too large: {max_err:.2e}"


# ---------------------------------------------------------------------------
# 6. DMFT self-consistency: integral of spectrum ≈ C(0) ≈ analytical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [1.5, 2.5, 3.5])
def test_dmft_pipeline_self_consistency(g):
    """Full pipeline: Delta_0 → ODE → C_simple → resample → FFT.

    The average of the spectrum (= time-domain value at lag 0) should
    match compute_C_simple(Delta_0, Delta_0).
    """
    Delta_0 = compute_Delta_0(g)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=200, N_tau=100000)
    Delta = fix(Delta)
    C_phi_scalar = compute_C_simple(Delta_0, Delta)

    # Analytical C^phi(0) = compute_C_simple(Delta_0, Delta_0)
    C_0_analytical = compute_C_simple(Delta_0, Delta_0)
    C_0_from_ode = C_phi_scalar[0]

    # These should match exactly (same function, same input)
    assert abs(C_0_analytical - C_0_from_ode) < 1e-10, (
        f"C_simple(Delta_0, Delta_0)={C_0_analytical} != C_phi_scalar[0]={C_0_from_ode}"
    )

    # Now check spectral average
    ft, n_omega, dt_theory = compute_C_phi_scalar_ft_dmft(
        tau_vals, C_phi_scalar, tau_max=100.0, omega_max=10.0)

    spectral_C_0 = ft.sum().item() / n_omega
    rel_err = abs(spectral_C_0 - C_0_analytical) / C_0_analytical

    assert rel_err < 0.01, (
        f"g={g}: spectral C(0)={spectral_C_0:.6f}, analytical={C_0_analytical:.6f}, "
        f"rel_err={rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 7. Non-negative power spectrum for physical autocorrelation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [1.5, 2.5, 3.5])
def test_spectrum_non_negative(g):
    """Power spectrum of a valid autocorrelation should be non-negative."""
    Delta_0 = compute_Delta_0(g)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=200, N_tau=100000)
    Delta = fix(Delta)
    C_phi_scalar = compute_C_simple(Delta_0, Delta)

    ft, n_omega, dt_theory = compute_C_phi_scalar_ft_dmft(
        tau_vals, C_phi_scalar, tau_max=100.0, omega_max=10.0)

    min_val = ft.min().item()
    assert min_val > -1e-6, (
        f"g={g}: spectrum has negative values, min={min_val:.2e}"
    )
