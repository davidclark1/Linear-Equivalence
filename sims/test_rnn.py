"""Tests for rnn.py: streaming vs full-timeseries agreement, normalization, determinism."""

import math

import numpy as np
import pytest
import torch

from rnn import (
    simulate_and_compute_covariance,
    compute_lagged_covariance,
    _erf_nonlinearity,
    _SQRT_PI_2,
)
from single_site_dmft import compute_Delta_0, compute_phi_prime_avg

# Use CPU for tests (no GPU required)
DEVICE = "cpu"


def _reference_simulation(N, g, dt, T, T_burn_in, T_save, n_ics, device, seed):
    """Reference Euler loop matching simulate_and_compute_covariance's RNG sequence.

    Returns the full timeseries of phi (activations) and the coupling matrix J.
    Used to validate the streaming computation against a post-hoc reference.
    """
    device = torch.device(device)
    n_euler_steps = round(T / dt)
    n_burn_in = round(T_burn_in / dt)
    save_iter = round(T_save / dt)
    n_save = (n_euler_steps - n_burn_in) // save_iter

    # Match the streaming function's RNG exactly
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    J = torch.randn(N, N, device=device, generator=gen) * (g / N**0.5)
    x = torch.randn(n_ics, N, device=device, generator=gen) * g

    sqrt_pi_2 = math.sqrt(math.pi) / 2
    phi_all = torch.zeros(n_ics, n_save, N, device=device)
    x_all = torch.zeros(n_ics, n_save, N, device=device)
    save_count = 0

    with torch.no_grad():
        for step in range(n_euler_steps):
            phi = torch.erf(sqrt_pi_2 * x)

            if step >= n_burn_in and (step - n_burn_in) % save_iter == 0:
                phi_all[:, save_count, :] = phi
                x_all[:, save_count, :] = x
                save_count += 1

            x.mul_(1 - dt).addmm_(phi, J.t(), alpha=dt)
            del phi

    return phi_all.cpu(), x_all.cpu(), J.cpu()


def _compute_lagged_cov_reference(phi_all, n_lags):
    """Compute lagged cross-covariance from full timeseries (reference impl)."""
    n_ics, n_save, N = phi_all.shape
    C_phi = torch.zeros(n_lags + 1, N, N)
    for ic in range(n_ics):
        a = phi_all[ic]  # (n_save, N)
        for lag in range(n_lags + 1):
            n_pairs = n_save - lag
            C_phi[lag] += (a[:n_pairs].T @ a[lag:lag + n_pairs]) / n_pairs
    C_phi /= n_ics
    C_phi[0] = (C_phi[0] + C_phi[0].T) / 2
    return C_phi


def _compute_residual_cov_reference(phi_all, x_all, alpha_dmft):
    """Compute equal-time residual covariance from full timeseries."""
    n_ics, n_save, N = phi_all.shape
    C_res_0 = torch.zeros(N, N)
    for ic in range(n_ics):
        res = phi_all[ic] - alpha_dmft * x_all[ic]
        C_res_0 += (res.T @ res) / n_save
    C_res_0 /= n_ics
    C_res_0 = (C_res_0 + C_res_0.T) / 2
    return C_res_0


def _compute_spectrum_reference(phi_all, n_spectrum_ics):
    """Compute scalar power spectrum from full timeseries."""
    n_save, N = phi_all.shape[1], phi_all.shape[2]
    phi_sub = phi_all[:n_spectrum_ics]  # (n_spectrum_ics, n_save, N)
    n_rfft = n_save // 2 + 1
    rfft_spec = torch.fft.rfft(phi_sub, dim=1)
    C_half = (rfft_spec.abs() ** 2).mean(dim=(0, 2)) / n_save
    C_full = torch.zeros(n_save)
    C_full[:n_rfft] = C_half
    if n_save % 2 == 0:
        C_full[n_rfft:] = C_half[1:-1].flip(0)
    else:
        C_full[n_rfft:] = C_half[1:].flip(0)
    return C_full


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

SMALL_PARAMS = dict(N=50, dt=0.05, T=100.0, T_burn_in=10.0, T_save=0.5,
                    n_ics=5, n_lags=5, n_spectrum_ics=3, device=DEVICE, seed=42)


@pytest.fixture(params=[1.5, 2.5])
def g_value(request):
    return request.param


# ---------------------------------------------------------------------------
# 1. Streaming vs full-timeseries C_phi agreement
# ---------------------------------------------------------------------------

def test_streaming_vs_full_C_phi(g_value):
    """Streaming lagged covariance matches post-hoc computation from full timeseries."""
    g = g_value
    p = SMALL_PARAMS.copy()
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    # Streaming path
    result = simulate_and_compute_covariance(g=g, alpha_dmft=alpha_dmft, **p)
    C_phi_stream = result['C_phi']

    # Reference path (same RNG sequence)
    phi_all, _, _ = _reference_simulation(
        p['N'], g, p['dt'], p['T'], p['T_burn_in'], p['T_save'],
        p['n_ics'], p['device'], p['seed'])
    C_phi_ref = _compute_lagged_cov_reference(phi_all, p['n_lags'])

    for lag in range(p['n_lags'] + 1):
        err = (C_phi_stream[lag] - C_phi_ref[lag]).abs()
        scale = C_phi_ref[lag].abs().max().item()
        rel_err = err.max().item() / max(scale, 1e-12)
        assert rel_err < 1e-4, (
            f"g={g}, lag={lag}: max relative error {rel_err:.2e} exceeds tolerance"
        )


# ---------------------------------------------------------------------------
# 2. Streaming vs full-timeseries C_res_0 agreement
# ---------------------------------------------------------------------------

def test_streaming_vs_full_C_res_0(g_value):
    """Streaming residual covariance matches post-hoc computation."""
    g = g_value
    p = SMALL_PARAMS.copy()
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    result = simulate_and_compute_covariance(g=g, alpha_dmft=alpha_dmft, **p)
    C_res_stream = result['C_res_0']

    phi_all, x_all, _ = _reference_simulation(
        p['N'], g, p['dt'], p['T'], p['T_burn_in'], p['T_save'],
        p['n_ics'], p['device'], p['seed'])
    C_res_ref = _compute_residual_cov_reference(phi_all, x_all, alpha_dmft)

    err = (C_res_stream - C_res_ref).abs()
    scale = C_res_ref.abs().max().item()
    rel_err = err.max().item() / max(scale, 1e-12)
    assert rel_err < 1e-4, (
        f"g={g}: max relative error {rel_err:.2e} exceeds tolerance"
    )


# ---------------------------------------------------------------------------
# 3. Streaming vs full-timeseries power spectrum agreement
# ---------------------------------------------------------------------------

def test_streaming_vs_full_spectrum(g_value):
    """Streaming power spectrum matches post-hoc computation."""
    g = g_value
    p = SMALL_PARAMS.copy()
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    result = simulate_and_compute_covariance(g=g, alpha_dmft=alpha_dmft, **p)
    spec_stream = result['C_phi_scalar_ft_sim']

    phi_all, _, _ = _reference_simulation(
        p['N'], g, p['dt'], p['T'], p['T_burn_in'], p['T_save'],
        p['n_ics'], p['device'], p['seed'])
    spec_ref = _compute_spectrum_reference(phi_all, p['n_spectrum_ics'])

    err = (spec_stream - spec_ref).abs()
    scale = spec_ref.abs().max().item()
    rel_err = err.max().item() / max(scale, 1e-12)
    assert rel_err < 1e-4, (
        f"g={g}: max relative error {rel_err:.2e} exceeds tolerance"
    )


# ---------------------------------------------------------------------------
# 4. Circular buffer boundary correctness
# ---------------------------------------------------------------------------

def test_circular_buffer_extreme():
    """With n_lags close to n_save, the buffer still works correctly."""
    g = 2.0
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    result = simulate_and_compute_covariance(
        N=10, g=g, dt=0.1, T=10.0, T_burn_in=1.0, T_save=1.0,
        n_ics=2, n_lags=7,  # n_save = 9, n_lags = 7
        alpha_dmft=alpha_dmft, n_spectrum_ics=1, device=DEVICE, seed=0,
    )
    C_phi = result['C_phi']

    assert (C_phi[0].diagonal() > 0).all(), "C_phi(0) diagonal should be positive"
    assert torch.isfinite(C_phi).all(), "C_phi contains non-finite values"

    # Cross-check against reference
    phi_all, _, _ = _reference_simulation(
        10, g, 0.1, 10.0, 1.0, 1.0, 2, DEVICE, 0)
    C_phi_ref = _compute_lagged_cov_reference(phi_all, 7)

    for lag in range(8):
        err = (C_phi[lag] - C_phi_ref[lag]).abs().max().item()
        scale = C_phi_ref[lag].abs().max().item()
        rel_err = err / max(scale, 1e-12)
        assert rel_err < 1e-4, f"lag={lag}: rel error {rel_err:.2e}"


# ---------------------------------------------------------------------------
# 5. Normalization check: Tr(C_phi(0))/N ≈ DMFT prediction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g", [1.5, 2.5])
def test_trace_matches_dmft(g):
    """Tr(C_phi(0))/N should approximately match DMFT single-site autocorrelation."""
    from single_site_dmft import integrate_potential, compute_C_simple, fix

    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)
    tau_vals, Delta = integrate_potential(Delta_0, g, tau_max=40, N_tau=10000)
    Delta = fix(Delta)
    C_phi_dmft_0 = compute_C_simple(Delta_0, Delta[0])

    # Use larger N and more ICs for better convergence
    result = simulate_and_compute_covariance(
        N=500, g=g, dt=0.05, T=1000.0, T_burn_in=100.0, T_save=0.5,
        n_ics=20, n_lags=1, alpha_dmft=alpha_dmft,
        n_spectrum_ics=0, device=DEVICE, seed=123,
    )
    tr_per_neuron = result['C_phi'][0].diagonal().mean().item()

    rel_err = abs(tr_per_neuron - C_phi_dmft_0) / C_phi_dmft_0
    assert rel_err < 0.10, (
        f"g={g}: Tr(C_phi(0))/N = {tr_per_neuron:.6f}, DMFT = {C_phi_dmft_0:.6f}, "
        f"rel err = {rel_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Determinism: same seed → same output
# ---------------------------------------------------------------------------

def test_determinism():
    """Same seed produces bit-exact results."""
    g = 2.0
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    params = dict(N=20, g=g, dt=0.05, T=50.0, T_burn_in=5.0, T_save=0.5,
                  n_ics=3, n_lags=3, alpha_dmft=alpha_dmft,
                  n_spectrum_ics=2, device=DEVICE, seed=99)

    r1 = simulate_and_compute_covariance(**params)
    r2 = simulate_and_compute_covariance(**params)

    assert torch.equal(r1['C_phi'], r2['C_phi']), "C_phi not deterministic"
    assert torch.equal(r1['C_res_0'], r2['C_res_0']), "C_res_0 not deterministic"
    assert torch.equal(r1['J'], r2['J']), "J not deterministic"
    assert torch.equal(r1['C_phi_scalar_ft_sim'], r2['C_phi_scalar_ft_sim']), \
        "C_phi_scalar_ft_sim not deterministic"


# ---------------------------------------------------------------------------
# 7. J matrix matches between streaming and reference
# ---------------------------------------------------------------------------

def test_J_matches_reference():
    """Streaming and reference generate the same J matrix."""
    g = 2.5
    p = SMALL_PARAMS.copy()
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    result = simulate_and_compute_covariance(g=g, alpha_dmft=alpha_dmft, **p)
    _, _, J_ref = _reference_simulation(
        p['N'], g, p['dt'], p['T'], p['T_burn_in'], p['T_save'],
        p['n_ics'], p['device'], p['seed'])

    assert torch.equal(result['J'], J_ref), "J matrices don't match"
