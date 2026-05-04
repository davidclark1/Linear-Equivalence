"""Tests for DMFT analytical Gaussian averages against Monte Carlo."""

import numpy as np
import pytest
from single_site_dmft import compute_C, compute_C_simple, compute_phi_prime_avg, compute_C_antideriv

phi = lambda x: np.vectorize(lambda z: __import__('scipy.special', fromlist=['erf']).erf(np.sqrt(np.pi) / 2 * z))(x)
phi_prime = lambda x: np.exp(-np.pi / 4 * x**2)

# Precompute phi for efficiency
from scipy.special import erf as _erf
phi = lambda x: _erf(np.sqrt(np.pi) / 2 * x)
phi_prime = lambda x: np.exp(-np.pi / 4 * x**2)

N_SAMPLES = 2_000_000
RNG = np.random.default_rng(0)


def sample_joint_gaussian(Delta_11, Delta_22, Delta_12, n):
    """Sample (x1, x2) from N(0, [[Delta_11, Delta_12], [Delta_12, Delta_22]])."""
    cov = np.array([[Delta_11, Delta_12], [Delta_12, Delta_22]])
    return RNG.multivariate_normal([0, 0], cov, size=n)


# --- compute_C: general case ---

@pytest.mark.parametrize("Delta_11, Delta_22, Delta_12", [
    (1.0, 1.0, 0.5),
    (2.0, 3.0, 1.0),
    (0.5, 0.5, 0.0),
    (1.0, 1.0, 0.9),
    (5.0, 2.0, -1.5),
])
def test_compute_C(Delta_11, Delta_22, Delta_12):
    samples = sample_joint_gaussian(Delta_11, Delta_22, Delta_12, N_SAMPLES)
    mc = np.mean(phi(samples[:, 0]) * phi(samples[:, 1]))
    analytical = compute_C(Delta_11, Delta_22, Delta_12)
    assert abs(mc - analytical) < 5e-3, (
        f"compute_C({Delta_11}, {Delta_22}, {Delta_12}): MC={mc:.6f}, analytical={analytical:.6f}"
    )


# --- compute_C_simple: equal-variance case ---

@pytest.mark.parametrize("Delta_0, Delta", [
    (1.0, 0.5),
    (1.0, 0.0),
    (1.0, 1.0),
    (3.0, 2.0),
    (0.3, 0.1),
])
def test_compute_C_simple(Delta_0, Delta):
    samples = sample_joint_gaussian(Delta_0, Delta_0, Delta, N_SAMPLES)
    mc = np.mean(phi(samples[:, 0]) * phi(samples[:, 1]))
    analytical = compute_C_simple(Delta_0, Delta)
    assert abs(mc - analytical) < 5e-3, (
        f"compute_C_simple({Delta_0}, {Delta}): MC={mc:.6f}, analytical={analytical:.6f}"
    )


# --- compute_phi_prime_avg ---

@pytest.mark.parametrize("Delta_0", [0.5, 1.0, 2.0, 5.0])
def test_compute_phi_prime_avg(Delta_0):
    x = RNG.normal(0, np.sqrt(Delta_0), size=N_SAMPLES)
    mc = np.mean(phi_prime(x))
    analytical = compute_phi_prime_avg(Delta_0)
    assert abs(mc - analytical) < 5e-3, (
        f"compute_phi_prime_avg({Delta_0}): MC={mc:.6f}, analytical={analytical:.6f}"
    )


# --- compute_C_antideriv: check derivative matches compute_C_simple ---

@pytest.mark.parametrize("Delta_0, Delta", [
    (1.0, 0.5),
    (3.0, 1.0),
    (2.0, 0.0),
])
def test_compute_C_antideriv(Delta_0, Delta):
    eps = 1e-6
    numerical_deriv = (
        compute_C_antideriv(Delta_0, Delta + eps) - compute_C_antideriv(Delta_0, Delta - eps)
    ) / (2 * eps)
    analytical = compute_C_simple(Delta_0, Delta)
    assert abs(numerical_deriv - analytical) < 1e-4, (
        f"d/dDelta C_antideriv({Delta_0}, {Delta}): numerical={numerical_deriv:.6f}, "
        f"compute_C_simple={analytical:.6f}"
    )
