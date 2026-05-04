"""DMFT solution for random RNN with erf nonlinearity.

Solves the Sompolinsky-Crisanti-Sommers (1988) self-consistent equations
for the single-site autocorrelation via the "particle in a potential" method.

Nonlinearity: phi(x) = erf(sqrt(pi)/2 * x)

Workflow:
    Delta_0 = compute_Delta_0(g)
    taus, Delta = integrate_potential(Delta_0, g)
    alpha = compute_phi_prime_avg(Delta_0)
    C_phi = compute_C_simple(Delta_0, Delta)
"""

import numpy as np
from scipy.optimize import root
from scipy.integrate import odeint


# --- Gaussian averages (analytical) ---

def compute_C(Delta_11, Delta_22, Delta_12):
    """<phi(x1) phi(x2)> for jointly Gaussian (x1, x2) with given covariance."""
    A = (Delta_11 + 2 / np.pi) * (Delta_22 + 2 / np.pi)
    in_sqrt = np.clip(A - Delta_12**2, a_min=1e-20, a_max=np.inf)
    return (2 / np.pi) * np.arctan(Delta_12 / np.sqrt(in_sqrt))


def compute_C_simple(Delta_0, Delta):
    """<phi(x1) phi(x2)> when Var(x1) = Var(x2) = Delta_0, Cov(x1,x2) = Delta."""
    return compute_C(Delta_0, Delta_0, Delta)


def compute_phi_prime_avg(Delta_0):
    """<phi'(x)> for x ~ N(0, Delta_0)."""
    return 1 / np.sqrt(1 + np.pi * Delta_0 / 2)


# --- Potential for the "particle" ODE ---

def compute_C_antideriv(Delta_0, Delta):
    """Antiderivative of compute_C_simple w.r.t. Delta."""
    A = (Delta_0 + 2 / np.pi)**2
    t1 = np.sqrt(A - Delta**2)
    t2 = Delta * np.arctan(Delta / t1)
    return (2 / np.pi) * (t1 + t2)


def compute_V(Delta_0, Delta, g):
    """Potential V(Delta) for the autocorrelation ODE."""
    t1 = -0.5 * (Delta**2 - Delta_0**2)
    t2 = g**2 * (compute_C_antideriv(Delta_0, Delta) - compute_C_antideriv(Delta_0, Delta_0))
    return t1 + t2


def compute_V_deriv(Delta_0, Delta, g):
    """dV/dDelta = -Delta + g^2 C^phi(Delta)."""
    return -Delta + g**2 * compute_C_simple(Delta_0, Delta)


# --- Solving the DMFT ---

def compute_Delta_0(g, x0=None):
    """Find equal-time variance Delta_0 by solving V(Delta_0, Delta=0, g) = 0."""
    def f_opt(Delta_0):
        Delta_0 = np.abs(Delta_0)
        return compute_V(Delta_0, 0.0, g)
    res = root(fun=f_opt, x0=g**2 if x0 is None else x0)
    if res.success:
        return np.abs(res.x.item())
    else:
        return np.nan


def integrate_potential(Delta_0, g, tau_max=40, N_tau=1000):
    """Integrate the 'particle in a potential' ODE for Delta(tau).

    Returns:
        taus: (N_tau,) array of lag values
        Delta: (N_tau,) array of autocorrelation values
    """
    def dy_dt(y, t):
        Delta, Delta_dot = y
        return [Delta_dot, -compute_V_deriv(Delta_0, Delta, g)]

    taus = np.linspace(0, tau_max, N_tau)
    out = odeint(dy_dt, y0=[Delta_0, 0.0], t=taus, rtol=1e-12)
    Delta = out[:, 0]
    return taus, Delta


def fix(Delta):
    """Clip integrated Delta(tau) to be monotonically decreasing and non-negative."""
    Delta = Delta.copy()
    deriv = np.diff(Delta)
    if np.any(deriv > 0):
        i = np.argmax(deriv > 0)
        Delta[i:] = Delta[i]
    if np.any(Delta < 0):
        i = np.argmax(Delta < 0)
        Delta[i:] = Delta[i - 1]
    return Delta


def symmetrize(Delta):
    """Extend Delta(tau) for tau >= 0 to the full range [-tau_max, tau_max]."""
    return np.concatenate((Delta, (Delta[-1],), Delta[1:][::-1]))
