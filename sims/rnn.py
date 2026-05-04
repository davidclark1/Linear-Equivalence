"""RNN simulation, streaming covariance computation, and theory prediction.

Contains:
- simulate_rnn(): Full-timeseries simulation with permutation trick (reference impl)
- simulate_and_compute_covariance(): Streaming simulation with inline covariance
- compute_lagged_covariance(): Post-hoc lagged covariance from full timeseries
- compute_equal_time_covariance(): Post-hoc equal-time covariance from full timeseries
- predict_covariance_theory(): Linear-system theory prediction via matrix inverse
- compute_C_phi_scalar_ft_dmft(): DMFT autocorrelation → frequency-domain resampling
"""

import math
import time

import numpy as np
import torch
from scipy.interpolate import interp1d
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Nonlinearity (erf matched to tanh at origin)
# ---------------------------------------------------------------------------

_SQRT_PI_2 = math.sqrt(math.pi) / 2


def _erf_nonlinearity(x):
    return torch.erf(_SQRT_PI_2 * x)


# ---------------------------------------------------------------------------
# Full-timeseries simulation (reference implementation)
# ---------------------------------------------------------------------------


def simulate_rnn(
    N,
    g,
    dt,
    T,
    n_realizations,
    n_ics,
    nonlinearity,
    T_burn_in,
    T_save,
    return_activations,
    device,
    seed,
    return_cpu=False,
):
    """Simulate a random RNN with the permutation trick.

    All times are in physical units (tau = 1).

    Args:
        N: int — number of neurons
        g: float — coupling strength (chaos transition at g=1)
        dt: float — Euler step size in physical time
        T: float — total simulation time
        n_realizations: int — number of effective J realizations (via permutation trick)
        n_ics: int — number of initial conditions per realization
        nonlinearity: callable — e.g. torch.tanh
        T_burn_in: float — transient time to discard
        T_save: float — time interval between saved snapshots
        return_activations: bool — if True return phi=nonlinearity(x), else return x
        device: str or torch.device — e.g. "cuda" or "cpu"
        seed: int — random seed
        return_cpu: bool — if True, allocate output on CPU and transfer at each
            save step (needed when output won't fit in GPU memory)

    Returns:
        (out, J_eff) where
        out: (n_realizations, n_ics, n_save, N) — collected activity
        J_eff: (n_realizations, N, N) — effective coupling matrices
    """
    torch.manual_seed(seed)
    device = torch.device(device)

    n_euler_steps = round(T / dt)
    n_burn_in = round(T_burn_in / dt)
    save_iter = round(T_save / dt)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    J = torch.randn(N, N, device=device, generator=gen) * (g / N**0.5)

    perm1 = torch.stack([torch.randperm(N) for _ in range(n_realizations)]).to(device)
    perm2 = torch.stack([torch.randperm(N) for _ in range(n_realizations)]).to(device)

    perm2_inv = torch.argsort(perm2, dim=1)
    J_eff = J[perm1[:, :, None], perm2_inv[:, None, :]]

    perm1_exp = perm1.unsqueeze(1).expand(n_realizations, n_ics, N)
    perm2_exp = perm2.unsqueeze(1).expand(n_realizations, n_ics, N)

    x = torch.randn(n_realizations, n_ics, N, device=device, generator=gen) * g

    n_save = (n_euler_steps - n_burn_in) // save_iter
    out = torch.empty(n_realizations, n_ics, n_save, N,
                      device='cpu' if return_cpu else device)

    collect_idx = 0
    with torch.no_grad():
        for t in tqdm(range(n_euler_steps), desc="simulate_rnn"):
            phi = nonlinearity(x)

            if t >= n_burn_in and (t - n_burn_in) % save_iter == 0:
                data = phi if return_activations else x
                out[:, :, collect_idx, :] = data.cpu() if return_cpu else data
                collect_idx += 1

            phi_perm = phi.gather(2, perm2_exp)
            eta = phi_perm @ J.T
            eta = eta.gather(2, perm1_exp)

            x.mul_(1 - dt).add_(eta, alpha=dt)

    return out.cpu(), J_eff.cpu()


# ---------------------------------------------------------------------------
# Streaming simulation with inline covariance computation
# ---------------------------------------------------------------------------


def simulate_and_compute_covariance(
    N, g, dt, T, T_burn_in, T_save, n_ics, n_lags,
    alpha_dmft, n_spectrum_ics, device, seed,
):
    """Simulate RNN and compute covariance statistics in a single streaming pass.

    Runs an inline Euler loop with erf nonlinearity. Computes lagged
    cross-covariance via circular buffer, equal-time residual covariance
    via addmm_ accumulation, and scalar power spectrum from a small
    subset of ICs.

    Args:
        N: int — number of neurons
        g: float — coupling strength (chaos transition at g=1)
        dt: float — Euler step size (physical time, tau=1 units)
        T: float — total simulation time
        T_burn_in: float — transient time to discard
        T_save: float — interval between saved snapshots
        n_ics: int — number of initial conditions
        n_lags: int — number of positive lags for cross-covariance
        alpha_dmft: float — DMFT gain <phi'(x)> (for residual: Delta = phi - alpha*x)
        n_spectrum_ics: int — number of ICs whose full timeseries is stored
            for power spectrum computation (stored on CPU)
        device: str or torch.device — e.g. "cuda" or "cpu"
        seed: int — random seed

    Returns:
        dict with keys:
            C_phi: (n_lags+1, N, N) float32 on CPU — lagged cross-covariance
            C_res_0: (N, N) float32 on CPU — equal-time residual covariance
            C_phi_scalar_ft_sim: (n_save,) float32 on CPU — scalar power spectrum
            J: (N, N) float32 on CPU — coupling matrix
            sim_time: float — wall-clock seconds for simulation loop
    """
    device = torch.device(device)

    # Convert physical times to discrete steps
    n_euler_steps = round(T / dt)
    n_burn_in = round(T_burn_in / dt)
    save_iter = round(T_save / dt)
    n_save = (n_euler_steps - n_burn_in) // save_iter
    n_spectrum_ics = min(n_spectrum_ics, n_ics)

    # Generate J and initial conditions
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    J = torch.randn(N, N, device=device, generator=gen) * (g / N**0.5)
    x = torch.randn(n_ics, N, device=device, generator=gen) * g

    # Allocate accumulators on GPU
    L = n_lags + 1  # circular buffer length
    phi_buf = torch.zeros(L, n_ics, N, device=device)
    C_phi = torch.zeros(L, N, N, device=device)
    C_res_0 = torch.zeros(N, N, device=device)

    # CPU storage for spectrum ICs
    phi_spectrum = torch.zeros(n_spectrum_ics, n_save, N) if n_spectrum_ics > 0 else None

    save_count = 0
    print_every = max(1, n_euler_steps // 20)

    t0 = time.time()
    with torch.no_grad():
        for step in range(n_euler_steps):
            if step % print_every == 0:
                elapsed = time.time() - t0
                if step > 0:
                    eta_s = elapsed / step * (n_euler_steps - step)
                    print(f"  step {step}/{n_euler_steps} ({100*step/n_euler_steps:.0f}%) "
                          f"ETA {eta_s:.0f}s", flush=True)
                else:
                    print(f"  step {step}/{n_euler_steps} (0%)", flush=True)

            phi = _erf_nonlinearity(x)

            if step >= n_burn_in and (step - n_burn_in) % save_iter == 0:
                s = save_count
                buf_idx = s % L

                # Write current phi into circular buffer
                phi_buf[buf_idx] = phi

                # Accumulate equal-time residual covariance
                res = phi - alpha_dmft * x
                C_res_0.addmm_(res.T, res)
                del res

                # Accumulate lagged covariances
                max_lag = min(s, n_lags)
                for lag in range(max_lag + 1):
                    old_idx = (s - lag) % L
                    C_phi[lag].addmm_(phi_buf[old_idx].T, phi_buf[buf_idx])

                # Store spectrum ICs on CPU
                if phi_spectrum is not None:
                    phi_spectrum[:, s, :] = phi[:n_spectrum_ics].cpu()

                save_count += 1

            # Euler step: x <- (1-dt)*x + dt*(phi @ J^T)
            x.mul_(1 - dt).addmm_(phi, J.t(), alpha=dt)
            del phi

    sim_time = time.time() - t0

    # Normalize accumulators
    for lag in range(L):
        pair_count = save_count - lag
        if pair_count > 0:
            C_phi[lag] /= pair_count * n_ics
    C_phi[0] = (C_phi[0] + C_phi[0].T) / 2

    C_res_0 /= save_count * n_ics
    C_res_0 = (C_res_0 + C_res_0.T) / 2

    # Compute empirical power spectrum from spectrum ICs
    if phi_spectrum is not None and n_spectrum_ics > 0:
        n_rfft = n_save // 2 + 1
        rfft_spec = torch.fft.rfft(phi_spectrum, dim=1)  # (n_spectrum_ics, n_rfft, N)
        C_phi_scalar_ft_half = (rfft_spec.abs() ** 2).mean(dim=(0, 2)) / n_save
        C_phi_scalar_ft_sim = torch.zeros(n_save)
        C_phi_scalar_ft_sim[:n_rfft] = C_phi_scalar_ft_half
        if n_save % 2 == 0:
            C_phi_scalar_ft_sim[n_rfft:] = C_phi_scalar_ft_half[1:-1].flip(0)
        else:
            C_phi_scalar_ft_sim[n_rfft:] = C_phi_scalar_ft_half[1:].flip(0)
        del rfft_spec, C_phi_scalar_ft_half
    else:
        C_phi_scalar_ft_sim = torch.zeros(n_save)

    # Move everything to CPU
    C_phi = C_phi.cpu()
    C_res_0 = C_res_0.cpu()
    J_cpu = J.cpu()

    del x, phi_buf, J
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        C_phi=C_phi,
        C_res_0=C_res_0,
        C_phi_scalar_ft_sim=C_phi_scalar_ft_sim,
        J=J_cpu,
        sim_time=sim_time,
    )


# ---------------------------------------------------------------------------
# Post-hoc covariance helpers (for use with simulate_rnn output)
# ---------------------------------------------------------------------------


def compute_lagged_covariance(activity, n_lags, center=False):
    """Compute time-lagged covariance matrices, pooling across ICs.

    Args:
        activity: (n_realizations, n_ics, n_save, N) — neural activity
        n_lags: int — number of positive lags (returns lags 0..n_lags)
        center: bool — if True, subtract temporal means before computing covariance

    Returns:
        C: (n_realizations, n_lags+1, N, N) — covariance at each lag
    """
    n_realizations, n_ics, n_save, N = activity.shape
    device = activity.device

    C = torch.zeros(n_realizations, n_lags + 1, N, N, device=device)

    for inst in range(n_realizations):
        for ic in range(n_ics):
            a = activity[inst, ic]  # (n_save, N)
            if center:
                a = a - a.mean(dim=0, keepdim=True)
            for lag in range(n_lags + 1):
                n_pairs = n_save - lag
                a_left = a[:n_pairs]
                a_right = a[lag:lag + n_pairs]
                C[inst, lag] += (a_left.T @ a_right) / n_pairs

        C[inst] /= n_ics

    C[:, 0] = (C[:, 0] + C[:, 0].transpose(1, 2)) / 2

    return C


def compute_equal_time_covariance(activity):
    """Compute equal-time covariance matrix, pooling across ICs and time.

    Args:
        activity: (n_realizations, n_ics, n_save, N) — neural activity

    Returns:
        C: (n_realizations, N, N) — one covariance matrix per instance
    """
    n_realizations, n_ics, n_save, N = activity.shape
    n_samples = n_ics * n_save

    a = activity.reshape(n_realizations, n_samples, N)
    mean = a.mean(dim=1)

    C = torch.bmm(a.transpose(1, 2), a) / n_samples - torch.bmm(
        mean.unsqueeze(2), mean.unsqueeze(1)
    )
    C = (C + C.transpose(1, 2)) / 2

    return C


# ---------------------------------------------------------------------------
# Theory prediction
# ---------------------------------------------------------------------------


def predict_covariance_theory(J, alpha, g, C_phi_scalar_ft, dt_spectrum, n_lags,
                              dt_out=None):
    """Predict covariance matrices from the linear-system ansatz.

    C^phi(omega) = sigma^2(omega) * M(omega)^{-1} M(omega)^{-H}
    M(omega) = (1/S^phi(omega)^*) I - J
    S^phi(omega) = alpha / (1 + i*omega)
    sigma^2(omega) = (1 - g^2 |S^phi|^2) C_phi_scalar(omega) / |S^phi|^2

    Loops one frequency at a time, accumulating the inverse Fourier transform.

    Args:
        J: (N, N) tensor — coupling matrix
        alpha: float — DMFT gain <phi'(x)>
        g: float — coupling strength
        C_phi_scalar_ft: (n_omega,) real tensor — DFT of single-site autocorrelation
        dt_spectrum: float — time spacing of the frequency grid
        n_lags: int — number of positive lags to compute
        dt_out: float — time spacing of output lags (default: dt_spectrum)

    Returns:
        C_tau: (n_lags+1, N, N) real tensor — predicted covariance at each lag
    """
    if dt_out is None:
        dt_out = dt_spectrum

    device = J.device
    N = J.shape[0]
    n_omega = C_phi_scalar_ft.shape[0]

    freq = torch.fft.fftfreq(n_omega, d=dt_spectrum, device=device)
    omega = 2 * torch.pi * freq

    tau = torch.tensor([l * dt_out for l in range(n_lags + 1)], device=device)

    C_tau = torch.zeros(n_lags + 1, N, N, device=device)

    I = torch.eye(N, device=device, dtype=torch.complex64)
    J_c = J.to(torch.complex64)

    for k in tqdm(range(n_omega), desc="predict_covariance_theory"):
        omega_k = omega[k]

        S_phi = alpha / (1 + 1j * omega_k)
        S_phi_abs2 = S_phi.real ** 2 + S_phi.imag ** 2

        sigma2 = (1 - g**2 * S_phi_abs2) * C_phi_scalar_ft[k] / S_phi_abs2

        M = (1.0 / S_phi.conj()) * I - J_c
        M_inv = torch.linalg.inv(M)

        C_phi_mat_ft = sigma2.to(torch.complex64) * (M_inv @ M_inv.conj().T)

        for l in range(n_lags + 1):
            phase = torch.exp(1j * omega_k * tau[l])
            C_tau[l] += (phase * C_phi_mat_ft).real

    C_tau /= n_omega

    return C_tau


# ---------------------------------------------------------------------------
# DMFT-to-frequency-grid resampling
# ---------------------------------------------------------------------------


def compute_C_phi_scalar_ft_dmft(tau_vals_dmft, C_phi_scalar_dmft, tau_max, omega_max):
    """Resample DMFT autocorrelation onto the theory frequency grid.

    Takes the real-space DMFT autocorrelation C^phi(tau) and produces its
    discrete Fourier transform on a grid determined by tau_max and omega_max.

    The frequency grid has:
        dt_theory = pi / omega_max
        n_omega = ceil(2 * tau_max * omega_max / pi)
        delta_omega = 2*pi / (n_omega * dt_theory) = pi / tau_max  (approx)

    The autocorrelation is interpolated onto a "wrapped" lag grid that maps
    each frequency bin index to its corresponding lag via the DFT convention
    (positive lags for the first half, negative/wrapped lags for the second half).

    Args:
        tau_vals_dmft: (N_tau,) numpy array — lag values from DMFT ODE
        C_phi_scalar_dmft: (N_tau,) numpy array — single-site autocorrelation C^phi(tau)
        tau_max: float — maximum lag time for the theory frequency grid
        omega_max: float — maximum frequency for the theory grid

    Returns:
        C_phi_scalar_ft: (n_omega,) float32 torch tensor — DFT of resampled autocorrelation
        n_omega: int — number of frequency bins
        dt_theory: float — time spacing of the frequency grid
    """
    dt_theory = math.pi / omega_max
    n_omega = math.ceil(2 * tau_max * omega_max / math.pi)

    # Wrapped lag grid: index k maps to lag |k| or |n_omega - k|, whichever is smaller
    indices = np.arange(n_omega)
    effective_lag = np.minimum(indices, n_omega - indices) * dt_theory

    # Interpolate DMFT autocorrelation onto this grid
    interp_fn = interp1d(tau_vals_dmft, C_phi_scalar_dmft,
                         bounds_error=False, fill_value=0.0)
    C_phi_scalar_wrapped = interp_fn(effective_lag)

    # FFT gives the DFT of the wrapped (symmetric) autocorrelation
    C_phi_scalar_ft = torch.from_numpy(
        np.fft.fft(C_phi_scalar_wrapped).real
    ).float()

    return C_phi_scalar_ft, n_omega, dt_theory
