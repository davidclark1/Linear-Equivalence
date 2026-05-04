"""Combined batch experiment: covariance prediction + residual scaling.

Runs DMFT → streaming simulation → theory prediction → diagnostics → save
for each task in a grid of (g, N, sampling_ratio, seed) values.

Each SLURM array element runs 10 seeds for one (g, N, ratio) combo.

Usage:
    python sims/run_combined_sweep.py --task-id 0 --output-dir /path/to/dir
    python sims/run_combined_sweep.py --list-tasks --output-dir /path/to/dir
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_site_dmft import compute_Delta_0, integrate_potential, compute_phi_prime_avg, compute_C_simple, fix
from rnn import simulate_and_compute_covariance, predict_covariance_theory, compute_C_phi_scalar_ft_dmft


# --- Task grid ---

G_VALUES = [1.5, 2.5]
N_VALUES = np.round(np.geomspace(100, 10000, 7)).astype(int).tolist()
ALPHA_VALUES = [50, 200, 800]
N_SEEDS = 10


def build_tasks():
    """Build list of task parameter dicts.

    Grid: 2 g x 7 N x 3 alpha x 10 seeds = 420 tasks.
    Grouped by (g, N, alpha): 42 SLURM tasks, each running 10 seeds.
    """
    tasks = []
    for g in G_VALUES:
        for N in N_VALUES:
            for sampling_ratio in ALPHA_VALUES:
                for run_idx in range(N_SEEDS):
                    tasks.append(dict(g=g, N=N, sampling_ratio=sampling_ratio,
                                      run_idx=run_idx))
    return tasks


# --- Fixed parameters ---

DT = 0.025
T = 5500.0
T_BURN_IN = 500.0
T_SAVE = 0.5
N_LAGS = 20
BLOCK_SIZE = 1000
TAU_MAX = 100.0
OMEGA_MAX = 10.0
N_SPECTRUM_ICS = 10
DMFT_TAU_MAX = 200.0
DMFT_N_TAU = 100000


def run_task(task_id, output_dir):
    """Run full pipeline for one (g, N, ratio, seed) combo."""
    tasks = build_tasks()
    task = tasks[task_id]
    N = task['N']
    g = task['g']
    sampling_ratio = task['sampling_ratio']
    run_idx = task['run_idx']
    seed = task_id

    T_collect = T - T_BURN_IN
    n_ics = max(1, round(sampling_ratio * N / T_collect))
    n_save = (round(T / DT) - round(T_BURN_IN / DT)) // round(T_SAVE / DT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = min(N, BLOCK_SIZE)

    print(f"=== Task {task_id}: N={N}, g={g}, ratio={sampling_ratio}, "
          f"run_idx={run_idx}, n_ics={n_ics}, n_save={n_save}, seed={seed} ===",
          flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # --- 1. DMFT ---
    t0 = time.time()
    Delta_0 = compute_Delta_0(g)
    alpha_dmft = compute_phi_prime_avg(Delta_0)

    tau_vals_dmft, Delta_dmft = integrate_potential(
        Delta_0, g, tau_max=DMFT_TAU_MAX, N_tau=DMFT_N_TAU)
    Delta_dmft = fix(Delta_dmft)
    C_phi_scalar_dmft = compute_C_simple(Delta_0, Delta_dmft)

    C_phi_scalar_ft_dmft, n_omega, dt_theory = compute_C_phi_scalar_ft_dmft(
        tau_vals_dmft, C_phi_scalar_dmft, TAU_MAX, OMEGA_MAX)

    dmft_time = time.time() - t0
    print(f"DMFT: Delta_0={Delta_0:.6f}, alpha={alpha_dmft:.6f}, "
          f"n_omega={n_omega}, time={dmft_time:.1f}s", flush=True)

    # --- 2. Streaming simulation + covariance ---
    print("Starting simulation...", flush=True)
    result = simulate_and_compute_covariance(
        N=N, g=g, dt=DT, T=T, T_burn_in=T_BURN_IN, T_save=T_SAVE,
        n_ics=n_ics, n_lags=N_LAGS, alpha_dmft=alpha_dmft,
        n_spectrum_ics=N_SPECTRUM_ICS, device=device, seed=seed,
    )
    C_phi = result['C_phi']
    C_res_0 = result['C_res_0']
    C_phi_scalar_ft_sim = result['C_phi_scalar_ft_sim']
    J = result['J']
    sim_time = result['sim_time']
    print(f"Simulation: {sim_time:.1f}s", flush=True)

    # --- 3. Theory prediction ---
    t0 = time.time()
    J_gpu = J.to(device)
    C_phi_scalar_ft_dmft_gpu = C_phi_scalar_ft_dmft.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    C_theory = predict_covariance_theory(
        J_gpu, alpha_dmft, g, C_phi_scalar_ft_dmft_gpu,
        dt_theory, N_LAGS, dt_out=T_SAVE,
    )

    gpu_peak_theory_gb = (torch.cuda.max_memory_allocated() / 1e9
                          if torch.cuda.is_available() else 0.0)
    C_theory = C_theory.cpu()
    del J_gpu, C_phi_scalar_ft_dmft_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    theory_time = time.time() - t0
    print(f"Theory: {n_omega} inversions, time={theory_time:.1f}s, "
          f"GPU peak={gpu_peak_theory_gb:.2f} GB", flush=True)

    # --- 4. Diagnostics ---
    C_phi_0 = C_phi[0].numpy()
    C_theory_0 = C_theory[0].numpy()
    C_res_0_np = C_res_0.numpy()
    J_np = J.numpy()

    mask = ~np.eye(N, dtype=bool)
    offdiag_phi = C_phi_0[mask]
    offdiag_theory = C_theory_0[mask]
    offdiag_res = C_res_0_np[mask]

    pearson_r = float(np.corrcoef(offdiag_phi, offdiag_theory)[0, 1])
    offdiag_std_phi = float(offdiag_phi.std())
    offdiag_std_res = float(offdiag_res.std())
    offdiag_rms_res = float(np.sqrt(np.mean(offdiag_res**2)))
    res_phi_ratio = offdiag_std_res / offdiag_std_phi

    tr_C_phi_0 = float(C_phi_0.trace() / N)
    tr_C_theory_0 = float(C_theory_0.trace() / N)
    tr_dmft = float(C_phi_scalar_ft_dmft.sum().item() / n_omega)

    tr_emp_lags = np.array([C_phi[l].numpy().trace() / N for l in range(N_LAGS + 1)])
    tr_theory_lags = np.array([C_theory[l].numpy().trace() / N for l in range(N_LAGS + 1)])

    print(f"Pearson r = {pearson_r:.6f}", flush=True)
    print(f"Off-diag std: phi={offdiag_std_phi:.6f}, res={offdiag_std_res:.6f}, "
          f"rms_res={offdiag_rms_res:.6f}, ratio={res_phi_ratio:.4f}", flush=True)
    print(f"Tr(C)/N: emp={tr_C_phi_0:.6f}, theory={tr_C_theory_0:.6f}, "
          f"DMFT={tr_dmft:.6f}", flush=True)

    del offdiag_phi, offdiag_theory, offdiag_res, mask

    # --- 5. Save ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"results_{task_id:03d}.npz")

    np.savez_compressed(
        out_path,
        # Params
        N=N, g=g, run_idx=run_idx, sampling_ratio=sampling_ratio,
        dt=DT, T=T, T_burn_in=T_BURN_IN, T_save=T_SAVE,
        n_ics=n_ics, n_save=n_save, n_omega=n_omega, dt_theory=dt_theory,
        tau_max=TAU_MAX, omega_max=OMEGA_MAX, seed=seed, n_lags=N_LAGS,
        # DMFT
        Delta_0=Delta_0, alpha_dmft=alpha_dmft,
        tau_vals_dmft=tau_vals_dmft, C_phi_scalar_dmft=C_phi_scalar_dmft,
        C_phi_scalar_ft_dmft=C_phi_scalar_ft_dmft.numpy(),
        # Single-site spectrum
        C_phi_scalar_ft_sim=C_phi_scalar_ft_sim.numpy(),
        # Covariance blocks
        C_phi_block=C_phi[:, :B, :B].numpy(),
        C_theory_block=C_theory[:, :B, :B].numpy(),
        C_res_0_block=C_res_0_np[:B, :B],
        J_block=J_np[:B, :B],
        # Trace curves
        tr_emp_lags=tr_emp_lags,
        tr_theory_lags=tr_theory_lags,
        # Scalars
        pearson_r=pearson_r,
        offdiag_std_phi=offdiag_std_phi,
        offdiag_std_res=offdiag_std_res,
        offdiag_rms_res=offdiag_rms_res,
        res_phi_ratio=res_phi_ratio,
        tr_C_phi_0=tr_C_phi_0,
        tr_C_theory_0=tr_C_theory_0,
        tr_dmft=tr_dmft,
        sim_time=sim_time,
        theory_time=theory_time,
        gpu_peak_theory_gb=gpu_peak_theory_gb,
    )

    file_size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved: {out_path} ({file_size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Combined covariance + residual sweep")
    parser.add_argument("--task-id", type=int, help="Task index (0-based)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for .npz outputs")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print task list and exit")
    args = parser.parse_args()

    tasks = build_tasks()

    if args.list_tasks:
        T_collect = T - T_BURN_IN
        for i, t in enumerate(tasks):
            n_ics = max(1, round(t['sampling_ratio'] * t['N'] / T_collect))
            print(f"{i:3d}: N={t['N']:5d}  g={t['g']:.1f}  "
                  f"ratio={t['sampling_ratio']:3d}  run={t['run_idx']}  n_ics={n_ics}")
        print(f"\nTotal: {len(tasks)} tasks "
              f"({len(tasks) // N_SEEDS} SLURM tasks x {N_SEEDS} seeds)")
        return

    if args.task_id is None:
        parser.error("--task-id is required (unless --list-tasks)")

    if args.task_id < 0 or args.task_id >= len(tasks):
        print(f"Error: task_id {args.task_id} out of range [0, {len(tasks) - 1}]")
        sys.exit(1)

    run_task(args.task_id, args.output_dir)


if __name__ == "__main__":
    main()
