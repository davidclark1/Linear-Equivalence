"""Residual covariance sweep at high sampling ratios (alpha=3200, 12800).

Lean inline Euler loop accumulating only C_res_0. No lagged covariance,
no theory prediction, no spectrum. One seed per task for maximum parallelism.

Usage:
    python sims/run_residual_highalpha.py --task-id 0 --output-dir /path/to/dir
    python sims/run_residual_highalpha.py --list-tasks --output-dir /path/to/dir
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from single_site_dmft import compute_Delta_0, compute_phi_prime_avg


# --- Task grid ---

G = 2.5
N_VALUES = np.round(np.geomspace(100, 10000, 7)).astype(int).tolist()
ALPHA_VALUES = [3200, 12800]
N_SEEDS = 5


def build_tasks():
    """7 N x 2 alpha x 5 seeds = 70 tasks, one seed per task."""
    tasks = []
    for N in N_VALUES:
        for sampling_ratio in ALPHA_VALUES:
            for run_idx in range(N_SEEDS):
                tasks.append(dict(N=N, sampling_ratio=sampling_ratio, run_idx=run_idx))
    return tasks


# --- Fixed parameters ---

DT = 0.025
T = 5500.0
T_BURN_IN = 500.0
T_SAVE = 0.5
BLOCK_SIZE = 1000


def run_task(task_id, output_dir):
    """Run residual covariance measurement for one (N, alpha, seed)."""
    tasks = build_tasks()
    task = tasks[task_id]
    N = task['N']
    sampling_ratio = task['sampling_ratio']
    run_idx = task['run_idx']
    seed = task_id

    T_collect = T - T_BURN_IN
    n_ics = max(1, round(sampling_ratio * N / T_collect))
    n_euler_steps = round(T / DT)
    n_burn_in = round(T_BURN_IN / DT)
    save_iter = round(T_SAVE / DT)
    n_save = (n_euler_steps - n_burn_in) // save_iter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = min(N, BLOCK_SIZE)

    print(f"=== Task {task_id}: N={N}, g={G}, ratio={sampling_ratio}, "
          f"run_idx={run_idx}, n_ics={n_ics}, n_save={n_save}, seed={seed} ===",
          flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # --- 1. DMFT (alpha only) ---
    Delta_0 = compute_Delta_0(G)
    alpha_dmft = compute_phi_prime_avg(Delta_0)
    print(f"DMFT: Delta_0={Delta_0:.6f}, alpha={alpha_dmft:.6f}", flush=True)

    # --- 2. Generate J and initial conditions ---
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    J = torch.randn(N, N, device=device, generator=gen) * (G / N**0.5)
    x = torch.randn(n_ics, N, device=device, generator=gen) * G

    # --- 3. Euler loop with inline C_res accumulation ---
    t0 = time.time()
    C_res = torch.zeros(N, N, device=device)
    sqrt_pi_2 = math.sqrt(math.pi) / 2
    count = 0
    print_every = max(1, n_euler_steps // 20)

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

            phi = torch.erf(sqrt_pi_2 * x)

            if step >= n_burn_in and (step - n_burn_in) % save_iter == 0:
                res = phi - alpha_dmft * x
                C_res.addmm_(res.T, res)
                count += 1
                del res

            x.mul_(1 - DT).addmm_(phi, J.t(), alpha=DT)
            del phi

    C_res /= (n_ics * count)
    C_res = (C_res + C_res.T) / 2

    sim_time = time.time() - t0
    print(f"Simulation: {count} save steps, time={sim_time:.1f}s", flush=True)

    # --- 4. Diagnostics ---
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    offdiag = C_res[mask]
    offdiag_rms_res = float(offdiag.pow(2).mean().sqrt())
    diag_mean_res = float(C_res.diagonal().mean())
    del offdiag, mask
    print(f"Off-diag RMS: {offdiag_rms_res:.8f}, Diag mean: {diag_mean_res:.6f}",
          flush=True)

    # --- 5. Save ---
    C_res_block = C_res[:B, :B].cpu().numpy()
    J_block = J[:B, :B].cpu().numpy()
    del C_res, J, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"residual_{task_id:03d}.npz")

    np.savez_compressed(
        out_path,
        N=N, g=G, sampling_ratio=sampling_ratio, run_idx=run_idx, seed=seed,
        dt=DT, T=T, T_burn_in=T_BURN_IN, T_save=T_SAVE,
        n_ics=n_ics, n_save=count, alpha=alpha_dmft,
        C_res_0_block=C_res_block,
        J_block=J_block,
        offdiag_rms_res=offdiag_rms_res,
        diag_mean_res=diag_mean_res,
        sim_time=sim_time,
    )

    file_size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved: {out_path} ({file_size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="High-alpha residual covariance sweep")
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
            print(f"{i:3d}: N={t['N']:5d}  ratio={t['sampling_ratio']:5d}  "
                  f"run={t['run_idx']}  n_ics={n_ics}")
        print(f"\nTotal: {len(tasks)} tasks")
        return

    if args.task_id is None:
        parser.error("--task-id is required (unless --list-tasks)")

    if args.task_id < 0 or args.task_id >= len(tasks):
        print(f"Error: task_id {args.task_id} out of range [0, {len(tasks) - 1}]")
        sys.exit(1)

    run_task(args.task_id, args.output_dir)


if __name__ == "__main__":
    main()
