#!/bin/bash
# =============================================================================
# High-alpha residual-only SLURM submission script
# =============================================================================
#
# CUSTOMIZE THE FOLLOWING FOR YOUR CLUSTER:
#  - --account, --partition       SLURM allocation
#  - --output                     log path (must exist before submission)
#  - module / conda activation    env setup commands
#  - OUTPUT_DIR                   shared writable directory for .npz results
#
# 70 tasks (1 seed each), 7 N × 2 alpha (3200, 12800) × 5 seeds, g=2.5 only.
# Tasks at N=10000 are slow (~6 h on H100); 24 h time limit recommended.
# =============================================================================

#SBATCH --job-name=res-highalpha
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --array=0-69
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%A_%a.out

set -e

# --- Environment setup (customize) ---
ml python cuda
eval "$(conda shell.bash hook)"
mamba activate <YOUR_ENV>

# --- Output directory (customize) ---
OUTPUT_DIR="./results/residual_highalpha"
mkdir -p "$OUTPUT_DIR" logs

# One seed per task for maximum parallelism
python sims/run_residual_highalpha.py --task-id "$SLURM_ARRAY_TASK_ID" --output-dir "$OUTPUT_DIR"
