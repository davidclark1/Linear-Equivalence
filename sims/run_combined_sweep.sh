#!/bin/bash
# =============================================================================
# Combined-sweep SLURM submission script
# =============================================================================
#
# CUSTOMIZE THE FOLLOWING FOR YOUR CLUSTER:
#  - --account, --partition       SLURM allocation
#  - --output                     log path (must exist before submission)
#  - module / conda activation    env setup commands
#  - OUTPUT_DIR                   shared writable directory for .npz results
#
# Each SLURM array task runs 10 seeds (1 (g, N, alpha) combo). 42 tasks × 10 = 420.
# Memory peak ~10 GB GPU at N=10000, alpha=800.
# =============================================================================

#SBATCH --job-name=combined-sweep
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --array=0-41
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%A_%a.out

set -e

# --- Environment setup (customize) ---
ml python cuda
eval "$(conda shell.bash hook)"
mamba activate <YOUR_ENV>

# --- Output directory (customize) ---
OUTPUT_DIR="./results/combined_sweep"
mkdir -p "$OUTPUT_DIR" logs

# Each array task runs 10 seeds for one (g, N, alpha) combo
START=$((SLURM_ARRAY_TASK_ID * 10))
for i in $(seq 0 9); do
    TASK_ID=$((START + i))
    echo "=== Running task $TASK_ID (seed $i of 10) ==="
    python sims/run_combined_sweep.py --task-id "$TASK_ID" --output-dir "$OUTPUT_DIR"
done
