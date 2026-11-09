#!/bin/bash
# One array task = one line of $NZM_TASKS: a script and its arguments, e.g. "run_one.py pm_a200 0 1000" or
# "cavity_stats.py 1000 800 0".  Several runs may share a line, separated by ';' (they then run one after another).
# run_one.py and cavity_stats.py skip runs whose result files already exist, so a requeued or resubmitted array only does
# what is missing.
#
# Edit the #SBATCH lines and the environment setup for your cluster.  Each run needs one GPU (the paper's runs used
# A100 or H100 GPUs with 80 GB of memory); the longest runs take up to about 10 hours.
#
# Example (from the repository root):
#   NZM_TASKS=slurm/tasks_scs.txt NZM_RUNS=results/moments sbatch --array=0-104 slurm/run_tasks.sh
#SBATCH --job-name=linear-equivalence
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/%x_%A_%a.out

# environment: activate the Python environment with the packages of requirements.txt, e.g.
# source /path/to/venv/bin/activate

: "${NZM_TASKS:?set NZM_TASKS to a task file}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$NZM_TASKS")
echo "task $SLURM_ARRAY_TASK_ID: $LINE on $(hostname)"
fail=0
IFS=';' read -ra RUNS <<< "$LINE"
for R in "${RUNS[@]}"; do
    R=$(echo $R)                                   # trim
    [ -z "$R" ] && continue
    echo "=== $R  $(date +%T)"
    python -u $R || { echo "FAILED: $R"; fail=1; }
done
exit $fail
