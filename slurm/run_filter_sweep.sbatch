#!/bin/bash
# One independent process per filter scale; serialize by default to limit memory demand.
#SBATCH --job-name=rb_filter_check
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --array=0-3%1
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
module load anaconda3
source activate "${SPIDER_ENV:-SPIDER}"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
while [[ "$REPO_ROOT" != / && ! -f "$REPO_ROOT/run_filter_sweep.py" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/run_filter_sweep.py" ]] || { echo 'Cannot find run_filter_sweep.py' >&2; exit 1; }
cd "$REPO_ROOT"

# delta/dx, with sigma=delta/sqrt(12). Override both FILTER_SCALES and --array together.
read -r -a scales <<< "${FILTER_SCALES:-10 25 50 75}"
task="${SLURM_ARRAY_TASK_ID:-0}"
(( task < ${#scales[@]} )) || { echo "No filter scale for array index $task" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
echo "Job $SLURM_JOB_ID task $task: delta/dx=${scales[$task]}, workers=$SLURM_CPUS_PER_TASK"
srun --ntasks=1 python -u run_filter_sweep.py \
    --delta-dx "${scales[$task]}" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --data-path "${DATA_PATH:-/storage/scratch1/2/ashelestov3/preprocessed_paper_data.npz}" \
    --output-root "${OUTPUT_ROOT:-thesis_runs}" "$@"
