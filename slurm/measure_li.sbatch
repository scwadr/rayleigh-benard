#!/bin/bash
#SBATCH --job-name=rb_measure_li
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/scratch1/2/ashelestov3/%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
while [[ "$REPO_ROOT" != / && ! -f "$REPO_ROOT/run_filter_sweep.py" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/run_filter_sweep.py" ]] || { echo 'Cannot find run_filter_sweep.py' >&2; exit 1; }
runtime="/storage/scratch1/2/ashelestov3/rayleigh-benard/li_runtime/$SLURM_JOB_ID"
mkdir -p "$runtime/tmp" "$runtime/cache" "$runtime/pycache"
export TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/cache" PYTHONPYCACHEPREFIX="$runtime/pycache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load anaconda3
source activate "${SPIDER_ENV:-SPIDER}"
cd "$REPO_ROOT"
srun --ntasks=1 python -u run_filter_sweep.py --measure-li --workers "$SLURM_CPUS_PER_TASK" "$@"
