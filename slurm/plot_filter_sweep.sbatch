#!/bin/bash
#SBATCH --job-name=rb_filter_figure2
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/scratch1/2/ashelestov3/%x-%j.out

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
while [[ "$REPO_ROOT" != / && ! -f "$REPO_ROOT/run_filter_sweep.py" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/run_filter_sweep.py" ]] || { echo 'Cannot find run_filter_sweep.py' >&2; exit 1; }
runtime="/storage/scratch1/2/ashelestov3/rayleigh-benard/plot_runtime/$SLURM_JOB_ID"
mkdir -p "$runtime/tmp" "$runtime/matplotlib" "$runtime/cache" "$runtime/pycache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib" XDG_CACHE_HOME="$runtime/cache"
export PYTHONPYCACHEPREFIX="$runtime/pycache"
module load anaconda3
source activate "${SPIDER_ENV:-SPIDER}"
cd "$REPO_ROOT"
srun --ntasks=1 python -u run_filter_sweep.py "$@"
