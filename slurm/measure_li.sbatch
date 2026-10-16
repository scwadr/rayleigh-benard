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
REPO_ROOT="$(realpath "$REPO_ROOT")"
runtime="${WORKFLOW_RUNTIME_ROOT:-/storage/scratch1/2/ashelestov3/rayleigh-benard/li_runtime}/$SLURM_JOB_ID"
runtime="$(realpath -m "$runtime")"
case "$runtime" in
    /storage/scratch1/2/ashelestov3/*) ;;
    *) echo 'Runtime must be on scratch' >&2; exit 1 ;;
esac
mkdir -p "$runtime/tmp" "$runtime/cache" "$runtime/pycache"
export TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/cache" PYTHONPYCACHEPREFIX="$runtime/pycache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$runtime/numba" "$runtime/joblib" "$runtime/cuda" "$runtime/config" "$runtime/state" "$runtime/data"
export NUMBA_CACHE_DIR="$runtime/numba" JOBLIB_TEMP_FOLDER="$runtime/joblib" CUDA_CACHE_PATH="$runtime/cuda"
export XDG_CONFIG_HOME="$runtime/config" XDG_STATE_HOME="$runtime/state" XDG_DATA_HOME="$runtime/data"
export TMP="$TMPDIR" TEMP="$TMPDIR" GIT_OPTIONAL_LOCKS=0
cd "$runtime"
module load anaconda3
source activate "${SPIDER_ENV:-SPIDER}"
cd "$runtime"
srun --ntasks=1 python -u "$REPO_ROOT/run_filter_sweep.py" --measure-li --workers "$SLURM_CPUS_PER_TASK" "$@"
