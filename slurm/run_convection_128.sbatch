#!/bin/bash
#SBATCH --job-name=dedalus_128
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

# 1. Load Environment
module load anaconda3
source activate dedalus3

# --- FIX START ---
# 2. Unload the tracking module causing the conflict
module unload xalt

# 3. Force Linux to look in your Conda environment for libraries FIRST
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# --- FIX END ---

# 4. Optimization for 128 Cores
export OMP_NUM_THREADS=1
export DEDALUS_FFTW_PLAN_RIGOR=ESTIMATE

echo "Job started on $(hostname) at $(date)"
echo "Running with 128 MPI tasks..."

# 5. Run the Script
REPO_ROOT="$SLURM_SUBMIT_DIR"
while [[ "$REPO_ROOT" != / && ! -d "$REPO_ROOT/dedalus" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -d "$REPO_ROOT/dedalus" ]] || { echo "Could not locate repository root" >&2; exit 1; }
cd "$REPO_ROOT" || exit 1
# Adding -u to force print statements to appear immediately in the log
mpiexec -n 128 python3 -u dedalus/rayleigh_benard-condition_setup_128.py

echo "Job finished at $(date)"
