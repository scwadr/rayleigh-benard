#!/bin/bash
#SBATCH --job-name=rb_convection       # A name for your job
#SBATCH --account=gts-rg119            # Your charge account
#SBATCH --qos=embers                   # The 'embers' QOS queue
#SBATCH --constraint=amd               # Request AMD nodes
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks-per-node=128          # Request all 128 cores on the node
#SBATCH --time=08:00:00                # Set 8 hour time limit
#SBATCH --output=slurm/logs/%x-%j-full.out  # Keep Slurm output under slurm/logs

# 1. Load your modules
module load anaconda3

# Note: If you use a specific conda environment for your Rayleigh-Benard project,
# uncomment the line below and add your environment name:
source activate dedalus3

# 2. Ensure we are in the correct directory
# (SLURM_SUBMIT_DIR points to where you ran the sbatch command)
cd "$SLURM_SUBMIT_DIR"

# 3. Run the Python scripts sequentially
echo "Starting Phase 1: Condition Setup (2048x768)..."
mpiexec -n 64 python3 dedalus/paper-test/rayleigh_benard-setup.py && \
echo "Phase 1 complete. Starting Phase 2: Stabilization (4096x1536)..." && \
mpiexec -n 64 python3 dedalus/paper-test/rayleigh_benard-stabilize.py && \
echo "Phase 2 complete. Starting Phase 3: High-Res Production..." && \
mpiexec -n 64 python3 dedalus/paper-test/rayleigh_benard.py

echo "Full 3-stage DNS pipeline complete."
