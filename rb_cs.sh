#!/bin/bash
#SBATCH --job-name=dedalus_mpi_setup
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=08:00:00
#SBATCH --output=dedalus_%j.out
#SBATCH --partition=pace-phoenix  # Adjust for Phoenix/Savannah

# Load modules (adjust for your cluster)
module load anaconda3
module load openmpi

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh

# Activate Dedalus environment
conda activate dedalus3

# Run Dedalus with MPI
mpirun -np $SLURM_NTASKS python rayleigh_benard-condition_setup.py