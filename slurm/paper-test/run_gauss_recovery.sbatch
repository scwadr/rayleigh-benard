#!/bin/bash
#SBATCH --job-name=rb_gauss
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load anaconda3
source activate "${SPIDER_CONDA_ENV:-SPIDER}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ -f requirements/chebyshev-gauss.txt ]] || { echo "Submit from repository root" >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export SPIDER_TEST_PARALLEL=0
export SPIDER_DIAGNOSTIC_ONLY="${SPIDER_DIAGNOSTIC_ONLY:-1}"
export SPIDER_TEST_NDOMAINS="${SPIDER_TEST_NDOMAINS:-32}"
export SPIDER_DATA="${SPIDER_DATA:-/storage/scratch1/2/ashelestov3/preprocessed_paper_data_gauss.npz}"
output_dir="executed-notebooks/${SLURM_JOB_ID}"
mkdir -p "$output_dir"
if [[ "${SPIDER_PREPROCESS:-0}" == 1 ]]; then
    python -m nbconvert --to notebook --execute prepare-numpy-files-chebyshev-gauss.ipynb \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=spider-gauss \
        --output-dir="$output_dir"
fi
python -m nbconvert --to notebook --execute COnvection-FILTER-Chebyshev-Gauss.ipynb \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=spider-gauss \
    --output-dir="$output_dir"
