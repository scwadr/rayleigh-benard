#!/bin/bash
# One fresh process per width. A dependent job makes the cross-width figures.
#SBATCH --job-name=rb_filter_check
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --array=0-3%1
#SBATCH --output=/storage/scratch1/2/ashelestov3/%x-%A_%a.out

set -euo pipefail
SCRATCH_ROOT=/storage/scratch1/2/ashelestov3
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
while [[ "$REPO_ROOT" != / && ! -f "$REPO_ROOT/run_filter_sweep.py" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/run_filter_sweep.py" ]] || { echo 'Cannot find run_filter_sweep.py' >&2; exit 1; }
REPO_ROOT="$(realpath "$REPO_ROOT")"

# Preserve explicit path overrides, but require all data and outputs on scratch.
data_path="${DATA_PATH:-$SCRATCH_ROOT/preprocessed_paper_data.npz}"
output_parent="${OUTPUT_ROOT:-$SCRATCH_ROOT/rayleigh-benard/filter_sweeps}"
auto_compare="${COMPARE_SWEEP:-1}"
scale_plan=""
runner_args=()
while (( $# )); do
    case "$1" in
        --data-path) data_path="${2:?--data-path needs a path}"; shift 2 ;;
        --data-path=*) data_path="${1#*=}"; shift ;;
        --output-root) output_parent="${2:?--output-root needs a path}"; shift 2 ;;
        --output-root=*) output_parent="${1#*=}"; shift ;;
        --scale-plan) scale_plan="${2:?--scale-plan needs a path}"; shift 2 ;;
        --scale-plan=*) scale_plan="${1#*=}"; shift ;;
        --skip-flux) auto_compare=0; runner_args+=("$1"); shift ;;
        --delta-dx|--delta-dx=*|--scale-index|--scale-index=*|--measure-li|--compare-runs|--compare-runs=*)
            echo 'Array widths come from --scale-plan or FILTER_SCALES; use the plotting script for --compare-runs.' >&2; exit 1 ;;
        *) runner_args+=("$1"); shift ;;
    esac
done
data_path="$(realpath -m "$data_path")"
output_parent="$(realpath -m "$output_parent")"
for path in "$data_path" "$output_parent"; do
    case "$path" in
        "$SCRATCH_ROOT"/*) ;;
        *) echo "Expected a scratch path, got $path. Clear stale DATA_PATH/OUTPUT_ROOT overrides." >&2; exit 1 ;;
    esac
done
[[ -n "$scale_plan" || -f "$data_path" ]] || { echo "Missing input: $data_path" >&2; exit 1; }

read -r -a scales <<< "${FILTER_SCALES:-10 25 50 75}"
if [[ -n "$scale_plan" ]]; then
    scale_plan="$(realpath "$scale_plan")"
    # Scientific widths are resolved/validated from the plan by the Python runner.
    scales=(0 1 2 3 4)
fi
task="${SLURM_ARRAY_TASK_ID:-0}"
(( task < ${#scales[@]} )) || { echo "No filter scale for array index $task" >&2; exit 1; }
sweep_id="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
sweep_root="$output_parent/sweep_$sweep_id"
runtime="$sweep_root/runtime/task_$task"
mkdir -p "$sweep_root/logs" "$runtime/tmp" "$runtime/matplotlib" "$runtime/cache" "$runtime/pycache"
export REPO_ROOT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TMPDIR="$runtime/tmp" MPLCONFIGDIR="$runtime/matplotlib" XDG_CACHE_HOME="$runtime/cache"
export PYTHONPYCACHEPREFIX="$runtime/pycache" JAX_COMPILATION_CACHE_DIR="$runtime/cache/jax"
module load anaconda3
source activate "${SPIDER_ENV:-SPIDER}"
cd "$REPO_ROOT"

echo "Sweep directory: $sweep_root"
scale_args=(--delta-dx "${scales[$task]}")
if [[ -n "$scale_plan" ]]; then
    scale_args=(--scale-plan "$scale_plan" --scale-index "$task")
    echo "Job $SLURM_JOB_ID task $task: scale plan $scale_plan, workers=$SLURM_CPUS_PER_TASK"
else
    echo "Job $SLURM_JOB_ID task $task: Delta/dx=${scales[$task]}, workers=$SLURM_CPUS_PER_TASK"
fi
srun --ntasks=1 python -u run_filter_sweep.py \
    --workers "$SLURM_CPUS_PER_TASK" --flux-workers "$SLURM_CPUS_PER_TASK" \
    "${runner_args[@]}" \
    "${scale_args[@]}" --data-path "$data_path" --output-root "$sweep_root"

# Exactly one task submits the comparison. afterok on the array job ID waits for
# ALL array tasks, regardless of their execution order or concurrency.
if [[ "$auto_compare" == 1 && "$task" == "${SLURM_ARRAY_TASK_MIN:-0}" ]]; then
    if (( ${SLURM_ARRAY_TASK_COUNT:-1} != ${#scales[@]} )); then
        echo 'Partial array: not submitting a full-sweep comparison. Compare the completed subset explicitly.'
    else
        comparison_args=(--compare-runs "$sweep_root" --expected-scales "${scales[@]}")
        if [[ -n "$scale_plan" ]]; then
            comparison_args=(--compare-runs "$sweep_root" --scale-plan "$scale_plan")
        fi
        for arg in "${runner_args[@]}"; do
            if [[ "$arg" == --prescribed-only || "$arg" == --discovery-benchmarks ]]; then
                comparison_args+=("$arg")
            fi
        done
        if [[ -n "${REFERENCE_LENGTH:-}" && -z "$scale_plan" ]]; then
            comparison_args+=(--reference-length "$REFERENCE_LENGTH")
        fi
        plot_job="$(sbatch --parsable --dependency="afterok:$sweep_id" --kill-on-invalid-dep=yes \
            --output="$sweep_root/logs/figure2-%j.out" \
            "$REPO_ROOT/slurm/plot_filter_sweep.sbatch" "${comparison_args[@]}")"
        printf '%s\n' "$plot_job" > "$sweep_root/comparison_job.txt"
        echo "Submitted Figure 2 comparison job $plot_job (after the full array succeeds)."
    fi
fi
