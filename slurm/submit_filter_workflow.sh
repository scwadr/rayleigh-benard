#!/bin/bash
# Submit from the checkout; all generated files, including Slurm logs, go to scratch.
set -euo pipefail
mode="${1:?Usage: bash slurm/submit_filter_workflow.sh measure|discover|prescribed|plot [runner options]}"
shift
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
base=/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps
case "$mode" in
    measure) script=measure_li.sbatch; options=() ;;
    discover) script=run_filter_sweep.sbatch; options=(--discovery-benchmarks) ;;
    prescribed) script=run_filter_sweep.sbatch; options=(--prescribed-only) ;;
    plot) script=plot_filter_sweep.sbatch; options=() ;;
    *) echo "Unknown mode: $mode" >&2; exit 1 ;;
esac
mkdir -p "$base"
submission="$(mktemp -d "$base/${mode}_$(date -u +%Y%m%dT%H%M%S)_XXXXXXXX")"
mkdir -p "$submission/logs" "$submission/bootstrap"
export OUTPUT_ROOT="$submission"
export WORKFLOW_RUNTIME_ROOT="$submission/bootstrap"
batch_options=(--chdir="$submission/bootstrap" --output="$submission/logs/%x-%A_%a.out")
if [[ "$mode" == discover || "$mode" == prescribed ]]; then
    batch_options+=(--array="${SWEEP_ARRAY:-0-5%1}")
fi
echo "Scratch submission directory: $submission"
sbatch "${batch_options[@]}" "$REPO_ROOT/slurm/$script" "${options[@]}" "$@" \
    | tee "$submission/submission.txt"
