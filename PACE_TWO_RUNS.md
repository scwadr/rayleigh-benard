# PACE: separate discovery and prescribed-only sweeps

Both pipelines now use the external package in `requirements-pyspider.txt`.
Install it in the PACE environment before these commands (see `README.md`).
The sweep snapshots the selected package on scratch and records its revision
and source hashes. This consolidation preserves Copy2's scientific cells.

## What will run

Both submissions use the same completed integral-scale measurement and scale
plan. New plans normally contain Delta/li = **0.05, 0.2, 0.4, 0.6, 0.8, 1.0**.
The five original widths may be reduced uniformly to fit available filter
support; 0.05 stays exact. Read the measurement log for the actual plan.

| Submission mode | Work at each width | Comparison curves |
| --- | --- | --- |
| `discover` | Filter data, discover general tau/phi closures (or reuse compatible saved fits), evaluate discovered and prescribed fluxes, archive all candidates | Discovered models versus prescribed gradient and gradient-plus-Hessian |
| `prescribed` | Filter data, evaluate prescribed fluxes, archive results; no library integration or model discovery | Prescribed gradient versus prescribed gradient-plus-Hessian |

Discovery mode is the same general-discovery mode introduced before the 0.05
and scratch-isolation changes. It still skips the optional frozen-baseline
residual searches that failed in the original sweep. It is not the original
unrestricted notebook execution. These are Figure-2-style comparisons on the
convection data, not reproductions of the paper's F1/F2 datasets.

Each width is a separate Slurm array task and Python process. A failed width
does not cancel sibling tasks or the other submission. The default `%1` runs
at most one width per array at a time; the two arrays may run concurrently.
Each width requests 64 CPUs, the node's available memory, and eight hours.
Shared input/site failures can independently affect multiple tasks.

Automatic plotting uses `afterok` on the whole array and therefore requires
every task to succeed. If the minimum-index task fails, it never submits the
plot job; if another task fails, the submitted plot dependency cannot succeed.
In either case, successful archives remain intact. Do not interpret a missing
plot as missing successful data. This follows the documented
[Slurm array dependency behavior](https://slurm.schedmd.com/job_array.html).

## Launch on PACE

Update the PACE checkout with these changes, then run the commands from that
checkout. The Slurm scripts activate `${SPIDER_ENV:-SPIDER}` via `anaconda3`.
The default input is
`/storage/scratch1/2/ashelestov3/preprocessed_paper_data.npz`.

First submit measurement, unless a completed plan already includes 0.05:

```bash
bash slurm/submit_filter_workflow.sh measure
```

The helper prints a unique scratch submission directory and job ID. Wait for
measurement to succeed, then read its `logs/rb_measure_li-*.out`. The log prints
the measurement directory containing `scale_plan.json` and `COMPLETE`.
Set `PLAN` to that actual absolute path, not the placeholder below:

```bash
PLAN=/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/measure_TIMESTAMP_ID/li_TIMESTAMP_ID/scale_plan.json

# Separate submission 1: general discovery plus prescribed benchmarks.
SWEEP_ARRAY=0-5%1 bash slurm/submit_filter_workflow.sh discover \
  --scale-plan "$PLAN"

# Separate submission 2: prescribed benchmarks only.
SWEEP_ARRAY=0-5%1 bash slurm/submit_filter_workflow.sh prescribed \
  --scale-plan "$PLAN"
```

If the plan contains only five widths (an older plan, or a support-limited plan
with a duplicate removed), use `SWEEP_ARRAY=0-4%1` for both submissions. Old
plans are immutable and do not automatically gain 0.05.

Record the two printed submission directories and array job IDs. The helper
uses `mktemp -d` separately for every invocation, with `discover_` and
`prescribed_` prefixes; the two runs cannot share their generated directories
with these commands. Each also stores its job ID in `submission.txt`.

Existing archives are never overwritten. Fresh discovery runs use the pinned
PySPIDER implementation. Add `--reuse-runs /path/to/completed/new-package/runs`
only for archives created with the same package identity and compatible inputs
and settings. Legacy vendored-SPIDER fits are rejected for reuse, but remain
available for archive-only inspection/plotting. The prescribed-only submission
evaluates its own benchmarks in its own directory.

## Where everything goes

All generated workflow files are under this scratch base:

```text
/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/
  measure_TIMESTAMP_RANDOM/
    submission.txt
    logs/rb_measure_li-*.out
    bootstrap/                         initial runtime/cache files
    li_TIMESTAMP_RANDOM/
      scale_plan.json, measurement.json, li_moments.npz, COMPLETE
      executed_notebook.ipynb, source_* source snapshots
      runtime/                         temporary files, caches, working source copy

  discover_TIMESTAMP_RANDOM/           first independent submission
    submission.txt
    logs/rb_filter_check-ARRAY_TASK.out
    bootstrap/                         plot-job bootstrap/runtime files
    sweep_ARRAY/
      comparison_job.txt               present after plot submission
      logs/figure2-JOB.out
      runtime/task_TASK/               array-task bootstrap/cache files
      TIMESTAMP_delta_dx_WIDTH_RANDOM/  unique per-width run
        execution.json, executed_notebook.ipynb, COMPLETE or FAILED
        shared_closures.json, shared_closure_progress.json (when fitted)
        discovery_TIMESTAMP_RANDOM/    equations, coefficients, libraries, provenance
        flux_TIMESTAMP_RANDOM/         flux.json, flux.csv, flux_moments.npz, sources
        runtime/                       working source copy, caches, temporary files
      comparison_runtime_RANDOM/       plotting caches and working source copy
      figure2_delta_TIMESTAMP_RANDOM/   cross-width plots and audit

  prescribed_TIMESTAMP_RANDOM/         second independent submission
    ...same directory structure with its own job IDs...
```

Discovery plots are `figure2_tau_P1_vs_delta.*` through `P5` and the equivalent
`figure2_phi_P1_vs_delta.*` through `P5` with default predictor budgets. The
prescribed-only plots are `figure2_tau_N0_vs_delta.*` and
`figure2_phi_N0_vs_delta.*`. Each is saved as PNG, PDF and SVG. Every comparison
directory also contains `analysis.json`, `analysis.txt`, `metrics.csv`, source
copies, the scale measurement/plan, and a completion or failure marker.
Plots have closure accuracy, local-flux accuracy and mean-flux ratio panels.

Input data is read in place. Source snapshots and execution working directories
are on scratch; Python bytecode, temporary files, and configured scientific
library caches are redirected there. The original checkout supplies read-only
source/Git provenance; Git optional index writes are disabled. No output is
intentionally written to project/home storage. Previously downloaded logs and
plots in the local repository are not deleted or moved.

## Monitor and inspect failures

Replace `ARRAY_ID` with each submission's actual job ID:

```bash
squeue -j ARRAY_ID
sacct -j ARRAY_ID --format=JobID,State,ExitCode,Elapsed
```

For a failed width, inspect its submission log and per-width `execution.json`
or `FAILED` file. A node kill or hard timeout can prevent final markers from
being written; use Slurm state as well. Do not resubmit the whole array merely
because one width failed. Successful per-width archives remain usable.

To rerun only plotting/auditing after a complete sweep, set `SWEEP` to its
actual `sweep_ARRAY` directory and use the matching mode:

```bash
bash slurm/submit_filter_workflow.sh plot \
  --compare-runs "$SWEEP" --scale-plan "$PLAN" --discovery-benchmarks

# For the separate prescribed-only sweep instead:
bash slurm/submit_filter_workflow.sh plot \
  --compare-runs "$SWEEP" --scale-plan "$PLAN" --prescribed-only
```

Add `--verify-only` to audit without plotting. This does not relaunch discovery
or field evaluation. The full-plan audit rejects missing widths; it does not
silently draw an incomplete curve. Plot submission logs get their own fresh
`plot_TIMESTAMP_RANDOM/` directory, while plots/audits are saved under `SWEEP`.

This run sheet was checked against the submission scripts and notebook output
paths. No numerical run or submission was performed from the development host.
