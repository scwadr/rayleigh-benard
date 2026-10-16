# Restored notebook and PACE filter sweep

For exact commands to launch both modes in separate scratch directories,
failure behavior, and the output/plot directory map, see
[the two-run PACE run sheet](PACE_TWO_RUNS.md).

For the current measure-ell_i-first, six-width discovery plus prescribed-model
comparison, archive validation, reuse of completed fits, and paper limitations, see
[Figure 2 audit and launch commands](FIGURE2_BENCHMARK_AUDIT.md).

The runner executes tagged cells from `COnvection-FILTER-Copy2.ipynb`. Scientific
implementation and archiving remain in the notebook. See the
[restoration record](NOTEBOOK_EQUIVALENCE.md) for the source comparison.

Batch execution runs tau/phi model discovery, fitted/theoretical coefficient
comparisons, residual discovery, and flux analysis. It excludes broad homogeneous
PDE recovery and the six legacy numerical-target experiments. Those cells remain
available for interactive notebook runs; the complete notebook order below is
not the batch execution order. The new `--discovery-benchmarks` mode retains
general discovery and skips the optional frozen-baseline residual searches.

## Complete notebook run

On PACE, run preparation/discovery/archive cells from the top, then the single
replacement flux cell (`dc5ac1e7`). Skip the two legacy flux cells;
they remain only for historical reference. The default filter
width is **Delta/dx=50**. The order is:

1. Imports, controls, loading, filtering, measured tau/phi, and library integration.
2. Broad homogeneous AIC sweep and jump search.
3. General tau and phi discovery from the original shared library.
4. Consolidated original inhomogeneous regression experiments.
5. Fitted/theoretical gradient comparisons and residual discovery.
6. Discovery archive, then the replacement cell combining tiled flux evaluation,
   flux archiving and optional single-width plots.
7. Optional original diagnostic plots for interactive notebook use only.

The `filter-sweep-comparison` cell compares archived scales only when
`SWEEP_COMPARE_CONFIG` is set; `measure-integral-scale` measures ell_i only when
`LI_CONFIG` is set. Otherwise they do nothing. Do not rerun the crop cell alone: like the original,
it crops arrays in place. Start from loading for a new filter width.

For optional overrides, define `RUN_CONFIG` before the controls cell, for example
`RUN_CONFIG = {'delta_dx': 25, 'workers': 32}`. Without overrides, original
scientific defaults apply. Gradient comparisons run by default.

## Background sweep

Use the existing Python 3.12+ SPIDER environment on PACE. From the repository:

```bash
sbatch slurm/run_filter_sweep.sbatch --plots
```

The array checks **Delta/dx = 10, 25, 50, 75**, one task at a time. Gaussian sigma
is Delta/sqrt(12). Each task requests 64 AMD CPUs on one node, all node memory,
and eight hours. The AMD constraint, account and QOS match `slurm/paper-test`.
One Python task uses the allocation for discovery workers and flux threads.
These requests have not been benchmarked for the SGS discovery workflow; adjust
wall time as needed. Existing account/QOS/node constraints remain in the Slurm file.

Input defaults to `/storage/scratch1/2/ashelestov3/preprocessed_paper_data.npz`.
Every array gets a unique directory:

```text
/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/sweep_<array-job-id>/
```

Each width has its own timestamped run/archive directory inside it. Temporary
files, Python bytecode, JAX caches, Matplotlib caches, archives and figures all go
to scratch. Source code is read from the checkout. `DATA_PATH`/`--data-path` and
`OUTPUT_ROOT`/`--output-root` may override paths **only within this scratch root**;
old home/repository overrides fail explicitly. `OUTPUT_ROOT` is a parent directory:
the launcher always adds `sweep_<array-job-id>`.

One task automatically submits a small plotting job with an `afterok` dependency
on the entire array. It reads the completed moment archives after every width
succeeds and writes `figure2_delta_<timestamp>_<id>/` inside the sweep directory.
`comparison_job.txt` records its Slurm ID. This happens with or without `--plots`;
`--plots` additionally saves each width's quick flux plots. `COMPARE_SWEEP=0` or
`--skip-flux` disables automatic comparison. Partial arrays do not auto-compare.

`sbatch` returns immediately. Array logs are
`/storage/scratch1/2/ashelestov3/rb_filter_check-<job>_<task>.out`;
comparison logs are `sweep_<array-job-id>/logs/figure2-<plot-job-id>.out`.
Monitor with `squeue -u "$USER"`. `SPIDER_ENV` selects the conda environment.
Slurm opens array logs before the script can create subdirectories, so those logs
use the existing scratch root. Cancel an array with `scancel <array-job-id>`;
a dependent comparison cannot run if any required task fails or is canceled.

To change scales, change both the list and array range:

```bash
export FILTER_SCALES="10 20 40"
sbatch --array=0-2%1 slurm/run_filter_sweep.sbatch --plots
```

Running Python directly processes **one scale**, in the foreground of an allocated
PACE compute session:

```bash
python -u run_filter_sweep.py --delta-dx 50 --workers 64 --flux-workers 64 --plots
```

## What each scale runs

- Preparation and shared-library integration. The shared library remains unchanged
  because tau/phi discovery uses it; broad homogeneous AIC/jump searches are skipped.
- General tau trace/STF and phi discoveries, requesting up to six terms including
  the forced target.
- The six legacy numerical-target experiments are skipped.
- Gradient coefficient calibration, theoretical Delta^2/12 comparison, and
  original residual searches for both baselines.
- Pi_tau = -S_ij tau_ij and Pi_phi = -phi_i d_i bbar, with S the symmetric resolved
  velocity gradient. Here b is the simulation's temperature-like scalar.
  First-order gradient and second-order gradient-plus-Hessian prescribed models
  are compared with fitted/theoretical baselines and discovered residual corrections.
  All use common spatial support within each run. Every trace/STF pairing is archived.
- Paper accuracy A for closures and local fluxes, mean-flux ratios, and relative
  errors. No best model is selected. General shared-library candidates are still
  discovered and archived; their flux evaluation is optional (`--flux-include-general`).

`--plots` saves the replacement cell's per-width flux figures. Legacy plotting
cells are excluded from batch execution. Numerical reports and integrated moments
are archived whether or not `--plots` is supplied.

## Batched Figure 2 diagnostics

Paper equations **(3.8)-(3.9)** require spatial/component contractions integrated
over time, followed by one magnitude-aware accuracy ratio. They do not require
per-frame accuracy plots, full flux snapshots, Pearson correlations, or sign sweeps.
The flux cell now accumulates only the moments needed for those accuracies, mean
flux ratios, and relative errors. All times are retained with trapezoidal weights;
there is no silent time subsampling or averaging of individual-frame accuracies.

The grid is divided into 64-by-64 spatial tiles with eight-point halos, retaining
every interior point. The batch launcher uses all 64 allocated CPUs for tiles,
with threads sharing the existing input arrays, in batches of eight times. Derivatives
and contracted terms are cached per tile and reused across candidates/families.
This avoids whole-grid cache thrashing: a single 4096-by-1536-by-8 float64 derivative
is 384 MiB before cropping, larger than the old 256 MiB cache. A default tile plus
halo needs about 0.39 MiB per derivative. Sparse coefficient plans are built once,
so inactive library columns do not add Python loops for every tile.

Trace/STF pair metrics still use orthogonality and scalar flux cross moments.
Tile moments are weighted by their retained point counts, including partial edge
tiles, to recover the same full-grid spatial average. All times, models and
coefficients are retained; only floating-point summation order changes.
The centered derivative stencils have the same spacing and accuracy as before,
and tile halos keep artificial tile boundaries outside the retained interior.

`--flux-tile-size N`, `--flux-workers N`, `--flux-batch-size N`, and
`--flux-cache-mib N` tune execution without changing the averaging window or grid.
The cache budget is per active tile/worker and includes derivative and term arrays;
total memory also includes input fields and candidate workspaces. Progress reports
tile counts, batch timings, cache hits and evictions, also saved in `flux.json`.
Runtime still needs measurement on PACE; no eight-hour completion guarantee is made.

### Continue in an existing notebook kernel

Interrupt the old flux cell without restarting the kernel. With discovery and
its archive already in memory, update execution controls directly:

```python
run_controls = dict(run_controls)
run_controls.update(flux_tile_size=64, flux_batch_size=8, flux_cache_mib=256)
```

Run only the replacement cell (`dc5ac1e7`, cell 33 counting Markdown), which
calculates and archives fluxes and makes optional per-width plots. Skip the two
old flux cells. Do not rerun setup, cropping, filtering, or regression. The live
kernel keeps its existing `analysis_run_dir`; the new scratch default applies to
new runs. Explicit execution controls are preserved rather than overwritten.
New archives preserve earlier results and capture the executed flux source.

Both model sets have explicit labels:

- **Prescribed first order:** gradient contraction with coefficient Delta^2/12.
- **Prescribed second order:** first order plus Hessian contraction with
  coefficient Delta^4/288, including both mixed-derivative contributions.
- **Fitted/theoretical first order:** the notebook's frozen gradient baseline.
- **Discovered residual-corrected:** that baseline plus each returned correction.
  These corrections are not assumed to equal the prescribed Hessian term.

For tau, the prescribed formulas are paper equations (3.1)-(3.4). For phi, use
the analogous Gaussian scalar-flux expansion, contracting velocity derivatives
with scalar derivatives. The paper studies stress; it does not establish this
scalar extension as an empirical result. Derivative conventions are archived:
first gradients retain the fitted baseline's `numpy.gradient` convention, and
Hessians/library derivatives use sixth-order finite differences on the common interior.

## Plot multiple filter widths

The array submits this automatically. To replot existing results without any
filtering, fitting, or flux computation, submit the archive-only job explicitly:

```bash
sbatch slurm/plot_filter_sweep.sbatch --compare-runs /storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/sweep_<array-job-id>
```

Use the actual numeric array ID in that path. At least two completed widths are
required. The comparison verifies the archived moments, time weights and JSON
metrics, then produces color-coded paper-style line plots with three rows:
closure accuracy, flux accuracy, and mean modeled/reference flux. The default
x-axis is physical **Delta**, not correction count or Delta/dx. Supply
`--reference-length L` to the plotting job for Delta/L, or set `REFERENCE_LENGTH`
before submitting an array. No integral length is inferred.

Separate tau and phi figures are saved for each available matched correction
budget: `figure2_tau_N1_vs_delta.png`, `figure2_phi_N1_vs_delta.png`, etc., also
PDF and SVG. Each figure holds N fixed and compares the theoretical baseline plus
N corrections, prescribed gradient plus Hessian, fitted gradient, fitted baseline
plus N corrections, and prescribed gradient. Tau pairs use N corrections for each
trace/STF fit. Colors, line styles and three-panel layout follow the next-cell
analysis; markers show evaluated widths. Missing/undefined values leave gaps;
axis limits expand to retain values outside the paper's displayed range.

`--correction-budgets 1 3 5` restricts the comparison figures. Every candidate,
including mixed trace/STF budgets, remains in `metrics.csv` and source snapshots.
`analysis.json`, `analysis.txt`, the executed analysis source and hashed copies of
the small source archives accompany each unique plot directory. This selects no
best candidate. `--include-corrections` is retained as a compatibility flag;
matched corrections are now included by default. Prescribed-only runs also plot.

Use one run per width with the same inputs/settings. Earlier schemas and incomplete
runs are skipped; duplicate widths or mixed settings are rejected. Automatic
comparisons additionally require every requested width. Candidates are fitted
independently at each width; term support and original fit cropping may change.

## Explicitly reduced runs

Overrides include `--closure-max-terms`,
`--maximum-correction-terms`, `--domains`, `--domain-width`, `--domain-time`,
`--time-start`, `--time-stop`, and `--alpha`. These change the run and are archived.

- `--skip-inhomogeneous-experiments` remains a compatibility no-op: those
  experiments are always excluded. `--homogeneous-max-k` is no longer accepted.
- `--skip-baseline-residuals` (alias `--skip-second-order`) omits gradient calibration
  and residual searches; general discovery still runs, and prescribed first/second-order
  flux benchmarks remain. Add `--flux-include-general` for general-candidate fluxes.
- `--skip-flux` omits flux evaluation.

SGS discovery and library integration can still be expensive. A reduced
execution check is possible, but is not equivalent to the complete run:

```bash
sbatch --array=0-0 slurm/run_filter_sweep.sbatch --domains 64 --time-stop 80 --maximum-correction-terms 1 --skip-flux
```

## Archives and status

Each execution has a unique directory with `execution.json`, the executed notebook
snapshot, cell timings, and `COMPLETE` or `FAILED`. Exceptions include a traceback.
The execution manifest lists excluded PDE/legacy cells, and run controls explicitly
record both stages as disabled. Empty homogeneous/legacy result lists in the
discovery archive therefore mean not run, rather than successful equation recovery.

Notebook archives contain returned candidate equations, coefficients and errors,
baseline choices, settings, integrated libraries, coordinates, input identity,
Git revision/status, source snapshots, and inferred Dedalus source parameters.
Historical simulation provenance remains explicitly unconfirmed. Training fits
and random weak-row splits are not labeled independent validation or thesis
selections. Flux archives include definitions, explicit prescribed coefficients,
candidate IDs, JSON/CSV, optional plots, and `flux_moments.npz` with coordinates,
all time weights, and sufficient moments to reproduce every reported metric.
Per-frame flux arrays/time series are no longer written. Previous archives are preserved;
old schema-3 flux archives are not mixed with the revised comparisons.

Only static parsing and source inspection were performed locally. Numerical
execution and resource requirements must be checked on PACE.
