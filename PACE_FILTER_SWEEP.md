# Restored notebook and PACE filter sweep

The runner executes tagged cells from `COnvection-FILTER-Copy2.ipynb`. Scientific
implementation and archiving remain in the notebook. See the
[restoration record](NOTEBOOK_EQUIVALENCE.md) for the source comparison.

## Complete notebook run

On PACE, restart the kernel and run all cells from the top. The default filter
width is **Delta/dx=50**. The order is:

1. Imports, controls, loading, filtering, measured tau/phi, and library integration.
2. Broad homogeneous AIC sweep and jump search.
3. General tau and phi discovery from the original shared library.
4. Consolidated original inhomogeneous regression experiments.
5. Fitted/theoretical gradient comparisons and residual discovery.
6. Discovery archive, flux evaluation, and flux archive.
7. Original diagnostic plots and new flux plots.

The last cell compares archived scales only when `SWEEP_COMPARE_CONFIG` is set.
Otherwise it does nothing. Do not rerun the crop cell alone: like the original,
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
is Delta/sqrt(12). Each task requests 32 CPUs, all node memory, and eight hours.
These requests have not been benchmarked for the restored broad searches; adjust
wall time as needed. Existing account/QOS/node constraints remain in the Slurm file.

By default, all run archives and plots go into unique per-scale directories under
`/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/`, outside the repository.
The runner creates this directory as needed. This is also the default for direct
Python runs. To group a particular sweep separately, set this before submission:

```bash
export OUTPUT_ROOT=/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/restored_sweep_01
```

`sbatch` returns immediately. Logs are
`/storage/scratch1/2/ashelestov3/rb_filter_check-<job>_<task>.out`;
monitor with `squeue -u "$USER"`. `SPIDER_ENV` selects the conda environment.
`DATA_PATH` and `OUTPUT_ROOT` override input/output locations. Use a dedicated
output directory for each sweep. An already-set `OUTPUT_ROOT` still overrides the
scratch default; unset it to use the default. Slurm logs use the existing scratch
root because Slurm opens the log before the script can create subdirectories.

To change scales, change both the list and array range:

```bash
export FILTER_SCALES="10 20 40"
sbatch --array=0-2%1 slurm/run_filter_sweep.sbatch --plots
```

Running Python directly processes **one scale**, in the foreground of an allocated
PACE compute session:

```bash
python -u run_filter_sweep.py --delta-dx 50 --workers 32 --plots
```

## What each scale runs

- Original homogeneous AIC `max_k=5..39` and jump `max_k=10` searches. The broad
  library includes continuity, momentum with tau, and heat transport with phi.
  Recovery is judged from equations and residuals, not assumed.
- General tau trace/STF and phi discoveries, requesting up to six terms including
  the forced target.
- Six original experiments: scalar column 31 (combinatorial), vector column 85
  (power), vector column 22 (combinatorial), scalar column 56 (combinatorial),
  scalar column 45 (power and combinatorial). Printed term names identify these
  original numerical targets. Column 22's invalid six-term request with max_k=5
  is archived as skipped; valid requests run with their original settings.
- Prescribed gradient calibration, theoretical Delta^2/12 comparison, and
  original residual searches for both baselines.
- Pi_tau = -S_ij tau_ij and Pi_phi = -phi_i d_i bbar, with S the symmetric resolved
  velocity gradient. Here b is the simulation's temperature-like scalar.
  First-order gradient and second-order gradient-plus-Hessian prescribed models
  are compared with fitted/theoretical baselines and discovered residual corrections.
  All use common spatial support within each run. Every trace/STF pairing is archived.
- Paper accuracy A for closures and local fluxes, mean-flux ratios, and relative
  errors. No best model is selected. General shared-library candidates are still
  discovered and archived; their flux evaluation is optional (`--flux-include-general`).

`--plots` saves final plotting cells as PNGs. Without it, numerical reports and
integrated flux moments are still archived. Original plot indices and zoom selections are
preserved and require a sufficiently large input window/grid.

## Batched Figure 2 diagnostics

Paper equations **(3.8)-(3.9)** require spatial/component contractions integrated
over time, followed by one magnitude-aware accuracy ratio. They do not require
per-frame accuracy plots, full flux snapshots, Pearson correlations, or sign sweeps.
The flux cell now accumulates only the moments needed for those accuracies, mean
flux ratios, and relative errors. All times are retained with trapezoidal weights;
there is no silent time subsampling or averaging of individual-frame accuracies.

Spatial derivatives are shared between candidates within batches of eight frames,
with a bounded 256 MiB derivative cache. Trace/STF pair metrics use orthogonality
and scalar flux cross moments, avoiding full tensor reconstruction for every pair.
`--flux-batch-size N` and `--flux-cache-mib N` tune temporary memory/performance;
they do not change the averaging window. Total working memory also includes input
fields and candidate batches, beyond the derivative cache. Runtime must still be
measured on PACE; this does not guarantee the complete discovery job fits eight hours.

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

After the array completes:

```bash
python run_filter_sweep.py --compare-runs "${OUTPUT_ROOT:-/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps}"
```

This reads completed schema-4 flux archives without rerunning any diagnostics.
It writes separate `accuracy_vs_filter_scale_tau.png` and
`accuracy_vs_filter_scale_phi.png` figures (also PDF), plus CSV and JSON, in a
unique comparison directory. Each figure has the three Figure 2 rows: closure
accuracy, local-flux accuracy, and mean modeled/reference flux versus filter width.
Prescribed first/second-order models, uncorrected baselines, and discovered
corrections with matching requested trace/STF budgets are included by default.
Add `--include-corrections` to also plot mixed-budget pairs; all pair metrics are
archived regardless. This is a plotting convention, not error-based model selection.
`--reference-length L` uses Delta/L; otherwise the axis is explicitly Delta/dx,
not the paper's Delta/integral-scale normalization.

Use one run per scale with the same inputs/settings. Earlier workflow archives
are skipped; duplicate scales or mixed settings are rejected. Candidates are
independently fitted at each width, so their term support may change.

## Explicitly reduced runs

Overrides include `--homogeneous-max-k`, `--closure-max-terms`,
`--maximum-correction-terms`, `--domains`, `--domain-width`, `--domain-time`,
`--time-start`, `--time-stop`, and `--alpha`. These change the run and are archived.

- `--skip-inhomogeneous-experiments` omits numerical-target experiments.
- `--skip-baseline-residuals` (alias `--skip-second-order`) omits gradient calibration
  and residual searches; general discovery still runs, and prescribed first/second-order
  flux benchmarks remain. Add `--flux-include-general` for general-candidate fluxes.
- `--skip-flux` omits flux evaluation.

Combinatorial searches and the broad homogeneous sweep can be expensive. A reduced
execution check is possible, but is not equivalent to the complete run:

```bash
sbatch --array=0-0 slurm/run_filter_sweep.sbatch --homogeneous-max-k 5 --domains 64 --time-stop 80 --skip-inhomogeneous-experiments --maximum-correction-terms 1 --skip-flux
```

## Archives and status

Each execution has a unique directory with `execution.json`, the executed notebook
snapshot, cell timings, and `COMPLETE` or `FAILED`. Exceptions include a traceback.
Caught errors from the original homogeneous sweep prevent successful completion.

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
