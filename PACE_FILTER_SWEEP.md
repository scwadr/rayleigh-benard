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
export OUTPUT_ROOT="$PWD/thesis_runs/restored_sweep_01"
sbatch slurm/run_filter_sweep.sbatch --plots
```

The array checks **Delta/dx = 10, 25, 50, 75**, one task at a time. Gaussian sigma
is Delta/sqrt(12). Each task requests 32 CPUs, all node memory, and eight hours.
These requests have not been benchmarked for the restored broad searches; adjust
wall time as needed. Existing account/QOS/node constraints remain in the Slurm file.

`sbatch` returns immediately. Logs are `rb_filter_check-<job>_<task>.out`;
monitor with `squeue -u "$USER"`. `SPIDER_ENV` selects the conda environment.
`DATA_PATH` and `OUTPUT_ROOT` override input/output locations. Use a dedicated
output directory for each sweep.

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
  General, fitted-gradient, and theoretical-gradient models use common spatial
  support within each run, including corrections and all trace/STF pairings.
- Paper accuracy A for closures and fluxes, mean-flux ratios, relative errors,
  correlations, and transfer-sign statistics. No best model is selected.

`--plots` saves final plotting cells as PNGs. Without it, numerical reports and
flux arrays are still archived. Original plot indices and zoom selections are
preserved and require a sufficiently large input window/grid.

## Plot multiple filter widths

After the array completes:

```bash
python run_filter_sweep.py --compare-runs "$OUTPUT_ROOT"
```

This reads completed restored-workflow archives and writes a unique comparison
directory with `accuracy_vs_filter_scale.png`, CSV, and JSON. Curves include all
general discoveries and uncorrected fitted/theoretical gradients. Add
`--include-corrections` for residual corrections. `--reference-length L` uses
Delta/L on the horizontal axis; otherwise it uses Delta/dx.

Use one run per scale with the same inputs/settings. Earlier workflow archives
are skipped; duplicate scales or mixed settings are rejected. Candidates are
independently fitted at each width, so their term support may change.

## Explicitly reduced runs

Overrides include `--homogeneous-max-k`, `--closure-max-terms`,
`--maximum-correction-terms`, `--domains`, `--domain-width`, `--domain-time`,
`--time-start`, `--time-stop`, and `--alpha`. These change the run and are archived.

- `--skip-inhomogeneous-experiments` omits numerical-target experiments.
- `--skip-baseline-residuals` (alias `--skip-second-order`) omits gradient calibration
  and residual searches; general discovery and its flux analysis still run.
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
selections. Flux archives include definitions, candidate IDs, arrays, JSON/CSV,
and optional plots. Previous archives are preserved.

Only static parsing and source inspection were performed locally. Numerical
execution and resource requirements must be checked on PACE.
