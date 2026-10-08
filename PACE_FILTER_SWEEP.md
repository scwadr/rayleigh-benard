# PACE filter-scale verification

Submit from this repository on PACE, with the Python 3.12+ environment used for
the SPIDER notebook already installed. The Slurm defaults follow the existing
repository's account/QOS/AMD configuration; override them if your allocation
differs. `SPIDER_ENV` accepts a conda environment name or path.

```bash
sbatch slurm/run_filter_sweep.sbatch
```

The array runs delta/dx = **10, 25, 50, 75**, one task at a time. Each task requests
one node, 32 CPUs, all node memory (`--mem=0`), and eight hours. These are starting
resource requests, not measured requirements. SPIDER uses the allocated CPUs;
BLAS/OpenMP threads are limited to one per worker. The shell returns immediately
after submission. Watch `squeue -u "$USER"` and the `rb_filter_check-<array>_<task>.out`
files; cancel with `scancel <array-job-id>`.

For a shorter first PACE execution, keep the full momentum sparsity range but
reduce sampled integration domains and correction sparsities:

```bash
sbatch --array=0-0 slurm/run_filter_sweep.sbatch \
  --domains 64 --time-stop 80 --maximum-correction-terms 1 --skip-flux --plots
```

This is an execution check with weaker sampling, not a thesis-quality inference.
The selected data window must have at least `--domain-time` snapshots. Spatial
integration domains must fit after trimming. No accuracy threshold is invented:
successful execution and accurate equation recovery are reported separately.

Customize the sweep and environment:

```bash
export FILTER_SCALES="10 20 40 60"
export SPIDER_ENV=/path/to/your/SPIDER/environment
export DATA_PATH=/path/to/preprocessed_paper_data.npz
export OUTPUT_ROOT=/path/to/scratch/filter_verification
sbatch --array=0-3%1 --cpus-per-task=16 slurm/run_filter_sweep.sbatch --plots
```

Change the array bounds when changing the number of filter widths. Command-line
options after the sbatch file are passed to the Python runner. For one scale
inside an existing compute allocation:

```bash
python -u run_filter_sweep.py --delta-dx 25 --workers 8 --plots
```

`--skip-second-order` runs momentum/heat recovery plus first-order and general
shared-library tau/phi discovery, and skips second-order/flux cells.
`--skip-flux` retains second-order discovery but avoids its expensive pointwise
flux evaluation. Full discovery and flux diagnostics are enabled by default;
plots are optional and always run last with a headless backend.

## What is checked

The notebook uses the saved uniform spatial and temporal coordinates, computes
`dt` from `np.diff(t)`, undoes the preprocessing velocity-component swap, and
applies a spatial Gaussian with `sigma = delta/sqrt(12)`. Filtering wraps in x;
the entire filter support is discarded at the vertical walls. Pressure remains
the ordinary filtered pressure and tau is the full SGS tensor.

Before regression it evaluates the known weak-form balances:

```text
dt(ubar) + ubar.grad(ubar) + grad(pbar) - nu*lap(ubar)
        - bbar*ez + div(tau) = 0
dt(bbar) + ubar.grad(bbar) - kappa*lap(bbar) + div(phi) = 0
```

Errors with and without SGS divergence show whether the measured terms improve
the balance. Regression anchors `dt u` and `dt b` structurally, normalizes their
coefficients to +1, removes numerical zero/duplicate predictor columns, and
sweeps retained term counts. Vorticity is excluded from the recovery library.
Known coefficients are used only for the direct check; the regression may
choose other terms from the shared library, including measured tau/phi terms.

First-order tau/phi fits share one implementation. Tau has trace and STF fits
and a combined coefficient; both closures are compared with delta²/12. General
SGS closures and second-order residual discoveries use inhomogeneous regression
with measured SGS fields as targets and resolved fields as predictors. The
second-order baseline is frozen, and every candidate remains in the archive.

## Output

Each task gets a unique directory beneath OUTPUT_ROOT. Existing runs are never
overwritten. Key files:

| File | Contents |
| --- | --- |
| `execution.json` | Requested configuration, executed cell IDs, timings, and failure traceback |
| `executed_notebook.ipynb` | Exact notebook snapshot used by the batch launcher |
| `provenance.json`, `input_source_*`, `input_coordinates.npz` | Data identity, source revision/snapshots including SPIDER, inferred Dedalus configuration, settings and coordinates |
| `recovery.json` | Known momentum/heat residuals, all recovered equations and physical coefficients, removed columns, sampled integration domains |
| `first_order.json` | Fitted/theoretical first-order comparisons and integration settings |
| `shared_closures.json` | General tau/phi closure candidates and removed columns |
| `second_order_results.json`, `second_order_archives/` | All second-order candidates, thesis archives, and optional flux archives |
| `*.png` | Optional plots generated after numerical reporting |
| `COMPLETE` or `FAILED` | Execution outcome for the requested stages |

An abrupt scheduler termination can leave status `running` without either
marker; consult the Slurm log. COMPLETE means the requested stages executed,
not that a model met a scientific accuracy criterion. Reports use training
data, distinguish inferred simulation configuration from confirmed provenance,
and do not designate a thesis-selected model. Inspect errors and their change
with filter scale before treating a run as scientifically successful.

All scientific code and archive construction stay in
`COnvection-FILTER-Copy2.ipynb`. The Python launcher executes `pace:*` tagged cells
directly, so notebook fixes automatically apply to the batch workflow. Save the
notebook before submission or interactive archiving. Local scientific execution
is intentionally excluded by the repository instructions; run verification on
PACE.
