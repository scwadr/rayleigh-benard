# Functionality audit

Comparison: notebook at `102c126` (before the consolidation) versus the
consolidated notebook at `6321ae1`, plus the paper-accuracy correction described
below. This is a source-level audit, not a PACE numerical equivalence test.

**The consolidation was not a functionality-neutral refactor.** The targeted
momentum/heat and SGS discovery workflows remain, but several broader searches
were removed and numerical choices changed. It should not have been described
as only removing duplicates.

| Capability | Earlier version | Updated version |
| --- | --- | --- |
| General homogeneous equation search | `interleave_identify`, AIC sweep over max_k 5–39, plus a separate jump-threshold search; train fractions .75/.8 | Removed; no equivalent mode in the current notebook or batch runner |
| Inhomogeneous momentum/heat recovery | Several manually selected numeric column indices; power and combinatorial initializers | Targets structurally identified as dt(u)/dt(b), power initialization, term counts 2–8 by default, training fraction 1 |
| Time-dependent competing predictors | Present in the broad library | Excluded from targeted recovery predictors; only the chosen time-derivative target remains |
| Shared equation/closure library | Complexity 4, at most two observables; all rank-2 irreps, including antisymmetric; w included | Complexity 4, max_dt=1, max_dx=2; scalar/vector/STF only; w omitted |
| General tau/phi closure discovery | Resolved spatial predictors, target included; counts started at 1 | Preserved in purpose, but uses the changed shared library and counts start at 2 |
| First-order tau | Duplicated full construction; separate trace/STF and combined fit | One construction; trace/STF and combined fits retained |
| First-order phi | Separate near-duplicate implementation | Shared helper, fitted and theoretical comparisons retained |
| Second-order tau/phi | Frozen fitted/theoretical baseline, sparse correction candidates | Retained; inputs, crops, and sampled domains changed upstream |
| Filtering | Edge padding in x; invalid trimming could fall back to no trim; fixed 50-point crop plus later scale-dependent crop | Periodic x; wall-support crop in y; one derivative-halo crop for first order |
| Time and physical parameters | dt=(t_last-t_first)/Nt; extra factor 10 in recovery scale; momentum residual plot used kappa | dt from np.diff(t), uniform-grid validation, no factor 10; momentum uses nu |
| Sampling | Recovery domain count depended on library size; first-order counts depended on CPU count; tau sampling lacked a fresh seed | Configured count (1000 default), configured seed before each domain construction, allocation-aware workers |
| Plots | Repeated plots, configurable field/zoom cells, stale coordinates in some plots | Consolidated recovery/velocity/continuity and pointwise momentum/heat plots at the end; the old interactive field selector/zoom controls are not preserved |
| Flux diagnostics | Full-tensor contractions, Pearson correlation, relative errors, signed means | Retained; paper accuracy A and integrated mean-flux ratio added by this audit |
| Archives | Second-order candidates and flux archives | Also includes recovery, first-order and general closure reports, source snapshots, and batch status |

Numerical identity is not expected even for retained model families: filtering,
time scaling, integration domains, crops, and some search spaces changed. No
independent validation is established by any of these training-data reports.
Several old numeric-target cells had no saved output, so their intended targets
cannot be reconstructed reliably from their column numbers alone.

## Batch equivalence and usage

The new `run_filter_sweep.py` executes the **current tagged notebook cells**, so
the scientific code is shared with the current notebook. Matching data,
configuration, source, and requested stages is necessary for a like-for-like
comparison; it does not reproduce the deleted historical search modes.

- Interactive default: one width, Delta/dx=50, workers=1 outside Slurm, and plots
  when the final plotting cells are run.
- Slurm default: widths 10,25,50,75 in four serial array tasks, 32 workers per
  task, full discovery and flux diagnostics, no plots unless `--plots` is given.
- `--skip-flux` and `--skip-second-order` explicitly omit stages. COMPLETE means
  requested execution finished, not that accuracy passed a scientific threshold.
- Existing `slurm/run_convection*.sbatch` and `slurm/paper-test/*.sbatch` were
  unchanged. Those run Dedalus simulations; the new batch job analyzes existing
  preprocessed data and does not replace DNS generation/preprocessing.
- `--compare-runs` is postprocessing only: it reads completed flux archives and
  runs the final comparison cell without loading the DNS fields or fitting models.

## Accuracy in the root plot

`plot.png` reproduces part of Figure 2 of the local paper
`2D_Incompressible_SGS_Paper_2 (44).pdf`. Equations (3.8) and (3.9), on pages 8–9,
define

```text
A(X) = integral <X_DNS : X_model> dt
       / max(integral <X_DNS : X_DNS> dt,
             integral <X_model : X_model> dt)
```

The contraction sums all tensor/vector components. No mean subtraction occurs.
The metric penalizes amplitude mismatch; Pearson correlation does not. For
example, if X_model=2 X_DNS with nonzero variation, A=1/2 while Pearson=1.
Values can be negative. When both fields vanish, the formula is undefined and
the implementation records null.

The earlier flux additions reported Pearson correlation, not this A. They now
also record `closure_accuracy_A`, `flux_accuracy_A`, and `mean_flux_ratio`, for
both tau and phi, using spatial means and trapezoidal time integration. A
near-zero signed reference mean makes the mean-flux ratio undefined; the saved
relative tolerance is applied against the reference RMS. Pearson and the old
equal-snapshot diagnostics remain separately labeled.

Single-run plots show these metrics against candidate ID. The comparison cell
plots them against filter width across completed runs. Its default axis is
Delta/dx; supplying a physical reference length changes it to Delta/reference
length. It does not silently equate grid units with the paper's nondimensional
delta. Default comparison curves use fitted/theoretical first-order baselines;
optional correction curves are independently refitted at each scale and can
have different supports. The convection models are not the five model families
in the paper. The valid vertical region also changes with filter-support trimming.

Old schema-1 flux archives cannot supply the new closure accuracy from the
saved relative errors alone. Rerun the flux evaluation and archive cells using
the existing fitted models in a live kernel, or rerun the batch workflow. The
comparison skips old/incomplete archives and rejects mixed data/settings or
duplicate model/scale points rather than silently combining them.
