# Notebook restoration record

Restoration source: `102c126:COnvection-FILTER-Copy2.ipynb`, immediately before
the batch refactor. The previous consolidation changed scientific defaults and
removed searches; it was not equivalent. This restoration reverses those changes.

## Retained from the original

| Part | Restored behavior |
| --- | --- |
| Data preparation | Original loading, JAX filtering, stress/scalar-flux construction and fixed crop |
| Broad library | Original observables, irreps, complexity 4, maximum 2 observables |
| Integration | Original domain-count formula, 64 spatial / 40 temporal domain widths, original scales |
| Homogeneous AIC | Original max_k=5 through 39 sweep, power initialization, 75% training rows |
| Homogeneous jump | Original max_k=10, 80% training rows, original thresholds and iteration settings |
| General tau/phi discovery | Original shared-library selection, predictor cleanup, and 1-6-term search |
| Trailing inhomogeneous regressions | Six original target/initializer/settings combinations in one table |
| Gradient comparison | Original per-irrep and combined tau calibration, phi calibration, theoretical coefficient Delta^2/12 |
| Residual discovery | Original fitted/theoretical baseline residual searches and reports |
| Plots | Original controls and diagnostics, repeated copies removed, plots at the end |

The original `dt=(t[-1]-t[0])/Nt`, recovery time-scale factor 10, edge-padded
filtering and trimming fallback, fixed 50-point crop, and `w` in the library are
retained. The separate TODO fixes have not been silently applied.

## Deliberate differences

- Optional batch controls supply data path, filter width, resources, time window,
  and explicit search limits. Without overrides, original notebook defaults apply.
- Duplicate standalone tau calibration is removed; its printed equation and the
  combined fit remain. Calibration is labeled as calibration.
- The six numerical-target experiments are consolidated without substituting
  momentum/heat targets. Original vector column 22 requests six terms with
  `max_k=5`. That impossible request is archived as skipped; valid requests run.
- Empty cells, an isolated lookup/print, inactive quoted copies, and duplicate plot
  blocks are removed. Stale saved outputs are cleared.
- Run controls, result capture, source snapshots, provenance, and unique archives
  remain notebook cells. The temporary homogeneous-regression script is deleted.
- Flux diagnostics now batch all times and accumulate the moments for paper
  equations (3.8)-(3.9) and Figure 2: closure accuracy, local-flux accuracy, and
  mean-flux ratios (plus relative errors). Per-frame snapshots, correlations, and
  transfer-sign diagnostics were removed. This changes the diagnostic implementation,
  not the discovery searches or fitted coefficients.
- Flux evaluation uses concurrent spatial tiles with halos, sharing input fields
  read-only and caching derivatives and contracted terms per tile. Point-count
  weighting preserves the full-grid average; all times and candidates are retained.
  Only reduction order changes. Batch timings/cache statistics and the executed
  flux cell source are archived separately from the earlier discovery provenance.
- Both prescribed gradient/gradient-plus-Hessian models and fitted/theoretical
  baselines with discovered residual corrections are evaluated on common support.
  Phi uses the analogous Gaussian scalar-flux expansion; the paper's stress result
  is not represented as scalar-model validation. General-discovery flux evaluation
  is opt-in; all general discovery results remain in the discovery archives.
  Every trace/STF pairing is retained numerically; default comparison curves use
  matching requested budgets, without error-based model selection. Earlier flux
  archives are preserved and not mixed with the new schema-4 comparisons.

Core stress, phi, crop, library, and phi-discovery call cells were compared
directly against the restoration source. Other scientific cells have runtime
overrides and archival hooks around the original operations. Static JSON/AST and
source review were performed; numerical equivalence has not been tested here.
Runs require PACE. Random sampling, worker-dependent domain rounding, and explicit
overrides can change numerical results.

The restored homogeneous search again explores the broad library containing the
filtered continuity, momentum (including tau), and heat (including phi) relations.
It does not guarantee recovery. Inspect returned equations and residuals.
Successful execution is not scientific validation or thesis model selection.

See [PACE_FILTER_SWEEP.md](PACE_FILTER_SWEEP.md) for notebook order and batch usage.
