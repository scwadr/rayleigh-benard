# Compare archived sweep models

Run on a PACE compute node with the analysis environment activated:

```bash
MPLBACKEND=Agg python scripts/compare_sweep_models.py /absolute/path/to/figure2_archive
```

The input directory is the downloaded `rundata` equivalent: it must contain
`metrics.csv`, `analysis.json`, and `COMPLETE`. No discovery notebook, PySPIDER
import, original DNS arrays, filtering, or regression is needed. This standalone
archive reader also works on the feature branch without importing main's workflow.

The default compares discovered P1, P2, P3, prescribed gradient (NGM2), and
prescribed gradient plus Hessian (NGM4). P4/P5 are excluded. Override with
`--budgets 1 2` if desired. P denotes requested predictors, excluding the target;
stress uses that budget separately for trace and symmetric trace-free fits.
Unequal-budget stress pairs and zero-predictor candidates are excluded.

A unique `model_comparison_<timestamp>_<id>/` directory contains:

- `models_vs_dns.png` and `.pdf`: one figure, with stress on the left and scalar
  flux on the right; rows show closure accuracy, local-flux accuracy, mean-flux ratio.
- `selected_models.csv`: the plotted rows with candidate IDs and source paths.
- `pairwise_score_differences.csv`: scores A and B and their difference for each
  pair of models at each width. Higher accuracy is better; mean-flux ratio should
  be closer to one, so a positive difference is not automatically an improvement.
- Source hashes, a script copy, interpretation notes, and a completion marker.

Use `--output-root /absolute/scratch/path` to change the output parent. Previous
archives/results are never overwritten. Undefined metrics remain gaps. Missing
model/width combinations or duplicates raise errors instead of selecting a model.
The script trusts the prior archive audit; it does not revalidate DNS moments.

## What the paper compares

The repository paper, Section 3.2, equations (3.8)-(3.9), and Figure 2 (page 9),
compares each model with quantities computed from filtered DNS. Its models are
NGMR, NGM4, similarity, dynamic mixed, and dynamic Smagorinsky. Panels (a), (b),
and (c) are flows F1, F2, and F3. Each has stress accuracy, local energy-flux
accuracy, and mean modeled/DNS energy-flux ratio. Accuracy is a magnitude-aware
cross-correlation, with tensor contractions and spatial/time integration:

`A(reference, model) = integral(reference * model) / max(integral(reference**2), integral(model**2))`.

It is not Pearson correlation. All three plotted metrics ideally equal one.
Our discovered P2 is not automatically NGM4, and P3 is not NGMR: the latter
includes a separate residual evolution equation. The scalar comparison is a
convection extension. These plots compare training-data performance of our
models against their common reference, not independent validation on F1/F2.

Pairwise score differences are not model-to-model field differences. Computing
the latter requires model/model cross-moments or reevaluation of the fields;
the model/reference scores alone do not determine them.

Validation: Python syntax and whitespace checks only. Plotting and numerical
execution must be validated on PACE; no local scientific execution was performed.
