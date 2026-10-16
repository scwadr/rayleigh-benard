# Figure 2 audit and reuse of sweep 14094098

The local paper `2D_Incompressible_SGS_Paper_2 (44).pdf`, pages 3–9,
defines Figure 2(a), (b), and (c) as velocity flows F1, F2, and F3. They
are not stress and scalar-flux panels. F1 has inverse-cascade initial
conditions; F2 has freely decaying initial conditions. These benchmarks use
viscosity 1e-5, a periodic square of side 2*pi, and five models: NGMR,
NGM4, similarity, dynamic mixed, and dynamic Smagorinsky.

The supplied convection workflow is a different data set. Its tau and phi
plots are useful analogous diagnostics, but cannot reproduce F1/F2 or the
four unavailable benchmark models. The paper's NGMR includes the residual
evolution equation (3.7); algebraic corrections discovered by the notebook
are not this model. Original F1/F2 DNS data and the remaining model
implementations are needed for the full paper comparison. The scalar phi
expansion is an extension to convection, not a curve in this paper.

## What matches and what differs

Source inspection confirms that the flux cell uses the prescribed stress
formulas (3.1)–(3.4): gradient coefficient Delta^2/12, Hessian coefficient
Delta^4/288, and both mixed-derivative contributions. It contracts stress
with the symmetric velocity gradient for energy flux. The accuracy uses
the ratio of time-integrated moments in (3.8)–(3.9), with trapezoidal time
weights and spatial/component contractions. The three plotted metrics are
closure accuracy, local-flux accuracy, and mean modeled/reference flux.

The current filter uses Gaussian sigma=Delta/sqrt(12), finite support,
edge padding, and retained fine-grid values. The paper additionally
projects onto a coarse grid with resolution Delta. The notebook's
`alpha*(delta+1)` initial trimming has a fallback that leaves these arrays
untrimmed, followed by a fixed 50-point crop. Full sweep runs then apply
the baseline-fit support crop plus derivative margin. The new prescribed
mode preserves that same support crop so existing results can be reused;
it does not correct the underlying filtering convention or introduce the
paper's projection. First derivatives use numpy.gradient and Hessians use
sixth-order finite differences. These are not an assertion of numerical
equivalence to the paper's spectral workflow.

The comparison labels the results as convection stress/scalar flux and
records these limitations in `analysis.txt`. Matching formulas and archive
integrity are not independent numerical validation of the computed fields.

## Filter scales

The logs give dx=0.0009765625. The requested scales therefore are:

| Delta/dx | Physical Delta |
|---:|---:|
| 10 | 0.009765625 |
| 25 | 0.0244140625 |
| 50 | 0.048828125 |
| 75 | 0.0732421875 |

The paper plots delta=Delta/ell_i over approximately 0–1, where ell_i is
the derivative-based integral scale in (3.3). Its table gives approximate
ell_i values of ell_d/30 for F1 and ell_d/4 for F2. These lengths belong to
those flows; do not substitute them for a measured convection scale.
Axis ticks 0, .25, .5, .75, 1 do not specify the paper's exact sampled
filter widths. The existing four widths cannot yet be certified as
covering a similar normalized range because no convection ell_i has been
computed/confirmed. Given a confirmed ell_i, convert any desired normalized
width using Delta/dx = delta * ell_i / dx. Use `--reference-length` with
that ell_i when plotting. Omitting it truthfully labels the axis Delta.

Do not change widths merely to imitate the paper's axis before determining
this scale. The commands below finish only the original four-width
convection benchmark; they do not claim matched F1/F2 scales.

## Verify existing results before launching anything expensive

Copy the updated notebook, runner and Slurm scripts to the PACE checkout.
From that checkout:

```bash
ROOT=/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/sweep_14094098
sbatch --wait slurm/plot_filter_sweep.sbatch \
  --compare-runs "$ROOT" --expected-scales 25 50 \
  --prescribed-only --verify-only
```

This performs no filtering, library integration, regression or field/flux
recomputation. It checks complete execution/cell markers, executed-notebook
checksum, discovery/flux manifest agreement, source snapshot checksums,
finite shared-library arrays, retained coordinate/time alignment, prescribed
coefficients, moment finiteness and bounds, trapezoidal weights, candidate
IDs, and agreement between JSON metrics and metrics recomputed from the
stored moments. It also requires exactly the two requested completed widths
and compatible fixed-model settings. Failed original runs are recorded as
skipped, not treated as successes.

Success prints `Archive integrity checks passed for Delta/dx: [25.0, 50.0]`.
A unique `figure2_delta_.../` directory contains `analysis.json` with
`audits`, `sources`, and `skipped`, `analysis.txt`, `metrics.csv`, source
copies, and `COMPLETE`. Verification-only mode creates no figures.
The Slurm log is `/storage/scratch1/2/ashelestov3/rb_filter_figure2-<job>.out`.
Any failed check raises an exception and the job exits unsuccessfully.

This verifies stored archive consistency. It does not recompute the moments
from DNS, recover missing full field snapshots, or establish historical
simulation provenance. Input identity in these archives is path/stat plus
coordinates, not a full hash of the original fields. Only the downloaded
logs are available locally, so the PACE archive checks have not been run here.

## Finish only missing widths, preserving successful runs

This guarded submission launches tasks 0 and 3 (widths 10 and 75) only
after the two existing archives pass verification:

```bash
ROOT=/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/sweep_14094098
if sbatch --wait slurm/plot_filter_sweep.sbatch \
    --compare-runs "$ROOT" --expected-scales 25 50 \
    --prescribed-only --verify-only; then
  FILTER_SCALES="10 25 50 75" COMPARE_SWEEP=0 OUTPUT_ROOT="$ROOT" \
    sbatch --array=0,3%1 slurm/run_filter_sweep.sbatch --prescribed-only
fi
```

Run either the first verification alone or this guarded version; they use
the same audit. Neither command relaunches widths 25 or 50. New runs go
under `sweep_14094098/sweep_<new-array-id>/`; previous archives remain intact.
The prescribed mode executes setup/filtering, fixed-model crop preparation,
provenance archiving, and tiled flux benchmarks for both tau and phi. No
weak libraries, general discovery, gradient fitting, or residual regression
are executed. `--skip-baseline-residuals` alone does not provide this mode:
it still runs general discovery and changes the evaluation crop.

Once the two new tasks complete, validate all four widths and generate both
convection plots from the existing and new archives:

```bash
sbatch slurm/plot_filter_sweep.sbatch \
  --compare-runs "$ROOT" --expected-scales 10 25 50 75 --prescribed-only
```

The output includes `figure2_tau_N0_vs_delta.*` and
`figure2_phi_N0_vs_delta.*` in PNG, PDF and SVG. They contain the fixed
gradient and gradient-plus-Hessian curves only. All archived candidates
remain in the audit/CSV; none are deleted or silently relabeled as NGMR,
similarity, mixed, or Smagorinsky. Add `--reference-length <measured-ell_i>`
only after establishing that length from the relevant data.

To inspect just the existing two widths now, use the last plotting command
with `--expected-scales 25 50`. This creates both plots without rerunning
either successful simulation/discovery workflow.

Local verification was limited to source inspection, Python/notebook syntax
parsing, and diff review. Numerical execution is reserved for PACE.
