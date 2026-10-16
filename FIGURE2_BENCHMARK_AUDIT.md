# Figure 2 audit and reuse of sweep 14094098

## Current workflow: measure first, discover models, compare both families

Use this workflow for the requested six-width comparison. The prescribed-only
commands later in this document are optional recovery of the original sweep;
they do **not** perform the requested model discovery.

After updating the PACE checkout, measure the convection integral scale first:

```bash
bash slurm/submit_filter_workflow.sh measure
```

The submission directory's `logs/rb_measure_li-*.out` prints the completed measurement directory.
Its `scale_plan.json` contains the measured `li`, physical widths,
`delta_dx`, and `delta_over_li`. The notebook computes the ratio of spatially
and temporally averaged Frobenius norms in paper Eq. (3.3), using unfiltered
velocity, sixth-order finite differences, periodic x, and a 50-point wall
exclusion in y. It uses all selected times with trapezoidal weights. This is
a convection adaptation, not the paper's spectral calculation on F1/F2.

The default normalized widths are **0.05, 0.2, 0.4, 0.6, 0.8, 1.0**. If the
largest width exceeds the current crop/support limit, the five original widths
are reduced uniformly; the extra 0.05 stays exact. Duplicate widths are removed.
The plan records the resulting widths explicitly. No widths are rounded to the
old grid widths. An unresolved 0.05 width or insufficient support raises an
error rather than silently changing that requested scale. Existing measurement
plans remain unchanged; run measurement again to create a new six-width plan.
The plan, moments, input member hashes,
measurement settings, and source snapshots are saved in a unique archive.
No numerical value of `li` has been measured in the development environment.

Once that job succeeds, set `PLAN` to its actual file and launch:

```bash
PLAN=/absolute/scratch/path/printed/by/measurement/scale_plan.json
OLD=/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/sweep_14094098
bash slurm/submit_filter_workflow.sh discover \
  --scale-plan "$PLAN" --reuse-runs "$OLD"
```

The helper defaults to `--array=0-5%1` for six widths. Check the printed plan;
if support limiting deduplicated a width, set `SWEEP_ARRAY=0-4%1` instead.
It also supports existing five-width plans with that override. Discovery mode executes
loading/filtering, general tau/phi discovery, archiving, and flux evaluation
for both discovered and prescribed models. It skips `second-order-regression`
and the other frozen-baseline calibration/residual stages. That skipped cell
does discover additive residual corrections; it is separate from general
model discovery and is unnecessary for these comparisons. General discovery
keeps the existing SPIDER power initializer that completed at every old width.

Completed widths 25 and 50 are preserved. If either exactly matches a planned
width and passes compatibility/integrity checks, its saved library integrals
and general fitted equations are reused without integration or refitting.
The old runs did not save general-model flux metrics, so those metrics still
require evaluation. If neither width matches, the new widths require new
fits; the old widths are not relaunched or substituted into the uniform grid.
Reused fits retain their original source/provenance archive in the new archive.

After all tasks succeed, the array automatically submits comparison.
Alternatively, rerun only auditing/plotting with the new sweep directory:

```bash
bash slurm/submit_filter_workflow.sh plot \
  --compare-runs /absolute/scratch/path/to/the/new/sweep_JOB \
  --scale-plan "$PLAN" --discovery-benchmarks
```

Add `--verify-only` to audit without drawing. The comparison checks the plan,
archive consistency, all planned widths, and complete discovered-model curves.
It produces `figure2_tau_P1_vs_delta.*` through `P5`, and equivalent phi
figures, in PNG/PDF/SVG. Each compares discovered, prescribed gradient, and
prescribed gradient-plus-Hessian models at a fixed requested predictor budget;
tau uses that budget separately for trace and STF. All candidates are saved,
including unequal-budget pairs. These are exploratory training fits, not
independently validated or thesis-selected models. The three panels show
closure accuracy, local-flux accuracy, and mean-flux ratio against Delta/ell_i.

## Launch without model discovery

Use the same measured plan:

```bash
bash slurm/submit_filter_workflow.sh prescribed --scale-plan "$PLAN"
```

This computes only prescribed gradient and gradient-plus-Hessian flux benchmarks
for both tau and phi; it performs no library integration or regression. Do not
pass `--reuse-runs` in this mode. Each submission is separate. Choose either
mode: discovery mode already includes the prescribed benchmarks, so submitting
both is unnecessary unless you specifically want separate runs.

## Scratch storage isolation

The submission helper creates a unique directory under
`/storage/scratch1/2/ashelestov3/rayleigh-benard/filter_sweeps/` and prints it.
Slurm logs and the submission receipt are saved there, with measurement/sweep
archives, bootstrap files and runtime directories beneath it. The dependent
plot job saves its figures/audit under the sweep and its log under `logs/`.

The Python runner executes from a small source snapshot in its own
`runtime/work/` on scratch. It copies the notebook and relevant sources, not
DNS data or `.git`. Imports use the scratch SPIDER copy. Python bytecode,
Matplotlib, JAX, Numba, CUDA, joblib, IPython/Jupyter and XDG caches/configuration,
and temporary files are redirected to scratch. Relative incidental output also
lands in the scratch working directory. Git metadata is read from the original
checkout with optional index writes disabled. The data file is read in place.
No generated files are intentionally written to the repository or project/home
storage. Old local `.out` files and plots are left untouched.

For manual comparison submissions with logs isolated in a fresh scratch folder:

```bash
bash slurm/submit_filter_workflow.sh plot \
  --compare-runs /absolute/scratch/path/to/sweep_JOB \
  --scale-plan "$PLAN" --discovery-benchmarks
```

Use `--prescribed-only` instead for a prescribed-only sweep. Direct `sbatch`
commands still use scratch, but their default bootstrap logs are at the scratch
root; use the helper to keep those logs in a separate submission folder too.

## Relation to the paper

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

## Original sweep filter scales

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

## Optional prescribed-only recovery of the original four widths

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
