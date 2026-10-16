# Native Gauss recovery on PACE

The new entry point is `COnvection-FILTER-Chebyshev-Gauss.ipynb`, using the
unmodified `pjo0213/PySPIDER:upstream-plus-quadrature` revision
`ccc892cdc84fb183c13531a7fa654132acbe1818`. Its dependency is pinned in
`requirements-chebyshev-gauss.txt`. The existing Lobatto notebooks and their
old requirement remain available as a historical comparison. The vendored
`SPIDER/` is a different implementation; this notebook imports `PySPIDER`.

## Findings from the source comparison

Compared directly with the old pin `91b9d6b3dee9d8c22118a9cd4a30132ea3fb5196`:

- The fork adds `chebyshev-gauss` (Fejer-I integration on first-kind roots)
  and `truncated-cg-grid`. These integrate the ordinary physical measure dz;
  they are not the classical Gauss-Chebyshev rule with weight 1/sqrt(1-z²).
- Quadrature helpers now take **number of nodes**, not number of intervals.
  The old filter call passing `n-1` must not be reused with this version.
- Domain weight scaling now uses the integration interval, including the
  unsampled endpoints of Gauss grids. Truncated/moment-matched integration
  maps the weight polynomial into local domain coordinates.
- Nonuniform finite differences using `grids` were already present at the old
  pin. The regression solver is unchanged between the two pins.
- Dedalus ChebyshevT output in this simulation uses Gauss nodes. This is not
  a claim about every Dedalus basis. The new preprocessor checks the actual
  coordinates against mapped `chebpts1` before proceeding.

Source: [fork integration API](https://github.com/pjo0213/PySPIDER/blob/ccc892cdc84fb183c13531a7fa654132acbe1818/src/PySPIDER/commons/integration.py),
[quadrature implementation](https://github.com/pjo0213/PySPIDER/blob/ccc892cdc84fb183c13531a7fa654132acbe1818/src/PySPIDER/commons/quadrature_schemes.py).

The new preprocessing notebook retains the original Gauss samples without a
spectral conversion. Filtering evaluates the Gaussian convolution on a complete
Gauss grid in the support-safe interior, using physical interval endpoints
rather than first/last sample locations. It still evaluates filtered fields at
new interior coordinates; it does not merely crop the original grid.

The existing full stress, ordinary pressure, periodic x filtering, corrected
time spacing, omission of vorticity, and numerical column pruning are retained.
The known momentum balance is evaluated before pruning/regression. Full-height
vertical domains remain a deliberate limitation: cropped Gauss nodes are not
a complete Gauss grid. The new truncated-grid API could support local domains,
but its moment-matching conditioning needs a separate convergence study.

The supplied October logs name `COnvection-FILTER-Copy2.ipynb` and
`run_filter_sweep.py`, with a forced-column second-order closure regression.
That runner/version is not present on this branch. The logs end in a singular
normal-equation solve. Zero/proportional-column removal cannot eliminate all
multicolumn dependencies, and the new fork does not change that solver.
Those logs do not establish that the Lobatto conversion caused the failure.
This new workflow diagnoses filtered momentum recovery, not that separate
second-order closure sweep.

## Install and submit

Use PACE, from this repository root. The examples use the existing `SPIDER`
Conda environment, Python >=3.12, and the repository's account/QOS defaults.
Install into a separate environment if retaining an installed old pin for
comparison. The package version remains 2.1.0, so force reinstall the exact pin:

```bash
module load anaconda3
source activate SPIDER
python -m pip install --no-deps --force-reinstall -r requirements-chebyshev-gauss.txt
python -m pip check
python -m ipykernel install --user --name spider-gauss --display-name 'SPIDER Gauss'
```

The environment also needs JAX/JAXlib, h5py, nbconvert and ipykernel, along with
the fork's dependencies. `--no-deps` preserves the existing scientific stack;
resolve missing dependencies before submitting. The notebook verifies the VCS
commit in the installed distribution's `direct_url.json`, not just its version.

First run preprocessing and a 32-domain diagnostic:

```bash
export SPIDER_SOURCE=/storage/scratch1/2/ashelestov3/snaps_paper/snaps_paper_s1.h5
export SPIDER_DATA=/storage/scratch1/2/ashelestov3/preprocessed_paper_data_gauss.npz
SPIDER_PREPROCESS=1 sbatch --export=ALL slurm/paper-test/run_gauss_recovery.sbatch
squeue -u "$USER"
```

Preprocessing refuses to overwrite an existing NPZ. If already prepared, omit
`SPIDER_PREPROCESS=1`; to regenerate, choose a new `SPIDER_DATA` name. The input
currently uses the production file's coordinate hash keys and physical z bounds
[0,1]. A different simulation requires checking those keys and bounds in the
preprocessing notebook. Snapshot pressure, buoyancy, velocity and vorticity
tasks must exist; vorticity is retained only in the data archive.

The batch script requests one process, four CPUs, 512 GB and 12 hours as starting
resource estimates. Adjust `sbatch --mem=... --time=...` to the dataset and queue;
these are not measured requirements. It uses serial matrix assembly because
forking after JAX initialization is unsafe. It creates executed notebooks under
`executed-notebooks/<job-id>/` and a Slurm log `rb_gauss-<job-id>.out`.

After the diagnostic checks below, request regression with more domains:

```bash
SPIDER_TEST_NDOMAINS=256 SPIDER_DIAGNOSTIC_ONLY=0 SPIDER_SEED=1 \
  sbatch --export=ALL slurm/paper-test/run_gauss_recovery.sbatch
SPIDER_TEST_NDOMAINS=512 SPIDER_DIAGNOSTIC_ONLY=0 SPIDER_SEED=2 \
  sbatch --export=ALL slurm/paper-test/run_gauss_recovery.sbatch
```

The notebook itself defaults to 256 domains; the batch script defaults to a
32-domain diagnostic. Other controls: `SPIDER_FILTER_DX` (default 50),
`SPIDER_ARCHIVES` (default `run-archives`), `SPIDER_CONDA_ENV` (batch environment),
and optional `SPIDER_RUN_PROVENANCE` (a user-supplied JSON run record to archive).
Do not set the parallel override for this combined JAX/recovery workflow.

## What to inspect

1. **Input and filter checks.** Grid checks and uniform x/time checks must pass.
   Time differences come from `np.diff(t)` with no factor of ten. Quadrature
   polynomial and constant-filter checks run before filtering the data.
   `filter-checks.json` records constant, linear and second-moment errors.
   Compare the second-moment error with delta²/12; substantial error means the
   filter is underresolved. Passing these checks does not establish derivative
   commutation or conservation for the turbulent fields.
2. **Prescribed momentum residual.** Inspect `known-momentum.json` and printed
   contribution norms for dt(u) + u.grad(u) + grad(p) - nu*lap(u) - b*z_hat
   + div(tau). The reported residual is norm(sum contributions) divided by the
   sum of their norms. Smaller is better; there is no established universal
   acceptance tolerance. If it is large, investigate filtering, differentiation,
   data conventions and simulation parameters before trusting regression.
   Repeat with more domains and compare filter widths on the same data.
3. **Matrix geometry.** Inspect `matrix-diagnostics.json`, including singular
   values, row/column counts and numerical ranks. A true equation creates a
   near-null direction, so rank loss alone is not a failure. Many unconstrained
   directions or fewer training rows than candidate terms can make recovery
   ambiguous. More domains help sampling but do not cure exact dependencies.
4. **Candidates and stability.** Compare accepted equation coefficient ratios
   with the prescribed momentum coefficients after normalizing dt(u) to one.
   Inspect train and test errors and all support sizes, not just the printed
   winner. Repeat seeds and domain counts. Random row splits can put overlapping
   domains, and different tensor components of one domain, in both sets;
   they are not independent physical validation. Thesis selection needs a
   separate time block or simulation and recorded validation/selection reasons.
5. **Failure records.** A run lacking `complete.json` is incomplete. Check
   `failed.json`, attempt failure records and the Slurm log. Earlier failures
   (loading/filtering/matrix construction) may leave only partial checkpoints.
   A singular regression is not evidence for a physical equation.

Rayleigh=1e8 and Prandtl=1 are assumptions from repository source. Confirm them,
the velocity components, boundary conditions, resolution, dealiasing and time
integration configuration against the production run. Dedalus tau lifts and
discretization errors are additional possible contributors to a nonzero balance.

## Archives and validation status

Every recovery execution creates a timestamp-plus-UUID directory and snapshots
both notebooks, requirements, this guide, all Dedalus sources and the imported
PySPIDER package. It records git revision/dirty status, installed packages,
data path/size/mtime/SHA256, physical coefficients, settings, exact domains,
original and pruned matrices, term ordering, pruning and balance diagnostics.
Each regression attempt records its selection, settings, split rows, returned
candidate coefficient arrays and errors, even when the candidate is rejected.
The fork exposes best candidates at each support size, not every transient
optimizer iterate; those returned candidates are all preserved. Physical and
internally scaled coefficient/error conventions are explicitly distinguished.

Previous run directories are preserved. Source snapshots are labeled inferred
configuration, not confirmed simulation provenance. Candidates are labeled
training fits; the thesis-selected set is empty. Keep the executed notebook and
Slurm log with the archive when transferring results. NPZ hashing adds an I/O
pass, and source/matrix snapshots require extra storage.

Development validation was limited to static source comparison, notebook JSON
inspection and whitespace checks. No scientific tests or pipeline executions
were performed locally; successful recovery remains to be demonstrated on PACE.
