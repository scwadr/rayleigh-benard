# Pipeline run report

This is the single detailed run report for the feature branch. For installation,
data creation, and both pipeline entry points, start with [README.md](README.md).

## Current status

- Native Gauss and the feature branch's Copy2 use one pinned, unmodified
  PySPIDER package. Main's newer closure-discovery/sweep workflow is not included.
- A successful PACE run was reported on October 17, 2026. The original tagged
  run is identifiable at `e431fcb`; the consolidated feature state is `7d60090`.
- Dependency consolidation and documentation changes received static checks
  only, not a new PACE run. Detailed recovered coefficients and the successful
  run's archive identifier have not been recorded here.
- The final section preserves the pre-consolidation compatibility audit against
  main. Its migration requirements and merge conflicts describe those historical
  revisions, not unfinished work on this feature branch.

## Native Gauss run details

The new entry point is `COnvection-FILTER-Chebyshev-Gauss.ipynb`, using the
unmodified `pjo0213/PySPIDER:upstream-plus-quadrature` revision
`ccc892cdc84fb183c13531a7fa654132acbe1818`. Its dependency is pinned in
`requirements/chebyshev-gauss.txt`. The obsolete Lobatto notebooks and their
old requirement are available in Git history. Both feature-branch notebooks
now use the same pinned external PySPIDER via `pyspider_runtime.py`; no
vendored package remains. Main's newer model-discovery workflow is not included.

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

### Use a Git checkout without pip

This option leaves installed packages unchanged. In a PACE terminal, clone to
a new directory, separate from the checkout used by your running sweep:

```bash
git clone --branch upstream-plus-quadrature https://github.com/pjo0213/PySPIDER.git "$HOME/PySPIDER-gauss"
git -C "$HOME/PySPIDER-gauss" checkout --detach ccc892cdc84fb183c13531a7fa654132acbe1818
```

In a fresh Jupyter kernel, run this configuration cell **before** the recovery
notebook's first code cell:

```python
import os
os.environ["SPIDER_CHECKOUT"] = os.path.expanduser("~/PySPIDER-gauss")
```

Then run the recovery notebook's existing code cells in order. The first code
cell puts this checkout's `src` directory first on this kernel's import path,
checks the pinned commit and clean package source, and prints the actual imported
path. The archive records the checkout and snapshots its package source. An old
installed `PySPIDER_SR` can remain in the environment; its distribution metadata
is not used to identify the active source checkout. Restart the kernel before
changing source checkouts. Do not update this checkout while its jobs are running.

For the batch launcher, set the same selection before submitting:

```bash
export SPIDER_CHECKOUT="$HOME/PySPIDER-gauss"
SPIDER_PREPROCESS=1 sbatch --export=ALL slurm/paper-test/run_gauss_recovery.sbatch
```

Existing dependencies still need to be available in the kernel (Python >=3.12,
NumPy, SciPy, z3-solver, findiff, numba-kdtree, JAX/JAXlib and h5py). This bypasses
installation of PySPIDER itself, not its dependencies. The batch launcher also
needs nbconvert and the `spider-gauss` kernel registration shown below. For
interactive Jupyter, an existing compatible kernel is sufficient.

You may use the existing environment for this option if its dependencies already
work, since no installed files are changed. Keep this project's notebooks and
output paths separate from the running sweep as well. No pip command below is
needed when using `SPIDER_CHECKOUT`.

### Alternative: install the pinned package

Use PACE, from this repository root. The examples use the existing `SPIDER`
Conda environment, Python >=3.12, and the repository's account/QOS defaults.
Install into a separate environment if retaining an installed old pin for
comparison. The package version remains 2.1.0, so force reinstall the exact pin:

```bash
module load anaconda3
source activate SPIDER
python -m pip install --no-deps --force-reinstall -r requirements/chebyshev-gauss.txt
python -m pip check
python -m ipykernel install --user --name spider-gauss --display-name 'SPIDER Gauss'
```

The environment also needs JAX/JAXlib, h5py, nbconvert and ipykernel, along with
the fork's dependencies. `--no-deps` preserves the existing scientific stack;
resolve missing dependencies before submitting. The notebook verifies the VCS
commit in the installed distribution's `direct_url.json`, not just its version,
and verifies that the imported package comes from that distribution.

### Run the pipeline

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

The checked-in notebook hard-codes 32 domains. To honor the environment
variable, use `ndomains = int(os.environ.get("SPIDER_TEST_NDOMAINS", "256"))`
in its domain-setup cell. The launcher defaults to a 32-domain diagnostic. Other controls: `SPIDER_FILTER_DX` (default 50),
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

Repeated pruning/archive execution now creates a new timestamp-plus-UUID archive
before writing matrices. NPZ writes use exclusive creation, just like JSON.
The new archive inherits input identity and source snapshots, records its parent,
and separately saves the notebook currently on disk. Repeating the final
regression cell copies its matrix checkpoints to a fresh archive, so switching
from diagnostics to regression does not require rebuilding matrices. A changed
domain count still requires rerunning domain/matrix construction and diagnostics.

After pulling this fix into an active session, copy the updated pruning/archive
cell and final regression cell into the open notebook without restarting its
kernel. Run the pruning/archive cell and then the regression cell. The new code
keeps an in-memory pre-pruning snapshot for subsequent retries. If an old kernel
already pruned before this fix, its earlier removed columns remain in its parent
archive; the new snapshot describes the currently available input columns.

In the older implementation, a repeated archive cell could overwrite
`matrix-original-0.npz` before failing on `terms-original-0.json`. Such an archive
may mix attempts and must not be treated as an intact record of its first run.
The fix prevents future overwrites but cannot restore an already replaced file.

The user reported a successful PACE run on October 17, 2026. Detailed coefficients
and the corresponding archive identifier have not been recorded in this guide.
Development validation remains limited to static source review, notebook JSON
inspection and whitespace checks; no scientific tests or pipeline executions
were performed locally.


## Historical compatibility and merge audit

Audited 2026-10-17 before consolidation. The following records the comparison
against main and the migration plan at that time. Current package setup is
described above and in the README. References to vendored code and merge
requirements in this section are historical.

### Decision

One implementation can support both workflows: use the unmodified pinned
`PySPIDER_SR` fork already required by native Gauss. There is no separate
regression or symbolic-library implementation in main that requires retaining
the vendored package for new runs. Migration is required before deleting
`SPIDER/`: imports, package provenance, scratch snapshots, and archive reuse
currently depend on that directory.

This is source-level compatibility for main's real-valued, positive-spacing,
uniform-grid datasets. It is not demonstrated numerical equivalence, identical
model selection, or permission to label legacy fits as validated with the new
package. Preserve the old revision for reproducing old runs.

Sharing the package does not make main's uniform-grid filtering and closure
pipeline native-Gauss-aware. Continue feeding it interpolated data; running its
closure/flux workflow directly on Gauss samples would be a separate numerical
port, beyond changing imports or choosing one dependency.

### Exact sources examined

- Remote `main`: `61cf03ebbf0b382bc723eea311b36452cf107e14`, confirmed with
  `git ls-remote origin refs/heads/main`.
- Native branch/tag: `e431fcb`, `successful-native-chebyshev-run`.
- External fork: `ccc892cdc84fb183c13531a7fa654132acbe1818`, checked out detached
  from `pjo0213/PySPIDER`; external source was not modified.
- Merge base: `000f3b87bf6c42d7192d2d0587b1f870522bdb02`.

The vendored Python files are identical on the two project branches. Comparing
the 24 vendored `.py` files against the pinned fork, with line endings normalized,
found 21 identical files and three changed files. The fork adds two Python
modules: `commons/integration.py` and `commons/quadrature_schemes.py`.

| Component | Comparison and implication |
| --- | --- |
| `commons/library.py`, `commons/z3base.py`, `continuous/library.py` | Identical: observables, terms, indices, irreps, equation representation and library generation are unchanged. |
| `commons/sparse_reg.py`, `commons/sparse_reg_bf.py`, `commons/sr_utils.py`, `commons/TInvPower.py` | Identical: scaling, initialization, sparse search, fixed-column regression, and linear algebra are unchanged. |
| `commons/identify_models.py` | Identical: interleaved model identification is unchanged. |
| `commons/weight.py` | Identical: polynomial weight construction is unchanged. |
| `commons/process_library_terms.py` | Adds grids and quadrature configuration and routes integration through the new module. Existing dataset/library methods remain. |
| `continuous/process_library_terms.py` | Its only change uses `self.diff_spacings(domain)` instead of `self.dxs` when differentiating. For unspecified grids this returns the same scalar spacings. |
| `discrete/process_library_terms.py` | Adds coordinate handling for particle/coarse-grained data. Neither audited convection workflow imports this dataset. |

Primary upstream references:
[dataset implementation](https://github.com/pjo0213/PySPIDER/blob/ccc892cdc84fb183c13531a7fa654132acbe1818/src/PySPIDER/commons/process_library_terms.py),
[integration](https://github.com/pjo0213/PySPIDER/blob/ccc892cdc84fb183c13531a7fa654132acbe1818/src/PySPIDER/commons/integration.py),
[regression](https://github.com/pjo0213/PySPIDER/blob/ccc892cdc84fb183c13531a7fa654132acbe1818/src/PySPIDER/commons/sparse_reg_bf.py).

### Main's numerical/API compatibility

Main constructs its shared, first-order tau/phi, and residual-search datasets
with keyword arguments, uniform `dxs`, and no `grids` or quadrature schemes.
The added dataclass fields therefore do not shift positional arguments in
these calls. `make_libraries`, `make_domains`, `make_weights`, `set_LT_scale`,
`make_library_matrices`, `IntegrationDomain`, `libs`, `terms`, `Q`, `col_weights`,
`xscale`, and `tscale` remain available. The optional residual dataset's copying
of domains, weights, and tensor-weight bases remains applicable to its uniform
grid; that code would need review if extended to nonuniform data.

For positive uniform spacings and the same domains:

1. Both versions use physical weight half-width `(n - 1) * dx / 2`.
2. Both use the same finite-difference implementation and spacing.
3. The fork defaults to trapezoidal integration on every axis. It does not
   automatically apply Chebyshev quadrature to main's uniform data.
4. Both trapezoidal paths use unit index spacing in the integral; neither
   multiplies the result by the physical sample spacings. Thus migration does
   not introduce a new physical volume factor into main's matrices.
5. The old implementation multiplies the complete separable weight array into
   the field before integration. The new one applies each axis's weight while
   reducing axes. These are algebraically equivalent for the audited fields,
   but floating-point operation order changes; bitwise-identical matrices and
   sparse-model choices are not guaranteed.

The fork's new integration path converts term arrays to floating point. Main's
fields and scalar term products are real; this audit does not extend the
compatibility claim to complex-valued datasets. Main's current time-step,
filter-support, and `T=dt*dom_time*10` choices are application code and are not
fixed by swapping packages. Existing singular regression failures are also not
fixed: the solver files are identical.

### Required single-package migration

1. Change all executable `SPIDER.*` imports to `PySPIDER.*`, including imports
   inside notebook helper functions and the interpolation preprocessor. Keep
   discovery and run archiving in notebook cells. Do not mix classes imported
   under two module namespaces in one process or hide this with an alias.
2. Use the same exact requirements pin for all entry points. Verify the actual
   imported source and revision, as Gauss does; package version `2.1.0` alone
   does not distinguish fork revisions. The fork requires Python >=3.12 and
   declares NumPy, SciPy >=1.6, z3-solver, findiff, and numba-kdtree dependencies.
3. Provide the NumPy compatibility alias before integrating in Copy2 when
   `np.trapezoid` is absent (`np.trapezoid = np.trapz`, conditional on absence),
   as Gauss already does. The fork does not declare a NumPy >=2 requirement,
   and its trapezoidal implementation calls `np.trapezoid`.
4. Update `run_filter_sweep.py::scratch_working_copy` to snapshot the active
   external package instead of `source/SPIDER`. Ensure workers import that
   same pinned snapshot and retain scratch-only cache/output behavior.
5. Update the notebook's source archive to record the actual imported package
   path, revision, source hashes and source copies. Merely renaming imports
   would leave the old archive code snapshotting an unused `SPIDER/` tree.
6. Update archive-reuse compatibility checks explicitly. They currently hash
   paths beginning `SPIDER/` against the working tree. Removing that tree makes
   these checks fail; leaving it while importing PySPIDER checks the wrong
   implementation. Do not bypass these checks. Default to rejecting reuse
   across package identities, while retaining old immutable archives for
   inspection/comparison. An explicit, separately validated migration can be
   considered later. Saved scientific results use JSON/NPZ here, rather than
   requiring both packages to unpickle their model objects.
7. After switching all entry points and provenance paths, remove the vendor
   tree and its sync utility. Keep only the pinned external dependency for new
   runs. Keep the original main revision accessible for legacy reproduction.

Neither fork removes main's multiprocessing-after-JAX issue: the process-pool
assembly code is unchanged. The Gauss launcher already chooses serial assembly;
main's notebook still requests parallel assembly. Treat safe process execution
as a separate PACE validation item, not a benefit supplied by the new fork.

### Merge preview and required resolutions

`git merge-tree --write-tree --name-only origin/main e431fcb` identified four
conflicts without changing either branch or the working tree:

- `.gitignore`: combine scratch/archive/cache/log exclusions.
- `.ipynb_checkpoints/COnvection-FILTER-Copy2-checkpoint.ipynb`: keep deleted.
- `COnvection-FILTER-Copy2.ipynb`: preserve main's current notebook, cell IDs,
  stage tags, closure/flux workflows, and archive checks. Relative to the merge
  base, this branch only changes its default input filename; do not replace
  main's notebook with the old tagged Copy2 notebook.
- `agents.md`: retain concise rules and main's requirement to keep discovery
  and complete immutable run archiving in notebook cells.

Additional issues will not appear as Git conflicts:

- The deleted interpolation notebook is still a mandatory file copied by
  `run_filter_sweep.py::scratch_working_copy`. A naive merge will raise a missing
  file error before the sweep starts. Change this and the notebook's archive
  source list to `prepare-numpy-files-interpolate.py`.
- The exported preprocessor writes `preprocessed_paper_data_interpolated.npz`;
  main's runner and Slurm defaults use `preprocessed_paper_data.npz`. Align the
  defaults or explicitly supply `DATA_PATH`/`--data-path` throughout.
- The README on this branch describes the older Copy2 notebook, including
  numbered cell selection. Those instructions must be replaced with main's
  tagged-cell runner and its `PACE_TWO_RUNS.md` workflow after the merge.
- Main-only sweep scripts and documentation survive the merge; their absence
  on this branch is not a deletion made here. Preserve them.
- Gauss's checked-in account-specific `SPIDER_CHECKOUT` assignment and literal
  `ndomains=32` override should be resolved so both pipelines can use the same
  configured installation and honor their documented controls.

The existing successful-run tag should remain on `e431fcb`; a merge or package
migration is a new revision and does not inherit numerical validation from
that tag.

### PACE acceptance before claiming numerical equivalence

Keep the legacy revision and candidate in separate processes/environments and
use the same NPZ, domains, weights, term order, seeds and regression settings.
Compare shared tau/phi matrices and, if supported, optional baseline/residual
matrices: shapes, term ordering, column scales, finite values, relative matrix
errors and singular-value structure. Also evaluate each known/archived equation
on both matrices. Report observed differences instead of assuming exact equality.

Then compare fresh closure fits and flux outputs, and rerun the native Gauss
diagnostic plus recovery. Check prescribed-only and discovery runs, snapshot
imports, fresh archive creation, and rejection of mismatched legacy reuse.
Fresh runs must never overwrite the successful baseline archives. No numerical
acceptance threshold or passing result is asserted by this audit.

Validation performed: exact-revision source comparison, notebook/source call
inspection, source-reference inspection, and Git merge preview. No local
scientific imports, matrix construction, regression, PACE submissions, package
migration, or branch merge were performed.
