# SPIDER compatibility and merge audit

Current branch scope: the final feature-branch notebooks now share the pinned
PySPIDER dependency, without importing main's newer model-discovery workflow.
The tag `successful-native-chebyshev-run` marks that cleaned feature state, not
the merged main branch. The audit below records the earlier comparison and
main migration requirements; those main-specific workflows are not on this branch.
The originally tagged successful run remains identifiable as commit `e431fcb`.
No new PACE run was performed for dependency consolidation.

Audited 2026-10-17. This is a static source audit, not a PACE execution result.

## Decision

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

## Exact sources examined

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

## Main's numerical/API compatibility

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

## Required single-package migration

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

## Merge preview and required resolutions

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

## PACE acceptance before claiming numerical equivalence

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
