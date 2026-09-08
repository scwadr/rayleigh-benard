# RULES

1. Commit every change.

2. These programs run only on a remote headless Linux cluster (Georgia Tech PACE), so testing cannot be performed in this development environment because the required packaging is too complex to support here.

# TODO

- Cleanup remaining filtered-equation issues:
  - Restore full `tau` plus ordinary filtered pressure.
  - Fix `dt` using `np.diff(t)` and remove the obsolete `*10` time-scale factor.
  - Fix filter-support trimming and use periodic filtering along `x`.
  - Omit `w` during equation recovery.
  - Remove zero and duplicate library columns numerically after constructing `Q`.
  - Directly evaluate the known momentum combination before regression.

# EXPERIMENT LOG

## 2026-09-08: Lobatto implementation audit against the pinned fork

- Read the continuous dataset, shared matrix assembly, integration, quadrature,
  weight, and regression implementations at the pinned PySPIDER revision
  `91b9d6b3dee9d8c22118a9cd4a30132ea3fb5196`. PySPIDER remains unmodified.
- The Gauss-to-Lobatto DCT-II/DCT-I normalization and ascending-node ordering
  are consistent. Left the preprocessing notebook unchanged.
- The full mapped interior Lobatto grid, physical `grids`, fixed vertical
  quadrature nodes, and full-height recovery domains match the fork's API.
  Its weight derivatives use each domain's physical half-width per axis;
  `set_LT_scale` only sets characteristic regression column sizes. Thus the
  single `L` is a conditioning choice, not an incorrect vertical derivative
  scale. The omitted x/time trapezoidal spacing factors multiply every column
  by the same constant here and do not change a homogeneous balance.
- Left the periodic x filter and quadrature-weighted z filter unchanged.
  These approximate an interior translation-invariant convolution; discrete
  filtering/differentiation commutation and product resolution still need
  numerical assessment on PACE, not an assumption of exact conservation.
- Fixed recovery-library bookkeeping:
  - Retained `dt(b) * z_hat`: a constant vertical unit vector does not make
    this time-dependent buoyancy term zero or redundant.
  - Moved the direct known-momentum diagnostic before numerical pruning, so
    small/proportional columns cannot remove a required diagnostic term.
  - Compare zero-column norms after dividing by PySPIDER's characteristic
    sizes, matching regression scaling instead of comparing raw magnitudes
    across different fields and derivative orders. Validate matrix/scale
    finiteness, and use relative tolerance directly for unit-norm duplicates.
  - Use momentum viscosity `nu` rather than thermal diffusivity `kappa` in
    the diagnostic (numerically unchanged for the current Prandtl number 1),
    and print each known term's contribution norm.
- Validation was static source review, notebook JSON inspection, and git
  whitespace checks only. No local scientific tests or cluster runs were
  performed. These fixes do not establish the cause of the failed September 4
  recovery or demonstrate successful recovery; domain count, sampling geometry,
  filtering errors, and residuals remain PACE follow-up items.

## 2026-09-04: Chebyshev-Lobatto PySPIDER test

### Confirmed grid findings

- Dedalus `ChebyshevT` output uses Chebyshev-Gauss (Gauss-Jacobi with
  `alpha = beta = -1/2`) vertical nodes. These nodes exclude the physical
  endpoints.
- The quadrature-enabled PySPIDER fork's `clenshaw-curtis` path expects a
  Chebyshev-Lobatto grid, which includes both endpoints.
- PySPIDER is outside this project's jurisdiction and must remain unmodified.
  The test pins the unmodified fork revision
  `91b9d6b3dee9d8c22118a9cd4a30132ea3fb5196` from
  `pjo0213/PySPIDER:upstream-plus-quadrature`.

### Pipeline changes

- Added `prepare-numpy-files-chebyshev-lobatto.ipynb`:
  - Reads the exact production file
    `/storage/scratch1/2/ashelestov3/snaps_paper/snaps_paper_s1.h5`.
  - Verifies that the Dedalus coordinates match Chebyshev-Gauss nodes.
  - Uses a spectral DCT transform to evaluate the fields on an equal-sized
    Chebyshev-Lobatto grid; it does not create a uniform grid.
  - Writes
    `/storage/scratch1/2/ashelestov3/preprocessed_paper_data_lobatto.npz`.
- Added `COnvection-FILTER-Chebyshev-Lobatto.ipynb`:
  - Imports the external package as `PySPIDER`; it does not use or alter the
    repository's vendored `SPIDER/` directory.
  - Uses periodic Gaussian filtering in `x`.
  - Uses Clenshaw-Curtis quadrature weights for filtering in `z` and evaluates
    the filtered fields directly on a complete Lobatto grid over the
    filter-support-safe interior.
  - Passes physical coordinate arrays through `SRDataset.grids` and selects
    PySPIDER's existing `clenshaw-curtis` integration scheme for `z`.
  - Every recovery domain spans the complete retained vertical grid because
    PySPIDER stores one fixed node specification per dataset. The test domain
    shape is therefore `[64, 768, 40]`, rather than the old `[64, 64, 40]`.
  - Applies the filtered-equation cleanup items listed above: full `tau`,
    ordinary filtered pressure, corrected `dt` and time scale, no `w`,
    numerical zero/duplicate-column pruning, and direct evaluation of the
    known momentum combination before regression.
- Added `requirements-chebyshev-lobatto.txt` to pin the fork revision.
- PACE's installed NumPy does not provide the NumPy 2.x name
  `np.trapezoid`. The notebook exposes the equivalent `np.trapz` function as
  a compatibility alias without changing NumPy or PySPIDER.
- PySPIDER's process pool uses `fork`, which is unsafe after JAX has started
  worker threads. Matrix construction now defaults to serial execution;
  `SPIDER_TEST_PARALLEL=1` is an explicit opt-in only for a workflow that
  separates JAX filtering and PySPIDER recovery into safe processes.

### Test result

- The first PACE matrix-construction attempt failed before regression because
  the fork called unavailable `np.trapezoid`; the notebook compatibility alias
  addresses that failure.
- After the compatibility changes, this Lobatto test did **not** successfully
  recover the expected equations. No detailed recovered coefficients or
  residual table has yet been recorded, so no single cause is established.

### Important experimental differences and follow-up

- The default smoke test uses only 32 domains and each domain spans 768
  vertical points. This is not an apples-to-apples comparison with the prior
  local 64-point vertical windows and may provide too few independent rows for
  recovery.
- The recovery domains are strongly anisotropic: the reported horizontal
  characteristic length was `0.0615234375`, while the vertical domain spans
  most of the layer. PySPIDER currently accepts one spatial `L` scale, so this
  geometry/scaling mismatch is a potential confounder.
- The reported characteristic time for 40 samples was
  `0.0009375209999998649` after removing the obsolete factor of 10.
- Before interpreting the failed recovery, record the direct known-momentum
  residual and recovered models, compare against the uniform pipeline using
  matched domains and term sets, test more domains, and check the
  Gauss-to-Lobatto/filtering diagnostics (field error, divergence, and
  conservation).
