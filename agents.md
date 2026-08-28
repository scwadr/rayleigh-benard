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

# FILTERED RAYLEIGH-BENARD ROOT-CAUSE ANALYSIS (2026-08-28)

The momentum equation is recoverable, and the SPIDER library contains every required term. The root cause is a combination of inconsistent field conventions and regression conditioning—not a missing momentum term.

## Resolved primary root cause: buoyancy pointed in the wrong direction

The component flow is:

| Stage | Component 0 | Component 1 |
|---|---:|---:|
| Dedalus velocity | `u_x` | `u_z` |
| Preprocessing save | `U = u_z` | `V = u_x` |
| Filter notebook reload | `U = u_x` | `V = u_z` |
| `u_bar` | `u_x` | `u_z` |

The double swap is confusing but ultimately correct. However, the unit vector in `COnvection-FILTER-Copy2.ipynb` was initially:

```python
[1, 0]
```

That is the horizontal vector `e_x`. The buoyancy force in the Dedalus momentum equation is `b e_z`, so it must be:

```python
unit_vect = np.stack(
    [np.zeros_like(b_filt), np.ones_like(b_filt)],
    axis=3,
)
```

SPIDER was therefore asked to fit buoyancy forcing in the horizontal momentum equation while omitting it from the vertical equation. No single equivariant rank-1 momentum relation could correct that mistake. This was corrected on the feature branch and the filtered momentum equation is now recovered.

The successful recovered relation is, up to small regression artifacts:

```text
partial_beta tau_alpha_beta - b zbar_alpha + partial_alpha p
+ u_beta partial_beta u_alpha - 1e-4 partial_beta^2 u_alpha
+ partial_t u_alpha = 0.
```

The recovered coefficients were approximately `1`, `-1`, `1`, `1`, `-1e-4`, and `0.99`, respectively. A small additional term, `-8.67e-06 partial_t partial_alpha p`, was also selected and should be evaluated by contribution magnitude during cleanup.

This did not affect the principal setup in `COnvection-Stable.ipynb`, because that cell only includes `u` and `b` as observables; `z̄` is excluded.

## Second cause: `tau` was changed from the proposal's definition

`Proposal_Living_Draft_RBConvection.pdf` defines the full SGS stress:

```text
tau_ij = overline(u_i u_j) - ubar_i ubar_j
```

and uses:

```text
partial_t ubar_i + ubar_j partial_j ubar_i
+ partial_i pbar - nu partial_jj ubar_i
- bbar e_z,i + partial_j tau_ij = 0.
```

The notebook instead makes `tau` exactly traceless and moves its isotropic part into modified pressure in cell 4.

That transformation is mathematically valid, but it produces exact algebraic identities in this general-purpose library. The latest PACE run proves this:

```text
Identified model: T_αα = 0
train residual 3.72e-18
```

That is not discovered physics—it is true by construction. It explains why `tau` appears to generate relations with itself. The library also generates derivatives and products of this zero trace, giving sparse regression many easier, exact relations to select before momentum.

For initial equation recovery, use the proposal's simpler representation:

- Full `tau`, not deviatoric `tau`.
- Filtered `p_filt`, not `p_filt_modified`.

Once that recovers momentum, the deviatoric formulation can be reintroduced with trace-derived library terms explicitly removed.

The later residual diagnostic is also internally inconsistent: it uses `_p = p_filt` while passing deviatoric `tau`. That diagnostic therefore cannot close even when the filtered fields are otherwise correct.

## Third cause: filter-edge trimming is broken

The filter support radius is:

```text
R = alpha delta = 3(50 dx) = 0.146484375.
```

But the trimming expression is:

```python
alpha * (delta + 1)
```

This evaluates to about `3.146`, which leaves no valid coordinates. `trim_edges_jax` then silently falls back to returning the entire untrimmed array.

The later manual crop removes only 50 points per side, while the Gaussian support spans approximately:

- 150 points in `x`.
- 225 points in `z`.

Consequently, about 24% of the retained vertical domain is still within an edge-contaminated band. This matters especially for momentum because pressure and stress divergence involve spatial derivatives and cancellation.

There is direct evidence in the saved PACE diagnostics:

```text
RMS divergence raw:      6.91e-4
RMS divergence filtered: 1.61e-3
```

An interior spatial filter should commute with divergence closely; increasing the residual by roughly 2.3 times signals filter-boundary or interpolation contamination.

Additionally, `x` is periodic in Dedalus, but the filter uses edge padding in both directions. Horizontal filtering should use periodic wrapping. Vertical filtering should either use a physically justified wall treatment or discard the full support radius.

## Fourth cause: time scaling is wrong twice

The generating Dedalus script writes every iteration and sets:

```python
dt = 2.4039e-5
```

The notebook instead computes:

```python
dt = (t[-1] - t[0]) / Nt
```

For 101 snapshots, there are 100 intervals, so this must use `Nt - 1`, preferably:

```python
dt = np.mean(np.diff(t))
```

The current value is `2.380099e-5`, exactly 100/101 of the true value. The recovered heat equation independently confirms this: its time coefficient is `0.99`.

The notebook then compounds the conditioning problem with:

```python
T = dt * dom_time * 10
```

The factor of ten came from older output written every ten iterations, but `snaps_paper` uses `iter=1`. This does not change the raw matrix entries, but it incorrectly scales time-derivative columns during regression and support selection.

## Library assessment

The rank-1 library is structurally complete. It contains:

- `partial_t ubar_alpha`.
- `ubar_beta partial_beta ubar_alpha`.
- `partial_alpha pbar`.
- `partial_beta^2 ubar_alpha`.
- `bbar zbar_alpha`.
- `partial_beta T_alpha_beta`.

So the equation is not irrecoverable and `T` is not missing its divergence term.

However, the library is badly conditioned by:

- Exact `Tr(T) = 0` identities.
- Constant-vector derivative terms that are identically zero.
- Duplicate identities such as `p = p z_alpha z_alpha`.
- Treating 2D vorticity as an ordinary scalar rather than a pseudoscalar.
- Including all seven observables in one complexity-3 library.
- Incomplete string-based removal of trivial terms.

The latest output is dominated by exactly these tautologies, while the physical heat relation is the first nontrivial equation found.

## Interpolation assessment

The velocity swaps in `prepare-numpy-files-interpolate.ipynb` cancel correctly downstream, so they are not themselves the momentum failure.

Concerns remain:

- A global degree-1535 `BarycentricInterpolator` is used for every field.
- Pressure and stress divergence are more sensitive to interpolation errors than the heat equation.
- The saved notebook provenance is inconsistent: the interpolation cell has no execution count, while the save cell predates the displayed load/extraction cells.
- The generated NPZ therefore cannot be confidently reproduced from the saved execution history.

Interpolation is likely a secondary contributor, not the primary cause. Recovering continuity and filtered heat strongly suggests the basic interpolated fields and `phi` are usable.

## Recommended correction order

1. [completed] Change `unit_vect` to `[0, 1]`.
2. [cleanup TODO] Return to full `tau` plus ordinary `p_filt`.
3. [cleanup TODO] Fix `dt` using `np.diff(t)` and remove the `*10` time-scale factor.
4. [cleanup TODO] Fix the support trimming and use periodic filtering along `x`.
5. [cleanup TODO] Omit `w` during equation recovery.
6. [cleanup TODO] Remove zero and duplicate library columns numerically after constructing `Q`.
7. [cleanup TODO] Directly evaluate the known momentum combination before regression:

```text
Q_partial_t_u + Q_u_dot_grad_u + Q_grad_p - 1e-4 Q_laplacian_u
- Q_b_e_z + Q_div_tau.
```

If that residual is small, regression conditioning is the remaining issue. If it is large, compare full/deviatoric `tau`, both unit-vector orientations, and both stress signs to isolate the data inconsistency.

The proposal PDF was committed as `d2eb045`.
