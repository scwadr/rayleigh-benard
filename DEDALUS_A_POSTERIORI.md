# A posteriori benchmarks in Dedalus

This is an implementation design based on the current solver, archived models,
the repository paper (Sections 3.2-3.4), and Dedalus v3 documentation. It is not a
validated LES implementation. No scientific programs were run locally.

## What must evolve

The production solver is `dedalus/paper-test/rayleigh_benard.py`. It evolves
velocity `u` and buoyancy `b`, with pressure `p`, Fourier x, Chebyshev z,
stress-free impermeable walls, and fixed wall temperatures. It currently has no
SGS closure. Its `tau_u1`, `tau_u2`, etc. are numerical boundary-enforcement tau
fields, not the physical SGS stress tensor. Use distinct names such as
`sgs_stress` and `sgs_buoyancy_flux`.

There are three distinct experiments:

| Experiment | Fields used to evaluate closure | Feedback into equations | Interpretation |
| --- | --- | --- | --- |
| Diagnostic evaluation | Filtered DNS | None | A priori benchmarks |
| Diagnostic evaluation on an LES trajectory | That LES's evolving u and b | Only the selected driving closure | Comparison on one trajectory |
| Independent LES per model | Each model's own evolving u and b | Its own stress and scalar flux | A posteriori benchmarks |

Saving five closures on one velocity field does not produce five independent
LES predictions. Use one run per driving model for a posteriori comparison,
including an otherwise identical coarse run without a closure.

## Algebraic closures: separate output fields, not new prognostic variables

The nonlinear/gradient/Clark model, NGM4, and archived P1-P3 models are algebraic.
Construct their expressions from the evolving resolved fields. Pass the same
expressions to the PDE RHS and the analysis file handler. Do not freeze a NumPy
evaluation at initialization, or refresh a forcing field only after each complete
RK step: the nonlinear expressions must use the current RK-stage state.

For the prescribed models, with physical width Delta fixed for each run:

    tau_ij = Delta^2/12 * sum_k (d_k u_i)(d_k u_j)
             + Delta^4/288 * sum_kl (d_k d_l u_i)(d_k d_l u_j)
    phi_i  = Delta^2/12 * sum_k (d_k u_i)(d_k b)
             + Delta^4/288 * sum_kl (d_k d_l u_i)(d_k d_l b)

Omit the second term for the gradient model. In 2D, the Hessian contraction is
the xx contribution plus twice the xz contribution plus the zz contribution.
The scalar formula is a convection extension, not a model benchmark in the paper.
Use physical derivatives of u and b for these expressions, not the solver's
tau-lifted auxiliary `grad_u` and `grad_b` used for viscous/diffusive terms.

Component expressions provide an easily reviewed first implementation:

    stress_force_x = -(dx(tau_xx) + dz(tau_xz))
    stress_force_z = -(dx(tau_xz) + dz(tau_zz))
    scalar_force   = -(dx(phi_x) + dz(phi_z))
    Pi = -(tau_xx*dx(u_x) + tau_xz*(dz(u_x)+dx(u_z)) + tau_zz*dz(u_z))
    Pi_phi = -(phi_x*dx(b) + phi_z*dz(b))

Here `dx`/`dz` denote Dedalus differentiation operators, not grid spacings.
The velocity components must follow the Dedalus (x,z) convention; explicitly
map the archive's swapped/preprocessed component convention before porting fits.
Tensor/VectorField storage is available, but is unnecessary simply to save an
operator expression. Schematic integration into the current solver:

```python
# Define sgs_stress and sgs_buoyancy_flux as symbolic expressions first.
problem.add_equation(
    "dt(b) - kappa*div(grad_b) + lift(tau_b2) = "
    "-u@grad(b) - div(sgs_buoyancy_flux)")
problem.add_equation(
    "dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = "
    "-u@grad(u) - div(sgs_stress)")
# After building the solver, with a new scratch output directory:
snapshots.add_task(sgs_stress, name='sgs_stress')
snapshots.add_task(sgs_buoyancy_flux, name='sgs_buoyancy_flux')
snapshots.add_task(Pi, name='energy_flux')
snapshots.add_task(Pi_phi, name='scalar_variance_flux')
```

This is an integration sketch, not a runnable patch. Do not add algebraic stress
to the IVP unknown list without its defining equation. No additional initial
condition is required when it remains an expression of u and b.

## Exporting P1-P3 faithfully

Use the selected IDs in `rundata/metrics.csv` and the corresponding
`discovery_bundle.closure_models` in `sources/width_NNN/source_flux.json`.
Use `rhs_coefficients`, whose signs already solve for the target, rather than
copying signs from the printed homogeneous equation. Only nonzero terms are
needed. Build an explicit supported-term translator; do not eval printed strings.
Reject unknown terms and verify tensor index contractions and projections.

P counts predictors excluding the forced target. For these archives, P1-P3
correspond to requested_terms 2-4. Stress consists of separately fitted trace q
and symmetric trace-free tensor S; reconstruct `tau = S + q*Identity/2` in 2D.
Apply the STF projection to each rank-two expression as the archive evaluator
does, rather than interpreting its printed indices as a full unprojected stress.

At width_000, P2 includes a buoyancy-gradient product in both the trace and STF
stress fits. P3 adds velocity/second-derivative products. Scalar fits also include
terms beyond the gradient contraction. These are not automatically NGM4 or NGMR.
Support and coefficients can differ at other widths. Freeze the exact selected
model at one archived Delta for the first test; do not continuously refit it or
assume a universal Delta scaling for every fitted coefficient.

The isotropic stress can be absorbed into pressure for an incompressible flow,
but retaining full stress initially makes archive comparisons easier. If it is
absorbed later, record the pressure transformation; pressure-dependent fitted
terms must not silently change meaning.

## NGMR really does require an additional evolving field

The paper's online model is equations (3.15a-c), not merely the unregularized
residual equation (3.7). It evolves the symmetric trace-free Reynolds stress
R, uses the algebraic k in (3.12), and introduces regularizations (3.13)-(3.14).
In 2D, represent R with two independent scalar fields, Rxx and Rxz, with
Rzz = -Rxx. Include these fields in the IVP state and checkpoints, with their
evolution equations, initial conditions, and boundary treatment.

The paper's periodic-flow equations do not specify the missing convection-wall
conditions. Its sixth-order regularizations also require a deliberate bounded-
domain formulation and corresponding boundary conditions/tau treatment. Copying
the periodic model into this Fourier-Chebyshev solver is not sufficient. NGMR
also supplies no buoyancy SGS closure; that remains a separate convection model.

## Main numerical issues before calling this an LES benchmark

1. **Filtering and walls.** Existing fits/flux tests use interpolated fields,
   edge-padded Gaussian filtering, and an interior crop. A full-domain solver
   cannot crop away the walls. Specify a boundary-compatible filter, reference
   fields, and wall SGS flux/traction treatment; examine filter/derivative
   commutation errors. Any wall damping or closure modification changes the model
   and must be recorded and checked separately.
2. **Coarse representation.** Choose Nx, Nz independently of Delta and document
   the projection. Chebyshev spacing is nonuniform, so Delta/dx alone does not
   set vertical resolution. Initialize every model from the same filtered,
   projected DNS state satisfying incompressibility and boundary conditions.
   Do not assume loading the existing fine checkpoint performs this operation.
3. **Derivative consistency.** Archived metrics use finite differences on the
   interpolated grid; Dedalus uses spectral derivatives. On PACE, compare the
   ported expressions against archived evaluations on a common physical state
   and interior, then assess derivative convergence rather than expect identity.
4. **Stability.** The production script computes a CFL object but actually uses
   a fixed timestep and stops after 100 steps. An advective CFL criterion alone
   does not cover explicit SGS/high-derivative restrictions. Check timestep and
   resolution convergence, budgets, and high-mode energy before long runs.
   Do not silently clip backscatter or add dissipation to make a model survive.
5. **Dealiasing.** The existing 3/2 setting is appropriate for quadratic terms.
   Review the polynomial degree of each added expression; flux diagnostics are
   cubic even when the closure itself is quadratic. Evaluate diagnostic products
   with sufficient padding or verify equivalent accurate contractions.

## Proposed PACE validation sequence

First validate the algebraic gradient and NGM4 operators on fixed fields,
including mixed Hessians, sign, component order, and stress symmetry. Then port
the exact P1-P3 archive expressions, including trace/STF reconstruction. These
operator checks are prerequisites to interpreting a posteriori failures.

Start with one filter width and separate coarse runs: no closure, gradient,
NGM4, P1, P2, P3. For each, prescribe both the stress and scalar-flux choice;
stress-only, scalar-only, and coupled tests can distinguish their effects.
Use identical initial states, physical parameters, forcing and wall conditions.
Preserve independent checkpoints and manifests containing model IDs, physical
coefficients, source hashes, Delta, grid, timestep controls, and modifications.

Track kinetic energy, enstrophy, energy spectra (horizontal spectra and vertical
profiles for this geometry), mean energy flux, buoyancy flux, wall heat transport,
divergence error, high-mode content, and energy/scalar budget residuals. Heat
transport must include the modeled vertical scalar flux as well as resolved
advection and molecular diffusion. Save fields less frequently than scalar
diagnostics to control I/O.

Compare short-time fields to synchronized filtered DNS on the same resolved
space. For longer chaotic trajectories compare statistics and spectra over
matched intervals, with sampling uncertainty, rather than require pointwise
agreement indefinitely. Include closure transport and wall terms in the energy
budget: volume SGS transfer alone is not the entire bounded-domain balance.

The existing 101-snapshot burst is suitable for initial operator/short-time
checks; it does not establish adequate duration for long-time statistical
benchmarks. Freeze the measured initial/reference ell_i when labeling a fixed-
Delta run, or report a time-dependent ell_i explicitly without changing Delta.

## Sources and validation status

- Repository paper: Sections 3.2-3.4, especially (3.7), (3.12)-(3.15).
- [Dedalus fields and operators](https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_2.html)
- [Dedalus problems and solvers](https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html)
- [Dedalus analysis tasks](https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_4.html)
- [Dedalus dealiasing](https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_1.html)

Checked solver source, archived JSON model definitions, notebook projection
logic, paper text, and official documentation. No Dedalus import, operator
evaluation, scientific test, or PACE simulation was executed for this design.
