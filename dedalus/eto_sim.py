import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = 4, 1
Nx, Nz = 4096, 1536
Rayleigh = 1e8
Prandtl = 1
dealias = 3/2
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1)
grad_b = d3.grad(b) + ez*lift(tau_b1)

# Analysis Operators
vorticity = -d3.div(d3.skew(u))

# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("b(z=0) = Lz")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0")

# Solver
timestepper = d3.RK443
solver = problem.build_solver(timestepper)

# --- 1. Load State ---
solver.load_state('/storage/scratch1/2/ashelestov3/check_4096_stabilized/check_4096_stabilized_s1.h5')

# --- 2. Robust Analysis Setup ---
# We calculate the INTEGRAL (scalar) directly to avoid "hypervolume" errors
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(d3.integ(u@u), name='total_kinetic')
flow.add_property(d3.integ(vorticity**2), name='total_enstrophy')
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# --- 3. Short Run Configuration ---
iterations_to_run = 50
solver.stop_iteration = solver.iteration + iterations_to_run
dt = 1e-4

# CFL (Updated to use compute_timestep)
CFL = d3.CFL(solver, initial_dt=dt, cadence=1, safety=0.2, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=0.01)
CFL.add_velocity(u)

# --- 4. Main Loop ---
print("\n" + "="*40)
print(f"CALCULATING EDDY TURNOVER TIME (Average over {iterations_to_run} steps)")
print("="*40 + "\n")

# Volume of the box
volume = Lx * Lz

try:
    while solver.proceed:
        # Fixed: compute_timestep instead of compute_dt
        timestep = CFL.compute_timestep() 
        solver.step(timestep)
        
        if solver.iteration % 10 == 0:
            # Get the integrated scalars (use max to retrieve the single scalar value)
            total_ke = flow.max('total_kinetic')
            total_ens = flow.max('total_enstrophy')
            
            # Manual RMS Calculation
            u_rms = np.sqrt(total_ke / volume)
            w_rms = np.sqrt(total_ens / volume)
            
            # 1. Large Eddy Turnover Time (Bulk Flow)
            ett_large = Lz / u_rms if u_rms > 0 else 0.0

            # 2. Microscale Turnover Time (Turbulence)
            ett_micro = 1.0 / w_rms if w_rms > 0 else 0.0
            
            re_val = flow.max('Re')

            print(f"Iter: {solver.iteration} | dt: {timestep:.2e} | Re: {re_val:.1f}")
            print(f"   >> Large Eddy Turnover Time (Lz/U_rms): {ett_large:.4e}")
            print(f"   >> Microscale Time (1/W_rms):           {ett_micro:.4e}")

except Exception as e:
    logger.error('Exception raised, triggering end of main loop.')
    raise e
finally:
    print("\n" + "="*40)
    print("DONE.")
    print("="*40)