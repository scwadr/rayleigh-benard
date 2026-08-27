"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
Upsampling script: 2048x768 -> 4096x1536
"""

import os
import pathlib
# Speeds up startup significantly by skipping exhaustive FFT tests
os.environ['DEDALUS_FFTW_PLAN_RIGOR'] = 'ESTIMATE'

import numpy as np
import dedalus.public as d3
import logging
import h5py

logger = logging.getLogger(__name__)

# --- Configuration ---
input_data = '/storage/scratch1/2/ashelestov3/checks_paper/checks_paper_s51.h5'
output_dir = '/storage/scratch1/2/ashelestov3/check_paper_stabilized'

# --- Parameters ---
Lx, Lz = 4, 1
Nx, Nz = 4096, 1536
scale_ratio = 0.5  # 2048 / 4096

Rayleigh = 1e8
Prandtl = 1
dealias = 3/2
stop_sim_time = 5e-2
timestepper = d3.RK443 
max_timestep = 1e-4
dtype = np.float64

# --- Bases ---
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# --- Fields ---
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))

# Tau terms (New resolution means new tau terms, we do not load these)
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# --- Substitutions ---
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

# ADDED: Extract components for stress-free boundaries
u_x = u @ ex
u_z = u @ ez
dzu_x = d3.Differentiate(u_x, coords['z'])

lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

# --- LOADING DATA ---
logger.info(f"Loading data from {input_data}...")

# 1. Get the domain object (needed for slicing)
dom = u.domain

with h5py.File(input_data, 'r') as f:
    
    # --- Velocity (Vector Field) ---
    u.change_scales(scale_ratio)
    local_slices = u.dist.grid_layout.slices(domain=dom, scales=scale_ratio)
    
    # FIX: Use integer -1 (not slice(-1)) to get the last timestep
    u_full_slice = (-1, slice(None)) + local_slices
    
    u['g'] = f['tasks']['u'][u_full_slice]
    u.change_scales(1.0)
    
    # --- Buoyancy (Scalar Field) ---
    b.change_scales(scale_ratio)
    local_slices = b.dist.grid_layout.slices(domain=dom, scales=scale_ratio)
    
    # FIX: Use integer -1
    b_full_slice = (-1,) + local_slices
    
    b['g'] = f['tasks']['b'][b_full_slice]
    b.change_scales(1.0)
    
    # --- Pressure (Scalar Field) ---
    p.change_scales(scale_ratio)
    local_slices = p.dist.grid_layout.slices(domain=dom, scales=scale_ratio)
    
    # FIX: Use integer -1
    p_full_slice = (-1,) + local_slices
    
    p['g'] = f['tasks']['p'][p_full_slice]
    p.change_scales(1.0)
    
logger.info("Data loaded and spectrally upsampled.")
# --- Problem Definition ---
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")

# Buoyancy (Temperature) boundaries
problem.add_equation("b(z=0) = Lz")
problem.add_equation("b(z=Lz) = 0")

# Impenetrability (vertical velocity is zero at the walls)
problem.add_equation("u_z(z=0) = 0")
problem.add_equation("u_z(z=Lz) = 0")

# Stress-free / Slip (vertical derivative of horizontal velocity is zero)
problem.add_equation("dzu_x(z=0) = 0")
problem.add_equation("dzu_x(z=Lz) = 0")

problem.add_equation("integ(p) = 0") # Pressure gauge

# --- Solver ---
logger.info("Building solver... (this may take a while)")
solver = problem.build_solver(timestepper)
logger.info("Solver built. Starting simulation.")
solver.stop_sim_time = stop_sim_time

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

solver.sim_time = 0.0
solver.iteration = 0

# Ensure limits are what you expect
solver.stop_sim_time = 5e-2
solver.stop_iteration = np.inf  # Make sure no iteration limit exists
solver.stop_wall_time = np.inf

# DEBUG: Print these values so we can see them in the log
logger.info(f"DEBUG: Start Time: {solver.sim_time}")
logger.info(f"DEBUG: Stop Time: {solver.stop_sim_time}")
logger.info(f"DEBUG: Stop Iteration: {solver.stop_iteration}")

# --- Main Loop ---
startup_iter = 10
min_timestep = 1

try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
            
except Exception as e:
    logger.error(f'Exception raised: {e}')
    raise

finally:
    # Ensure directory exists
    import pathlib
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create handler for the FINAL state
    checkpoints = solver.evaluator.add_file_handler(output_dir, max_writes=1)
    checkpoints.add_tasks(solver.state)
    
    # Force the write immediately
    solver.evaluator.evaluate_handlers([checkpoints], sim_time=solver.sim_time, iteration=solver.iteration)
    
    solver.log_stats()
    logger.info(f"Stabilization complete. Restart file saved to {output_dir}")