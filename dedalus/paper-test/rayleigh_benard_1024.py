"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 64 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
from dedalus.extras.plot_tools import plot_bot_2d
logger = logging.getLogger(__name__)

#import matplotlib
#matplotlib.use('Agg')
#matplotlib.rcParams['cache.directory'] = '/mpl'

# Parameters
Lx, Lz = 4, 1
Nx, Nz = 1024, 768
Rayleigh = 1e8
Prandtl = 1
dealias = 3/2
stop_sim_time = 52.1
timestepper = d3.RK443 #default RK222 !!!!
max_timestep = 0.01
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

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
solver.load_state('/storage/scratch1/2/ashelestov3/checks_1024/checks_1024_s51.h5')

# Analysis #/storage/scratch1/2/ashelestov3/snaps_2048_1e8
snapshots = solver.evaluator.add_file_handler('/storage/scratch1/2/ashelestov3/snaps_t', iter = 10, max_writes=2e9)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u, name='velocity')
snapshots.add_task(p, name='pressure')

print (b.domain == p.domain == solver.state[0].domain)


dt = 1e-3

solver.stop_sim_time = solver.sim_time + 0.5

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.1, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
min_timestep=1
try:
    logger.info('Starting main loop')
    while solver.proceed:
        
        timestep = dt
        #timestep = 0.0002859165868008745 / 2
        solver.step(timestep)
        logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
        if timestep < min_timestep:
            min_timestep = timestep
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
            print(CFL)
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
