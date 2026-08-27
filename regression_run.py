import h5py
import numpy as np
import gc
import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries, grids, pressure, finite_differences as fd
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import xarray 
import xarray as xr
import dataclasses
import time
from utils import save, load
from library import *
from continuous_process_library_terms import *
import sys

jax.config.update("jax_enable_x64", True)

log_file = "output_log2.txt"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

size = 1024
density = 1.
viscosity = 1e-3
#viscosity = 0
seed = 12412
inner_steps = 20
outer_steps = 100

max_velocity = 3.2
cfl_safety_factor = 0.1
Lx =  2*jnp.pi
Ly = Lx
# Define the physical dimensions of the simulation.
grid = cfd.grids.Grid((size, size), domain=((0, Lx), (0, Ly)))
Nx = Ny = size
dy = Lx/size
dx = Ly/size

grid = grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dt = 1e-6
# Define the step function with Kolmogorov forcing
smooth = True
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.NavierStokes2D(viscosity, grid, smooth=smooth), dt)

# Create the trajectory function
trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

# Initial velocity field and FFT of vorticity
v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, max_velocity, 4)
vorticity0 = cfd.finite_differences.curl_2d(v0).data
vorticity_hat0 = jnp.fft.rfftn(vorticity0)

# Run the simulation to get the trajectory of vorticity in spectral space
_, trajectory_hat = trajectory_fn(vorticity_hat0)

# Transform vorticity back to real space for each time step
vorticity_trajectory = jnp.fft.irfftn(trajectory_hat, axes=(1, 2))

u_trajectory = []
v_trajectory = []

velocity_solve = spectral.utils.vorticity_to_velocity(grid)

# Loop over each time step in the vorticity trajectory
for vorticity in vorticity_trajectory:
    # Transform vorticity to Fourier domain
    vorticity_hat = jnp.fft.rfftn(vorticity, axes=(0, 1))

    # Use velocity_solve to compute velocity components in Fourier domain
    uhat, vhat = velocity_solve(vorticity_hat)

    # Transform velocity components back to real space
    u_real = jnp.fft.irfftn(uhat, axes=(0, 1))
    v_real = jnp.fft.irfftn(vhat, axes=(0, 1))

    # Append the real-space velocity components to the trajectory lists
    u_trajectory.append(u_real)
    v_trajectory.append(v_real)

# Convert velocity trajectories to arrays for easier manipulation
u_trajectory = jnp.stack(u_trajectory)
v_trajectory = jnp.stack(v_trajectory)

# Create spatial coordinates
spatial_coord = jnp.linspace(0, 2 * jnp.pi, size, endpoint=False)  # Same for x and y
coords = {
    'time': dt * jnp.arange(outer_steps) * inner_steps,
    'x': spatial_coord,
    'y': spatial_coord,
}

# Save the real-space velocity trajectory in an xarray Dataset
ds = xr.Dataset(
    {
        'u': (('time', 'x', 'y'), u_trajectory),
        'v': (('time', 'x', 'y'), v_trajectory),
    },
    coords=coords,
)

V = ds['v'].values 
U = ds['u'].values
U = np.transpose(U, (1,2,0))
V = np.transpose(V, (1,2,0))
Nt = outer_steps
dt= dt*inner_steps
Lt = outer_steps*dt

del smooth, step_fn, trajectory_fn, v0, vorticity0, vorticity_hat0
del trajectory_hat, vorticity_trajectory, u_trajectory, v_trajectory
del velocity_solve, spatial_coord, coords


print(ds)
print(dt)
print(dt*viscosity/dx/dx)
u_has_nan = ds['u'].isnull().any().item()
v_has_nan = ds['v'].isnull().any().item()

print(f"Does 'u' contain NaN values? {u_has_nan}")
print(f"Does 'v' contain NaN values? {v_has_nan}")
print(cfd.equations.stable_time_step(max_velocity, cfl_safety_factor, viscosity, grid))

print(U.shape)
def pressure_poisson(U, V, dx, dy, density=1.0):
    nx, ny, nt = U.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.rfftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = np.inf 
    P = np.zeros((nx, ny, nt))
    for t in range(nt):
        u_FT = np.fft.rfftn(U[:, :, t])
        v_FT = np.fft.rfftn(V[:, :, t])
        i = 1j
        dxu = np.fft.irfftn(i * kx * u_FT, s=(nx, ny))
        dyu = np.fft.irfftn(i * ky * u_FT, s=(nx, ny))
        dxv = np.fft.irfftn(i * kx * v_FT, s=(nx, ny))
        dyv = np.fft.irfftn(i * ky * v_FT, s=(nx, ny))
        rhs = dxu**2 + 2 * dyu * dxv + dyv**2
        rhs_FT = np.fft.rfftn(rhs)
        pressure_FT = density * rhs_FT / k_squared
        pressure = np.fft.irfftn(pressure_FT, s=(nx, ny))
        P[:, :, t] = pressure
    return P

P = pressure_poisson(U, V, dx, dy)
print(ds)
del ds
del uhat, vhat, vorticity_hat, u_real, v_real

import numpy as np
from scipy.fft import fft2, ifft2

def fourier_filter(field, width, filterscale, kernel):
    """
    Filters a given field based on the specified mode and kernel.

    Parameters:
    - field: the field to base the filter on
    - width: controls the filter's boundary steepness
    - filterscale: determines the filtering scale
    - kernel: "tanh", "gaussian", "heaviside", etc.
    
    Returns:
    - filtered_field: the filtered field
    """
    # Generate the grid

    hey = np.linspace(0,Nx//2,(Nx//2)+1)
    hey2 = np.linspace(-((Nx//2)-1),-1,((Nx//2)-1))
    q = np.concatenate((hey,hey2))

    X = np.outer(q, np.ones_like(q))
    Y = np.outer(np.ones_like(q), q)
    R = np.sqrt(X**2 + Y**2)
    
    # Create the filter kernel
    if kernel == "tanh":
        G = -0.5 * (np.tanh(width * (R - filterscale)) - 1)
    elif kernel == "gaussian":
        G = np.exp(-((R - filterscale) ** 2) / (2 * width**2))
    elif kernel == "Itanh":
        G = -0.5 * (-np.tanh(width * (R - filterscale)) - 1)
    elif kernel == "Igaussian":
        G = 1 - np.exp(-((R - filterscale) ** 2) / (2 * width**2))
    elif kernel == "symheaviside":
        G = (np.abs(R) <= filterscale).astype(float)
    elif kernel == "Isymheaviside":
        G = (np.abs(R) >= filterscale).astype(float)
    elif kernel == "symtanh":
        G = -0.5 * (np.tanh(width * (R - filterscale)) + 1) - 0.5 * (np.tanh(-width * (R + filterscale)) - 1)
    elif kernel == "Isymtanh":
        G = -0.5 * (-np.tanh(width * (R - filterscale)) - 1) - 0.5 * (-np.tanh(-width * (R + filterscale)) - 1)
    elif kernel == "boxheaviside":
        n = Nx
        k = np.arange(n)
        k[k > n/2] = k[k > n/2] - n
    
        mask = np.abs(k) < filterscale
        G = np.outer(mask, mask)
    else:
        raise ValueError('ERROR: valid kernels are "tanh", "gaussian", "heaviside", etc.')

    # Apply the filter based on the mode
    filtered_field = np.zeros_like(field)
    if field.ndim == 4:  # Tensor field
        for i in range(field.shape[3]): # iterates over tensor components
            for j in range(field.shape[2]): # iterates over time
                fourier_field = np.fft.fft2(field[:, :, j, i])
                filtered_fourier_field = fourier_field * G
                filtered_field[:, :, j, i] = np.real(np.fft.ifft2(filtered_fourier_field))
    elif field.ndim == 3:  # Single 3D field
        for i in range(field.shape[2]):
            fourier_field = np.fft.fft2(field[:, :, i])
            filtered_fourier_field = fourier_field * G
            filtered_field[:, :, i] = np.real(np.fft.ifft2(filtered_fourier_field))
    else:
        raise ValueError('ERROR: field must have right dimensions')

    return filtered_field


def curl(U, V, dx, dy): # curl of vector fields
    nx, ny, nt = U.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.rfftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    output = np.zeros((nx, ny, nt))
    for t in range(nt):
        u_FT = np.fft.rfftn(U[:, :, t])
        v_FT = np.fft.rfftn(V[:, :, t])
        i = 1j
        dyu = np.fft.irfftn(i * ky * u_FT, s=(nx, ny))
        dxv = np.fft.irfftn(i * kx * v_FT, s=(nx, ny))
        rhs = dxv - dyu
        rhs_FT = np.fft.rfftn(rhs)
        curl = np.fft.irfftn(rhs_FT, s=(nx, ny))
        output[:, :, t] = curl
    return output


def DX(U, V, dx, dy): # derivative of vector fields
    nx, ny, nt = U.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.rfftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    output = np.zeros((nx, ny, nt, 2))
    for t in range(nt):
        u_FT = np.fft.rfftn(U[:, :, t])
        v_FT = np.fft.rfftn(V[:, :, t])
        i = 1j
        dxu = np.fft.irfftn(i * kx * u_FT, s=(nx, ny))
        dxv = np.fft.irfftn(i * kx * v_FT, s=(nx, ny)) 
        rhs_FT = np.fft.rfftn(dxu)
        rhs2_FT = np.fft.rfftn(dxv)
        output[:, :, t, 0] = np.fft.irfftn(rhs_FT, s=(nx, ny))
        output[:, :, t, 1] = np.fft.irfftn(rhs2_FT, s=(nx, ny))
    return output

def DY(U, V, dx, dy): # derivative of vector fields
    nx, ny, nt = U.shape
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.rfftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    output = np.zeros((nx, ny, nt, 2))
    for t in range(nt):
        u_FT = np.fft.rfftn(U[:, :, t])
        v_FT = np.fft.rfftn(V[:, :, t])
        i = 1j
        dyu = np.fft.irfftn(i * ky * u_FT, s=(nx, ny))
        dyv = np.fft.irfftn(i * ky * v_FT, s=(nx, ny)) 
        rhs_FT = np.fft.rfftn(dyu)
        rhs2_FT = np.fft.rfftn(dyv)
        output[:, :, t, 0] = np.fft.irfftn(rhs_FT, s=(nx, ny))
        output[:, :, t, 1] = np.fft.irfftn(rhs2_FT, s=(nx, ny))
    return output
    

## construct observables
filterscale = 32
filterkernel = "symtanh"
width = 10
u = np.stack((U, V), axis=3) # indexed as x,y,t,vector_component
u_bar = fourier_filter(u, width, filterscale, filterkernel) # MAKE SURE DATA DEALIASED ON MASK WITH CORRESPONDING TO FILTER, ie, circular or rectangular
p_bar = fourier_filter(P, width, filterscale, filterkernel)
u_prime = u - u_bar
print(u.shape)
print(p_bar.shape)

'''# dissipation rate
dxu = DX(u_prime[:,:,:,0],u_prime[:,:,:,1],dx,dy)
dyu = DY(u_prime[:,:,:,0],u_prime[:,:,:,1],dx,dy)
epsilon = np.square(dxu[:,:,:,0]) + np.square(dxu[:,:,:,1]) + np.square(dyu[:,:,:,0]) + np.square(dyu[:,:,:,1])
del dxu
del dyu
epsilon = fourier_filter(epsilon, width, filterscale, filterkernel)
print(epsilon.shape)

# reynolds stresses
R_xx = fourier_filter( np.multiply(u_prime[:,:,:,0],u_prime[:,:,:,0]) , width, filterscale, filterkernel)
R_xy = fourier_filter( np.multiply(u_prime[:,:,:,0],u_prime[:,:,:,1]) , width, filterscale, filterkernel)
R_yy = fourier_filter( np.multiply(u_prime[:,:,:,1],u_prime[:,:,:,1]) , width, filterscale, filterkernel)
del u_prime
a1 = np.stack((R_xx, R_xy), axis=3) # indexed as x,y,t,vector_component
a2 = np.stack((R_xy, R_yy), axis=3)
del R_xx
del R_xy
del R_yy
a = np.stack((a1, a2), axis=4)
del a1
del a2
print(a.shape)'''

# exact closure
corr_xx = fourier_filter( np.multiply(u[:,:,:,0],u[:,:,:,0]) , width, filterscale, filterkernel)
corr_xy = fourier_filter( np.multiply(u[:,:,:,0],u[:,:,:,1]) , width, filterscale, filterkernel)
corr_yy = fourier_filter( np.multiply(u[:,:,:,1],u[:,:,:,1]) , width, filterscale, filterkernel)
tau_xx = np.multiply(u_bar[:,:,:,0],u_bar[:,:,:,0]) - corr_xx # now here u is indexed as x,y,t instead of matlab's y,x,t
tau_xy = np.multiply(u_bar[:,:,:,0],u_bar[:,:,:,1]) - corr_xy
tau_yy = np.multiply(u_bar[:,:,:,1],u_bar[:,:,:,1]) - corr_yy
del corr_xx
del corr_xy
del corr_yy
print(f"tau_xx: Shape = {tau_xx.shape}, Type = {type(tau_xx)}")
print(f"tau_xy: Shape = {tau_xy.shape}, Type = {type(tau_xy)}")
print(f"tau_yy: Shape = {tau_yy.shape}, Type = {type(tau_yy)}")
b1 = np.stack((tau_xx, tau_xy), axis=3) # indexed as x,y,t,vector_component
b2 = np.stack((tau_xy, tau_yy), axis=3)
del tau_xx
del tau_xy
del tau_yy
b = -np.stack((b1, b2), axis=4)
del b1
del b2
print(b.shape)

from commons_utils import *
from commons_identify_models import *
import copy
beta=16
u = np.stack((U, V), axis=3) # indexed as x,y,t,vector_component

bobs = Observable(string='τ', rank=2, can_commute_indices = True, antisymmetric = False)
uobs = Observable(string='u', rank=1)
pobs = Observable(string='p', rank=0)

robs = Observable(string='R', rank=2, can_commute_indices = True, antisymmetric = False)
epsilonobs = Observable(string='epsilon', rank=0)

#observables = [uobs, pobs, bobs, robs, epsilonobs]
observables = [uobs, pobs, bobs]
#data_dict = {'p': p_bar, 'u': u_bar, 'τ': b, 'R': a, 'epsilon': epsilon}
data_dict = {'p': p_bar, 'u': u_bar, 'τ': b}
#data_dict = {'p': p_bar, 'u': u_bar}


# fix random seed
np.random.seed(1)
world_size = np.array(p_bar.shape)
pad = 0
# fix random seed
np.random.seed(1)
dxs = [dx, dy, dt]
# initial setup of dataset
#corr_L = 5
#corr_T = 5
srd = SRDataset(world_size=world_size, data_dict=data_dict, observables=observables, dxs=dxs, 
                irreps=SRDataset.all_rank2_irreps(), cache_primes=False)
                #irreps=(0, 1, 2))
# initialize libraries, domains, and weights
#srd.make_libraries(max_complexity=5)
srd.make_libraries(max_complexity=5, max_observables = 3)
dom_width = 32
dom_time = 30 #previously 20 (without interpolation)
srd.make_domains(ndomains=1000, domain_size=[dom_width, dom_width, dom_time], pad=pad)
#srd.make_doma]ins(ndomains=10, domain_size=[dom_width, dom_width, dom_time], pad=pad)
srd.make_weights(m=beta, qmax=0)
srd.set_LT_scale(L=Lx*dom_width/Nx, T=Lt*dom_time/Nt) # note that this line must go before make_library_matrices
srd.make_library_matrices(debug=False)

from commons_identify_models import *
import copy

libs = srd.libs

# Parameter ranges for different aspects of the regression process

for threshold_loop in [1000,100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]:
    print("-----------------------------------------------------------")
    print("threshold ", threshold_loop)
    libs = srd.libs
    
    reg_opts_list = []
    for irrep in srd.irreps:
        # for regression we now need to construct a Scaler, Initializer, ModelIterator, and Threshold
        scaler = Scaler(sub_inds=None, char_sizes=libs[irrep].col_weights, row_norms=None, unit_rows=True, train_fraction=0.75)
        #init = Initializer(method='combinatorial', start_k=2)
        #init = Initializer(method='combinatorial', start_k=9999)
        init = Initializer(method='power', start_k=20)
        #res = Residual(residual_type='fixed_column', anchor_col=0)
        res = Residual(residual_type='matrix_relative')
        
        iterator = ModelIterator(max_k=10, backward_forward=True, max_passes=15)
        thres = Threshold(threshold_type='jump', gamma=1.5, delta=1e-8, n_terms=None)
        #thres = Threshold(threshold_type='information', ic=AIC)
        
        opts = {'scaler': scaler, 'initializer': init, 'residual': res,
                'model_iterator': iterator, 'threshold': thres}
        opts['verbose'] = False
        opts['inhomog'] = None
        opts['inhomog_col'] = None
        reg_opts_list.append(opts)

    eqs, lambdas, reg_results, derived_eqs, excluded_terms = interleave_identify([libs[i] for i in srd.irreps], 
    reg_opts_list, threshold=threshold_loop, experimental=True)

from commons_identify_models import *
import copy

libs = srd.libs

# Parameter ranges for different aspects of the regression process
threshold_values = [1000,100,10,1,1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
initializer_methods = ['combinatorial', 'power']
residual_types = ['matrix_relative', 'dominant_balance']
threshold_types = ['jump', 'information']  # Added 'information' type

# Iterate through all combinations
for threshold_loop in threshold_values:
    for init_method in initializer_methods:
        for res_type in residual_types:
            for thres_type in threshold_types:
                
                print("-----------------------------------------------------------")
                print(f"Threshold: {threshold_loop}, Initializer: {init_method}, Residual: {res_type}, Threshold Type: {thres_type}")
                
                reg_opts_list = []
                for irrep in srd.irreps:
                    scaler = Scaler(sub_inds=None, char_sizes=libs[irrep].col_weights, row_norms=None, unit_rows=True, train_fraction=0.75)
                    init = Initializer(method=init_method, start_k=10)
                    
                    if res_type == "dominant_balance":
                        res = Residual(residual_type='dominant_balance')
                    else:
                        res = Residual(residual_type='matrix_relative')

                    iterator = ModelIterator(max_k=20, backward_forward=False, max_passes=30)
                    
                    if thres_type == "jump":
                        thres = Threshold(threshold_type='jump', gamma=1.5, delta=1e-8)
                    else:  # 'information'
                        thres = Threshold(threshold_type='information', ic=AIC)

                    opts = {
                        'scaler': scaler,
                        'initializer': init,
                        'residual': res,
                        'model_iterator': iterator,
                        'threshold': thres,
                        'verbose': False,
                        'inhomog': False,
                        'inhomog_col': None
                    }
                    reg_opts_list.append(opts)
                
                try:
                    eqs, lambdas, reg_results, derived_eqs, excluded_terms = interleave_identify(
                        [libs[i] for i in srd.irreps], reg_opts_list, threshold=threshold_loop, experimental=True
                    )
                except TypeError as e:
                    print(f"Encountered an error: {e}")


sys.stdout.close()


