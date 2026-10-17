from pyspider_runtime import activate_pyspider
activate_pyspider()

# Copy2 preprocessing, exported from prepare-numpy-files-interpolate.ipynb.

# Run on PACE from the repository root: MPLBACKEND=Agg python prepare-numpy-files-interpolate.py



import joblib
import h5py
import numpy as np
from PySPIDER.commons.utils import save, load
from PySPIDER.continuous.library import *
import gc
from PySPIDER.continuous.process_library_terms import *
from PySPIDER.commons.identify_models import *
import copy
import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries, grids, pressure, finite_differences as fd
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import numpy as np
#import xarray
#import xarray as xr
import dataclasses
import time
jax.config.update("jax_enable_x64", True)
import numpy as np
import os, joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from numpy.polynomial.chebyshev import chebvander

import contextlib
import joblib
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


f = h5py.File('/storage/scratch1/2/ashelestov3/snaps_paper/snaps_paper_s1.h5', 'r')
# '/storage/scratch1/2/ashelestov3/snaps_1024_1e8/snaps_1024_1e8_s1.h5'
# '/storage/scratch1/2/ashelestov3/snaps_2048_1e8/snaps_2048_1e8_s1.h5'
print(list(f['tasks'].keys()))
'''
f['tasks']['buoyancy'] = f['buoyancy'][901:]
f['tasks']['pressure'] = f['pressure'][901:]
f['tasks']['velocity'] = f['velocity'][901:]
f['tasks']['vorticity'] = f['vorticity'][901:]
'''
#print(f.type)

scales = f['scales']


f = f['tasks']
list(f.keys())



velocity = f['velocity']
vorticity = f['vorticity']
print(velocity.shape)
pressure_import = f['pressure']
buoyancy = f['buoyancy']
print(velocity)
print(velocity.shape)

del(f)

size_x = 4096
size_y = 1536
inner_steps = 1
outer_steps = 286

Lx =  4
Ly = 1

import numpy as np
import h5py
from scipy.interpolate import BarycentricInterpolator

# ... (Previous code for loading data remains the same) ...

print(-1)
P = np.array(pressure_import)
del(pressure_import)
B = np.array(buoyancy)
del(buoyancy)
vel = np.array(velocity)
del(velocity)
print("0")

# Extract components
u_trajectory = vel[:,0,:,:]
v_trajectory = vel[:,1,:,:]
del(vel)

P = P[:,:,:]
B = B[:,:,:]
print("1")

# Transpose to (x, y, t)
U = np.transpose(u_trajectory, (1,2,0))
del(u_trajectory)
V = np.transpose(v_trajectory, (1,2,0))
del(v_trajectory)
P = np.transpose(P, (1,2,0))
B = np.transpose(B, (1,2,0))
W = np.transpose(np.array(vorticity), (1,2,0))
del(vorticity)

print("**EDDY TURNOVER TIME**:")
print(1/np.mean(W))
print("2")

# Swap U and V as per original script logic
temp = U
U = V
V = temp
del(temp)

print(list(scales.keys()))

# Load coordinates
# Ensure you use the correct hash keys for your specific resolution (4096 in this case)
x = scales['x_hash_4846e26f3f50e48ca5da4b8e0970d850261c704a'][:]
y = scales['z_hash_050b2c5be05fb65046d37e4f71ee7ab1962cccdc'][:]
t = scales['sim_time'][:]

del(scales)

import numpy as np
import matplotlib.pyplot as plt

print("Calculating Continuity Residual (Divergence)...")

t_idx = 50
cmap = 'seismic'

# --- 1. Raw (Unfiltered) Divergence ---
# U, V, x, y are original loaded arrays
U_raw_t = U[:, :, t_idx]
V_raw_t = V[:, :, t_idx]

# FIX: Pass the full 1D coordinate arrays 'x' and 'y', not scalar step sizes!
du_dx_raw = np.gradient(V_raw_t, x, axis=0)
dv_dy_raw = np.gradient(U_raw_t, y, axis=1)

div_raw = du_dx_raw + dv_dy_raw

# --- 2. Diagnostics ---
rms_div_raw = np.sqrt(np.mean(np.square(div_raw)))

print(f"Raw Divergence - Mean: {np.mean(np.abs(div_raw)):.6e}, Max: {np.max(np.abs(div_raw)):.6e}")
print(f"Raw Divergence - RMS:  {rms_div_raw:.6e}")

# --- 3. Plotting ---
extent_raw = [x[0], x[-1], y[0], y[-1]]

fig, ax = plt.subplots(figsize=(12, 6))

v_limit_raw = np.max(np.abs(div_raw)) * 0.5

im = ax.imshow(
    div_raw.T,
    origin='lower',
    cmap=cmap,
    extent=extent_raw,
    aspect='auto',
    vmin=-v_limit_raw,
    vmax=v_limit_raw
)

ax.set_title(f"Unfiltered Continuity Residual ($\\nabla \\cdot \\mathbf{{u}}$) at t={t_idx}", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Divergence")

plt.tight_layout()
plt.show()

# ==========================================
# BARYCENTRIC INTERPOLATION LOGIC ADDED HERE
# ==========================================
print("Starting Barycentric Interpolation...")

# Define new uniform grid for y
# 1536 matches size_y defined earlier in your script
ys = np.linspace(0, 1, 1536)

# Interpolate all variables along axis 1 (y-axis)
# We overwrite the original variables to save memory and prepare for saving
print(U.shape)
print("Interpolating U...")
U = BarycentricInterpolator(y, U, axis=1)(ys)

print("Interpolating V...")
V = BarycentricInterpolator(y, V, axis=1)(ys)

print("Interpolating B...")
B = BarycentricInterpolator(y, B, axis=1)(ys)

print("Interpolating P...")
P = BarycentricInterpolator(y, P, axis=1)(ys)

print("Interpolating W...")
W = BarycentricInterpolator(y, W, axis=1)(ys)

# Update y to the new interpolated grid ys
y = ys

print(f"Interpolation complete. New shape of P: {P.shape}")

print("Saving preprocessed arrays. This may take a few minutes...")

# Define the path for your preprocessed file
save_path = '/storage/scratch1/2/ashelestov3/preprocessed_paper_data_interpolated.npz'

# Save the INTERPOLATED arrays
np.savez_compressed(
    save_path,
    U=U,
    V=V,
    P=P,
    B=B,
    W=W,
    x=x,
    y=y, # This is now the uniform grid 'ys'
    t=t
)

print(f"Done saving preprocessed data to {save_path}!")
