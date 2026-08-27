"""
Plot 2D vorticity snapshots manually.

Usage:
    plot_vorticity.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames_vorticity]

"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
from docopt import docopt
from dedalus.tools import post
from dedalus.tools.parallel import Sync

def get_scale_key(file_obj, axis_name):
    """Automatically find the available scale key (e.g. '1.0' or '1.5')."""
    try:
        keys = list(file_obj[f'scales/{axis_name}'].keys())
        return keys[0]
    except KeyError:
        # Fallback if specific axis not found, useful for debugging
        return None

def main(filename, start, count, output):
    """Save plot of vorticity for given range of analysis writes."""
    
    with h5py.File(filename, mode='r') as file:
        
        # 1. Automatically find the correct scale factor (1.0 vs 1.5)
        x_key = get_scale_key(file, 'x')
        z_key = get_scale_key(file, 'z')
        
        if x_key is None or z_key is None:
            print(f"Skipping {filename}: Could not find 'x' or 'z' scales.")
            return

        # 2. Extract Grids
        x = file[f'scales/x/{x_key}'][:] 
        z = file[f'scales/z/{z_key}'][:] 
        
        # Create meshgrid
        # Transpose might be needed depending on Dedalus version/layout
        # Standard: (x, z) for pcolormesh usually requires (X, Z) matching dimensions
        X, Z = np.meshgrid(x, z)
        
        # 3. Access the Data
        # Check if 'vorticity' is in tasks
        if 'vorticity' not in file['tasks']:
            print(f"Skipping {filename}: 'vorticity' task not found.")
            return

        dset = file['tasks']['vorticity']
        sim_times = file['scales/sim_time']
        write_nums = file['scales/write_number']

        # 4. Loop through the requested frames
        for index in range(start, start+count):
            
            # Load data
            raw_data = dset[index]
            
            # Handle array shape to match meshgrid (Nz, Nx)
            # If data is (Nx, Nz), transpose it.
            if raw_data.shape == X.T.shape: 
                data_slice = raw_data.T
            else:
                data_slice = raw_data

            t_curr = sim_times[index]
            write_curr = write_nums[index]
            
            # --- PLOTTING ---
            fig = plt.figure(figsize=(12, 3), dpi=200) # Aspect ratio 4:1
            ax = fig.add_subplot(111)
            
            # Plot data
            # Determine robust limits (ignore outliers/artifacts)
            lim = np.max(np.abs(data_slice)) * 0.8
            if lim == 0: lim = 1.0 # Prevent crash on empty fields

            pcm = ax.pcolormesh(X, Z, data_slice, cmap='RdBu_r', 
                                vmin=-lim, vmax=lim, shading='gouraud')
            
            # Formatting
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_title(f'Vorticity at t = {t_curr:.3f}')
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            
            # Colorbar
            cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
            cbar.set_label('Vorticity')
            
            # Save
            savename = f'write_{write_curr:06d}.png'
            savepath = output.joinpath(savename)
            
            # Rank 0 prints progress
            if index == start:
                print(f"Processing: {savepath.name} ...")
                
            fig.savefig(str(savepath), bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    # docopt uses the docstring at the TOP of the file to parse arguments
    # If __doc__ is None, the script crashes.
    if __doc__ is None:
        print("Error: The script is missing its docstring. Please ensure the top of the file contains the usage instructions.")
        exit(1)

    args = docopt(__doc__)
    
    # Clean up output path
    out_str = args['--output']
    if out_str is None: out_str = './frames_vorticity'
    output_path = pathlib.Path(out_str).absolute()

    # Create output directory safely with MPI
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
    
    # Visit writes
    post.visit_writes(args['<files>'], main, output=output_path)