# Rayleigh–Bénard recovery on PACE

Both pipelines start from the same Dedalus snapshots. Run simulations,
preprocessing, and recovery on PACE compute nodes. Submit from the repository
root, not from `slurm/`.

| Pipeline | Preprocessing | Recovery | Package |
| --- | --- | --- | --- |
| Native Gauss (uninterpolated input) | `prepare-numpy-files-chebyshev-gauss.ipynb` | `COnvection-FILTER-Chebyshev-Gauss.ipynb` | Installed, pinned `PySPIDER` |
| Interpolated uniform grid | `prepare-numpy-files-interpolate.py` | `COnvection-FILTER-Copy2.ipynb` | Same installed, pinned `PySPIDER` |

Gauss preprocessing preserves the original vertical samples; its filtering
stage evaluates convolutions on a new interior Gauss grid. Copy2 interpolates
raw fields onto a uniform vertical grid first. The notebooks also differ in
libraries, stress/pressure treatment, and diagnostics, so they are not a
controlled comparison of interpolation alone.

## 1. Get the repository and choose storage

```bash
git clone git@github.com:scwadr/rayleigh-benard.git
cd rayleigh-benard
module load anaconda3
# Replace this with your actual writable PACE scratch/project allocation.
export RB_DATA=/absolute/path/to/your/scratch/rb-run-001
mkdir -p "$RB_DATA/preprocessed" "$RB_DATA/archives"
```

Use a new run directory for each simulation. `RB_DATA` is a convenience for
these shell commands: simulation and Copy2 sources do **not** read it. Replace
their hard-coded paths below with absolute paths. Python strings containing
`$RB_DATA` do not expand automatically. Re-export variables in new terminals.

The supplied Slurm scripts use account `gts-rg119`, QOS `embers`, and constraint
`amd`. Change these to your allocation, including in the Copy2 example below.
Check memory, runtime, and core requests against your queue. `sbatch` allocates
resources; a preceding `salloc` is unnecessary.

## 2. Install environments and kernels

Keep Dedalus's MPI stack separate from analysis. Reuse working environments if
available. These are installation recipes, not a lockfile for the successful
run; only the Gauss PySPIDER revision is pinned here.

### Dedalus

For a fresh environment, upstream's full-stack Conda installation is:

```bash
conda create -n dedalus3 -c conda-forge dedalus
source activate dedalus3
conda env config vars set OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
conda deactivate
source activate dedalus3
```

Use the MPI launcher belonging to this environment. If your PACE installation
requires site MPI/FFTW libraries, use upstream's custom installation with those
paths instead. Both methods and threading settings are described in
[Installing Dedalus](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html).
The simulation launcher activates `dedalus3`; edit its `source activate` line
if you choose another name.

### Analysis: both pipelines

```bash
conda create -n SPIDER python=3.12 pip
source activate SPIDER
python -m pip install numpy scipy h5py matplotlib joblib tqdm z3-solver findiff \
  numba-kdtree jax jaxlib jax-cfd nbconvert nbformat ipykernel
python -m pip install --force-reinstall -r requirements-pyspider.txt
python -m pip check
python -m ipykernel install --user --name spider-gauss --display-name 'SPIDER Gauss'
python -m ipykernel install --user --name spider-copy2 --display-name 'SPIDER Copy2'
```

Both kernels use the same pinned PySPIDER installation. No vendored package
or sync script is required. CPU JAX suffices for the supplied CPU jobs. In an already provisioned environment, reinstall just the
fork using `python -m pip install --no-deps --force-reinstall -r requirements-pyspider.txt`,
then run `python -m pip check`. `--no-deps` assumes dependencies are installed.
Restart notebook kernels after changing packages.

The requirements file installs commit `ccc892cdc84fb183c13531a7fa654132acbe1818`
from `pjo0213/PySPIDER:upstream-plus-quadrature`, not the latest branch head.
`pyspider_runtime.py` selects and verifies the pinned installation for both
notebooks. Leave `SPIDER_CHECKOUT` unset for pip, or set it to a clean checkout of
the same commit. Restart kernels when changing source installations. The old
`requirements-chebyshev-gauss.txt` filename remains a wrapper around the shared
requirements file. The external package remains unmodified.

This is the final feature-branch workflow, cleaned up to one PySPIDER package.
It does not include main's newer closure-discovery/sweep/flux workflow. Existing
regression cells in the feature notebooks are preserved. Package consolidation
was checked statically; it is not a new successful PACE run.

## 3. Generate the common data

`slurm/paper-test/run_convection.sbatch` runs setup at 2048 × 768,
stabilization at 4096 × 1536, and production at 4096 × 1536. The domain is 4 × 1,
Rayleigh number is 1e8, and Prandtl number is 1.

Replace account-specific paths in these files. Keep the directory basenames
below so generated filenames match the next stage's input:

| File under `dedalus/paper-test/` | Setting | Path under your run directory |
| --- | --- | --- |
| `rayleigh_benard-setup.py` | checkpoint `add_file_handler(...)` | `checks_paper` |
| `rayleigh_benard-stabilize.py` | `input_data` | `checks_paper/checks_paper_s51.h5` |
| Same stabilization file | `output_dir` | `check_paper_stabilized` |
| `rayleigh_benard.py` | `solver.load_state(...)` | `check_paper_stabilized/check_paper_stabilized_s1.h5` |
| Same production file | snapshot `add_file_handler(...)` | `snaps_paper` |

For example, replace `/storage/scratch1/2/ashelestov3/checks_paper` with
`/absolute/path/to/your/scratch/rb-run-001/checks_paper`. Setup uses overwrite
mode: do not point it at a run you want to preserve. If you change setup duration
or output cadence, update the stabilization input to the actual checkpoint
produced rather than assuming `_s51.h5`.

```bash
sbatch slurm/paper-test/run_convection.sbatch
squeue -u "$USER"
# Replace JOBID with the ID printed by sbatch.
tail -f rb_convection-JOBID-full.out
```

The launcher runs all three stages sequentially, stopping on a failed command.
It requests 128 MPI tasks but launches each stage with 64; align these with
your intended allocation. Allow enough runtime for all three stages.
After completion:

```bash
export SPIDER_SOURCE="$RB_DATA/snaps_paper/snaps_paper_s1.h5"
```

If you already have production snapshots, skip simulation and point this at
an existing file. Both preprocessors read **one HDF5 file**, not a directory or
sequence of sets. They require tasks `velocity`, `pressure`, `buoyancy`, and
`vorticity`, plus the `sim_time` coordinate.

Both preprocessors explicitly name coordinate datasets
`x_hash_4846e26f3f50e48ca5da4b8e0970d850261c704a` and
`z_hash_050b2c5be05fb65046d37e4f71ee7ab1962cccdc`. For different coordinates,
inspect `list(h5file['scales'].keys())` with h5py on PACE and update the lookups.
Also match grid sizes, bounds, and physical parameters throughout preprocessing
and recovery. The Gauss preprocessor checks nodes on `z_interval=(0.0, 1.0)`.

## 4. Native Gauss (uninterpolated input)

### Required notebook configuration

Package selection now uses the shared runtime helper; no account-specific
checkout assignment needs removal. The scientific settings remain those of the
feature branch. In the Gauss domain-setup cell, replace
`ndomains = 32 #int(os.environ.get("SPIDER_TEST_NDOMAINS", "256"))`
with `ndomains = int(os.environ.get("SPIDER_TEST_NDOMAINS", "256"))` before using
environment-variable domain counts below. Otherwise every run uses 32 domains.

### Preprocess and inspect diagnostics

```bash
unset SPIDER_CHECKOUT
export SPIDER_CONDA_ENV=SPIDER
export SPIDER_SOURCE="$RB_DATA/snaps_paper/snaps_paper_s1.h5"
export SPIDER_DATA="$RB_DATA/preprocessed/gauss.npz"
export SPIDER_ARCHIVES="$RB_DATA/archives/gauss"
export SPIDER_FILTER_DX=50
SPIDER_PREPROCESS=1 SPIDER_DIAGNOSTIC_ONLY=1 SPIDER_TEST_NDOMAINS=32 \
  sbatch --export=ALL slurm/paper-test/run_gauss_recovery.sbatch
```

The launcher executes preprocessing followed by recovery diagnostics. The NPZ
writer refuses to overwrite `SPIDER_DATA`; choose a new filename to regenerate,
or skip preprocessing when reusing it. Create its parent directory first.
The archives directory is created automatically.

The launcher requests one process, four CPUs, 512 GB, and 12 hours. These are
starting estimates, not measured requirements. Matrix assembly is serial to
avoid forking after JAX initializes threads; leave `SPIDER_TEST_PARALLEL=0`.

### Recover equations from prepared data

After checking diagnostics, submit regression without preprocessing:

```bash
SPIDER_PREPROCESS=0 SPIDER_DIAGNOSTIC_ONLY=0 SPIDER_TEST_NDOMAINS=256 SPIDER_SEED=1 \
  sbatch --export=ALL slurm/paper-test/run_gauss_recovery.sbatch
```

`SPIDER_FILTER_DX` is the filter width in horizontal grid spacings;
`SPIDER_SEED` controls sampling. Match the notebook's Rayleigh and Prandtl values
to your simulation. Each job reloads data and builds its matrices.

Results:

- `rb_gauss-JOBID.out`: Slurm output in the submission directory.
- `executed-notebooks/JOBID/`: executed notebooks and their output.
- `$SPIDER_ARCHIVES/<timestamp-and-UUID>/`: source/environment records, matrices,
  terms, filter checks, prescribed-momentum residual, and regression candidates.
  Inspect `complete.json`, `known-momentum.json`, and `matrix-diagnostics.json`.
  A diagnostic-only completion is not a regression result.

See [CHEBYSHEV_GAUSS_RUN.md](CHEBYSHEV_GAUSS_RUN.md) for interpretation and retries.
The source-override edits above also apply when following that guide.

## 5. Interpolated uniform-grid pipeline (Copy2)

### Configure paths and matrix assembly

Copy2 does not read the Gauss environment variables. Edit these source settings:

| File | Setting | Value |
| --- | --- | --- |
| `prepare-numpy-files-interpolate.py` | `h5py.File(...)` input | Absolute production snapshot path |
| Same preprocessor | `save_path` near the end | Absolute path to your `preprocessed/interpolated.npz` |
| `COnvection-FILTER-Copy2.ipynb` | `load_path` in cell 2 | The same interpolated NPZ path |

The preprocessor interpolates onto `np.linspace(0, 1, 1536)`. Recovery also
hard-codes that grid and `size_x=4096`, `size_y=1536`; keep these consistent if
changing resolution or bounds. Preserve the existing velocity-component swaps
for the existing data convention. Plot diagnostics use time index 50: provide
at least 51 time samples or change those indices. The NPZ writer overwrites
existing files, so use a new filename for each dataset you want to retain.

For a first headless run, use serial matrix assembly. In the Copy2 cell containing
`remove_terms` and `srd.make_library_matrices`, replace
`num_processorz = num_cores - 4` with `num_processorz = 1`, and replace the
matrix-construction call with:

```python
srd.make_library_matrices(debug=False, parallel=False)
```

This avoids the current CPU-count-based pool exceeding a Slurm allocation and
avoids forking after JAX initialization. Copy2 does not honor
`SPIDER_TEST_PARALLEL`. Its domain count is calculated from library size in that
cell; `dom_width=64` and `dom_time=40` must fit the retained data.

### Preprocess and run the main recovery

Copy2 contains an exploratory sweep and later experiments using hard-coded
column indices. For a first run, execute cells 1–15, then cell 17 (beginning
`reg_opts_list = []`), skipping the sweep in cell 16 and later experiments.
These are notebook positions, not execution counters. In Jupyter on a PACE
compute node, select `SPIDER Copy2` and execute that sequence after running
`MPLBACKEND=Agg python prepare-numpy-files-interpolate.py` in the same allocation.

For a fully headless run, save this as `run_copy2.sbatch` in the repository root.
Edit the allocation directives first. It runs the same selection in a generated
notebook, leaving the original notebook intact:

```bash
#!/bin/bash
#SBATCH --job-name=rb_copy2
#SBATCH --account=gts-rg119
#SBATCH --qos=embers
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
module load anaconda3
source activate SPIDER
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export COPY2_OUTPUT="$PWD/executed-notebooks/$SLURM_JOB_ID"
mkdir -p "$COPY2_OUTPUT"
if [[ "${COPY2_PREPROCESS:-1}" == 1 ]]; then
    python prepare-numpy-files-interpolate.py
fi
python - <<'PY'
import os
from pathlib import Path
import nbformat
nb = nbformat.read('COnvection-FILTER-Copy2.ipynb', as_version=4)
assert 'reg_opts_list = []' in nb.cells[16].source
assert 'interleave_identify' in nb.cells[16].source
nb.cells = nb.cells[:15] + [nb.cells[16]]
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
nbformat.write(nb, Path(os.environ['COPY2_OUTPUT']) / 'Copy2-input.ipynb')
PY
python -m nbconvert --to notebook --execute "$COPY2_OUTPUT/Copy2-input.ipynb" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=spider-copy2 \
    --output=Copy2-executed --output-dir="$COPY2_OUTPUT"
```

Submit from the repository root:

```bash
sbatch run_copy2.sbatch
# Later, reuse the existing NPZ without overwriting it:
COPY2_PREPROCESS=0 sbatch --export=ALL run_copy2.sbatch
```

Adjust memory and runtime to data size: full fields and products are held in
memory. Results are `rb_copy2-JOBID.out` and
`executed-notebooks/JOBID/Copy2-executed.ipynb`, containing equations, residuals,
and plots. Copy2 does not create Gauss-style archives or save matrices
automatically. Preserve the executed notebook, source edits, input identity,
job log, and `python -m pip freeze` output together.

Copy2 retains historical time-step/time-scale calculations, stress/pressure
conventions, trimming, and term-removal choices. Running it reproduces that
workflow; it does not apply native Gauss's equation cleanup. Inspect diagnostics
before interpreting coefficients.

## 6. Monitor and preserve results

Monitor with `squeue -u "$USER"` and inspect job logs afterward. On failure, read
the traceback or Slurm resource error before resubmitting; partial notebook
output does not establish successful recovery. Keep simulation configuration
and the checkpoint chain with your snapshots. Record `git rev-parse HEAD` and
any source edits used for each run.

Logs, executed notebooks, and `run-archives/` are ignored by Git. Archive external
data separately according to your allocation's retention policy. A successful
PACE run was reported on October 17, 2026; the fresh-environment recipes above
have not been executed in the local development environment.
