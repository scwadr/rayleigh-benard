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

Both kernels use the same pinned `PySPIDER` package. There is no vendored
implementation or vendor-sync step. CPU JAX suffices for the supplied CPU jobs. In an already provisioned environment, reinstall just the
fork using `python -m pip install --no-deps --force-reinstall -r requirements-pyspider.txt`,
then run `python -m pip check`. `--no-deps` assumes dependencies are installed.
Restart notebook kernels after changing packages.

The requirements file installs commit `ccc892cdc84fb183c13531a7fa654132acbe1818`
from `pjo0213/PySPIDER:upstream-plus-quadrature`, not the latest branch head.
`pyspider_runtime.py` checks the selected package revision and imported path.
The sweep runner snapshots that package into each scratch working directory;
new run archives store its source hashes and revision. Leave `SPIDER_CHECKOUT`
unset for the pip installation, or point it to a clean checkout of the same pin.
Do not set `SPIDER_RUNTIME_MANIFEST` yourself; it selects the runner's verified
scratch snapshot. Restart kernels before switching source installations.
`requirements-chebyshev-gauss.txt` remains a compatibility wrapper around the
shared requirements file. The external package remains unmodified.

This consolidation preserves main's current Copy2 scientific workflow.
Chebyshev's closure/flux functionality has **not** been brought up to Copy2's
level; that is separate future work. Sharing a package does not make Copy2
accept native Gauss data. Use the preprocessor appropriate to each notebook.

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

Both notebooks now use the shared package selector; no account-specific
checkout assignment needs removal. Gauss's scientific cells remain unchanged.
Its domain-setup cell still contains the session-specific line:

```python
ndomains = 32 #int(os.environ.get("SPIDER_TEST_NDOMAINS", "256"))
```

Before using environment-variable domain counts below, replace that line with
`ndomains = int(os.environ.get("SPIDER_TEST_NDOMAINS", "256"))`. Otherwise every
run uses 32 domains. This is configuration for your run, not a change to Copy2.

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

## 5. Interpolated uniform-grid pipeline (current Copy2)

Use main's tagged-cell runner, `run_filter_sweep.py`. It preserves general tau/phi
discovery, prescribed gradient/Hessian benchmarks, flux analysis, and immutable
archives. Do not use numbered cell selections from the older feature notebook.

### Configure paths and preprocess

| File | Setting | Value |
| --- | --- | --- |
| `prepare-numpy-files-interpolate.py` | `h5py.File(...)` | Absolute production snapshot path |
| Same preprocessor | `save_path` | Your interpolated NPZ path |
| Shell before submission | `DATA_PATH` | Exactly the same NPZ path |
| Copy2, interactive use only | `load_path` or `RUN_CONFIG['data_path']` | Exactly the same NPZ path |

The preprocessor and runner default to
`/storage/scratch1/2/ashelestov3/preprocessed_paper_data.npz`. The preprocessor
overwrites an existing NPZ; use a new path to preserve older data. It interpolates
onto `np.linspace(0, 1, 1536)`. Match the hard-coded grid dimensions/bounds in
preprocessing and Copy2 if your simulation differs. Diagnostics use time index
50, so use at least 51 samples or edit the plot indices. Preserve the existing
velocity-component swaps for the existing data convention.

The sweep intentionally restricts data, caches and output to its configured
scratch root. For another account, replace `/storage/scratch1/2/ashelestov3`
consistently in `run_filter_sweep.py`, `slurm/submit_filter_workflow.sh`,
`slurm/measure_li.sbatch`, `slurm/run_filter_sweep.sbatch`, and
`slurm/plot_filter_sweep.sbatch`, including the path-validation patterns and
default log paths. Merely exporting a path outside that root is rejected.
Also change the Slurm account/QOS directives for your allocation.

Save this preprocessing-only launcher as `run_preprocess.sbatch` in the
repository root, editing allocation and resource requests for your run:

```bash
#!/bin/bash
#SBATCH --job-name=rb_preprocess
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
source activate "${SPIDER_ENV:-SPIDER}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python prepare-numpy-files-interpolate.py
```

```bash
export SPIDER_ENV=SPIDER
export DATA_PATH="$RB_DATA/preprocessed/interpolated.npz"
# First edit save_path in the preprocessor to this absolute path.
sbatch run_preprocess.sbatch
```

Wait for successful completion before measurement. Reuse the NPZ by skipping
this step. Resource requests are estimates; full fields/products occupy memory.

### Measure, discover, and compare

```bash
bash slurm/submit_filter_workflow.sh measure
```

Read the printed submission directory's measurement log after completion. Set
`PLAN` to the actual completed `scale_plan.json` it reports:

```bash
PLAN=/absolute/path/printed/by/measurement/scale_plan.json
SWEEP_ARRAY=0-5%1 bash slurm/submit_filter_workflow.sh discover --scale-plan "$PLAN"
```

The usual plan has six widths; match the array range to the actual plan if
support limiting removes a duplicate. `discover` runs general tau/phi discovery
plus discovered and prescribed flux benchmarks. It excludes broad homogeneous
PDE recovery, legacy numerical-target experiments, and optional frozen-baseline
residual searches. Those remain available in the full notebook; see
[PACE_FILTER_SWEEP.md](PACE_FILTER_SWEEP.md) for its interactive stage order.
The existing parallel settings remain; this consolidation does not change the
scientific workflow or resolve its multiprocessing-after-JAX limitations.

To run prescribed benchmarks without discovery, submit separately:

```bash
SWEEP_ARRAY=0-5%1 bash slurm/submit_filter_workflow.sh prescribed --scale-plan "$PLAN"
```

Discovery already includes prescribed benchmarks, so this second submission is
optional. Each array task processes one width. The helper prints a unique
scratch submission directory containing logs and the job ID. It sets its own
`OUTPUT_ROOT`; edit the helper's `base` to change that parent location.
The array requests 64 CPUs, node memory, and eight hours per task by default.
Adjust those in `slurm/run_filter_sweep.sbatch` as needed.

Each run saves `execution.json`, `COMPLETE` or `FAILED`, source snapshots,
discovery candidates/matrices, and flux metrics. Automatic comparison requires
the whole array to succeed. For manual plotting/auditing:

```bash
bash slurm/submit_filter_workflow.sh plot \
  --compare-runs /absolute/path/to/sweep_JOB --scale-plan "$PLAN" --discovery-benchmarks
```

Add `--verify-only` for archive checks without plotting. The full directory map,
failure handling, and commands are in [PACE_TWO_RUNS.md](PACE_TWO_RUNS.md).

### Reuse after consolidation

`--reuse-runs` remains available for completed fits with the **same pinned
package source hashes**, compatible data, settings, and notebook definitions.
Legacy vendored-SPIDER fits lack that package identity and are rejected for
reuse. Keep them intact; omit `--reuse-runs` to create fresh fits under the new
package. Archive-only checks/plots of old runs remain available and do not
relabel them as new-package results. Do not mix package identities in one
cross-width comparison.

Copy2's filtering, time-scale conventions, term libraries, model searches, and
flux definitions are unchanged from main. Numerical equivalence after the
integration implementation change still needs PACE validation; source-level
compatibility alone does not establish identical fitted coefficients.

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
