# Rayleigh-Benard / SPIDER experiments

The native Chebyshev-Gauss pipeline is the primary workflow. A successful PACE
run was reported on October 17, 2026; see the run guide for provenance and
diagnostics to retain with results.

For the updated quadrature fork and native Dedalus Chebyshev-Gauss recovery,
see [the PACE run guide](CHEBYSHEV_GAUSS_RUN.md). This uses the separately pinned
`PySPIDER` package; the vendored `SPIDER/` sync below does not update that package.

## Repository layout

- `SPIDER/` — vendored `src/PySPIDER` code from the upstream PySPIDER repository.
- `dedalus/` — Dedalus simulations and plotting scripts.
- `slurm/` — batch scripts; generated job output is ignored by Git.
- `dedalus/paper-test/` and `slurm/paper-test/` — the full-resolution workflow.

## Run the full workflow

From the repository root on the cluster:

```bash
sbatch slurm/paper-test/run_convection.sbatch
```

This runs condition setup, stabilization, and production. No prior `salloc` is
needed; `sbatch` requests the resources declared in the script. Monitor it with:

```bash
squeue -u "$USER"
```

The log is created in the submission directory as
`rb_convection-<job-id>-full.out`. Before running, check that the `dedalus3`
Conda environment and the `/storage/scratch1/2/ashelestov3/...` paths used by the
Python scripts are valid for your account.

Other batch scripts can also be submitted from the repository root or any of its
subdirectories. They locate the root before launching files from `dedalus/`.

## Analysis entry points

- `prepare-numpy-files-chebyshev-gauss.ipynb` prepares native Gauss data.
- `COnvection-FILTER-Chebyshev-Gauss.ipynb` runs native Gauss recovery.
- `COnvection-FILTER-Copy2.ipynb` retains the legacy uniform-grid analysis.
  Its preprocessor is now `prepare-numpy-files-interpolate.py`; run it on PACE
  from the repository root with `MPLBACKEND=Agg python prepare-numpy-files-interpolate.py`.
  It preserves the old notebook's account-specific paths and numerical code.

Obsolete notebooks, Lobatto requirements, notebook checkpoints, historical Slurm
logs and the proposal PDF were removed from this release. Their history remains
in Git. Simulation sources are retained as provenance for the production data.

## Install the native Gauss dependency

In the PACE environment used by the notebook:

```bash
python -m pip install --no-deps --force-reinstall -r requirements-chebyshev-gauss.txt
```

This installs the exact `pjo0213/PySPIDER:upstream-plus-quadrature` commit pinned
in the requirements file, not the latest branch head. It does not update a Git
branch or the vendored `SPIDER/`. Restart the notebook kernel after installation.
If `SPIDER_CHECKOUT` is set, the notebook uses that checkout instead of the pip
installation; see the run guide. The batch launcher does not install packages.

Keep the pin for reproducible tagged runs. Adopting a newer fork revision requires
updating the requirements pin and the notebook's expected revision together,
then validating on PACE.

## Update legacy Copy2's vendored SPIDER

`SPIDER/` contains only `src/PySPIDER` from
<https://github.com/sibirica/PySPIDER>. Sync the upstream `main` branch with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-pyspider.ps1
```

Review and commit the result. The upstream commit is recorded in
`.pyspider-revision` when the sync script runs.
This script is only for Copy2 and does not update native Gauss's dependency.
