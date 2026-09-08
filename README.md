# Rayleigh-Benard / SPIDER experiments

The notebooks in the repository root contain the main analysis workflow.

`COnvection-FILTER-Copy2.ipynb` now includes an **SGS stress model discovery**
section based on sections 2.2 and 3.1 of the draft in `papers/`. On PACE, run the
loading and filtering cells, the known momentum-balance diagnostic, and then the
discovery section. Its helper is `scripts/tau_discovery.py` (use the repository
root as the notebook working directory).

The searches fit the scalar trace and symmetric traceless parts of full `tau`,
then the residuals after the paper's second- and fourth-order gradient stresses.
Each stage saves symbolic terms, physical coefficients, removed columns, and
training/later-time validation errors to `tau-discovery-delta-<width>.json`.
The configuration cell controls library size, integration domains, worker count,
optional buoyancy predictors, and an optional evolution search for residual `R`.
Increase domain counts if the library is underdetermined. `max_terms` includes
the fixed target column. Inspect residuals and repeat over filter widths; the code
does not assume that the paper's freely decaying-flow closure fits convection.

This uses the notebook's filtered DNS grid and wall-normal support trimming, not
the paper's additional spectral projection onto an LES grid. The optional `R`
search is an offline inference step, not an implemented LES time integrator.
Local review is limited to notebook JSON, Python syntax, and API inspection;
numerical validation requires the cluster environment and data.

## Repository layout

- `SPIDER/` — vendored `src/PySPIDER` code from the upstream PySPIDER repository.
- `dedalus/` — Dedalus simulations and plotting scripts.
- `slurm/` — batch scripts and historical job output.
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

## Update SPIDER

`SPIDER/` contains only `src/PySPIDER` from
<https://github.com/sibirica/PySPIDER>. Sync the upstream `main` branch with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-pyspider.ps1
```

Review and commit the result. The upstream commit is recorded in
`.pyspider-revision` when the sync script runs.
