# Rayleigh-Benard / SPIDER experiments

The notebooks in the repository root contain the main analysis workflow.

For the updated quadrature fork and native Dedalus Chebyshev-Gauss recovery,
see [the PACE run guide](CHEBYSHEV_GAUSS_RUN.md). This uses the separately pinned
`PySPIDER` package; the vendored `SPIDER/` sync below does not update that package.

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
