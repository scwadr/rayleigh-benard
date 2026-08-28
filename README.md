# Rayleigh-Benard / SPIDER experiments

The notebooks in the repository root contain the main analysis workflow.

## Repository layout

- `SPIDER/` - vendored contents of `src/PySPIDER` from the upstream PySPIDER repository
- `dedalus/` - Dedalus simulations and Dedalus-based plotting scripts
- `slurm/` - batch scripts and historical job output
- `paper-test/` variants are grouped under the corresponding `dedalus/` and `slurm/` directories

## Running Slurm jobs

Jobs can be submitted from the repository root or any subdirectory within it, for example:

```bash
sbatch slurm/run_convection.sbatch
```

The batch scripts locate the repository root before invoking scripts from `dedalus/`.
Slurm writes each new job log to the directory from which the job is submitted.

## Updating SPIDER

`SPIDER/` contains only the contents of `src/PySPIDER` from
<https://github.com/sibirica/PySPIDER>. To replace it with the latest version from
the upstream `main` branch, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\sync-pyspider.ps1
```

Review and commit the resulting changes so this repository continues to record an
exact PySPIDER snapshot. The synced upstream commit is recorded in
`.pyspider-revision`.
