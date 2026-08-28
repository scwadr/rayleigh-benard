# Rayleigh-Bénard / SPIDER experiments

The notebooks in the repository root contain the main analysis workflow.

## Repository layout

- `SPIDER/` — local SPIDER library checkout (`commons`, `continuous`, and `db`)
- `dedalus/` — Dedalus simulations and Dedalus-based plotting scripts
- `slurm/` — batch scripts; generated and historical job output is under `slurm/logs/`
- `paper-test/` variants have been grouped under the corresponding `dedalus/` and `slurm/` directories

## Running Slurm jobs

Submit jobs from the repository root, for example:

```bash
sbatch slurm/run_convection.sbatch
```

The batch scripts write output to `slurm/logs/` and invoke scripts from `dedalus/`.

## Connecting SPIDER to its own repository

`SPIDER/` is arranged at one directory boundary so it can become a Git submodule.
Once these files exist in the SPIDER repository, replace this tracked directory with
the submodule (substitute the actual URL):

```bash
git rm -r SPIDER
git submodule add <SPIDER-GITHUB-URL> SPIDER
git commit -m "Track SPIDER as a submodule"
```

After cloning this project, initialize it with:

```bash
git clone --recurse-submodules <THIS-REPOSITORY-URL>
```

To update SPIDER to the latest commit on its configured branch:

```bash
git submodule update --remote SPIDER
```

The parent repository records an exact SPIDER commit, so commit the updated
submodule pointer when you want collaborators to receive that version.
