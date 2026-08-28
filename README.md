# Rayleigh-Benard workflow

## Run the full Dedalus pipeline on the cluster

The full batch workflow runs condition setup, stabilization, and production in
that order. From a login shell at the repository root, submit it with:

```bash
sbatch slurm/paper-test/run_convection.sbatch
```

You do not need to run `salloc` or otherwise allocate resources first. `sbatch`
submits a non-interactive job, and the `#SBATCH` directives in the script request
the node, tasks, time limit, account, QOS, and hardware constraint. Use `salloc`
only when you intentionally want an interactive debugging session.

The script can also be submitted while your shell is in a repository
subdirectory. It uses `SLURM_SUBMIT_DIR` to locate the repository root before
launching the files under `dedalus/paper-test`.

Check the queued or running job with:

```bash
squeue -u "$USER"
```

The job ID is printed by `sbatch`. Standard output is written in the directory
from which `sbatch` was invoked, using a name like
`rb_convection-<job-id>-full.out`.

The pipeline stops on the first failed phase. The final success message is
printed only after all three phases finish successfully.

Before submission, confirm that the `dedalus3` Conda environment exists and that
the hard-coded `/storage/scratch1/2/ashelestov3/...` checkpoint and snapshot
paths in the three Python scripts are appropriate for the account running the
job.
