# Agent instructions

- Commit each logical change.
- Programs run only on remote headless Georgia Tech PACE. Do not run scientific
  tests or pipelines locally; use static source, notebook JSON, and whitespace
  checks. State what was and was not validated.
- Both pipelines use the pinned external `PySPIDER`; leave it unmodified and
  do not reintroduce a vendored implementation. See `pyspider_runtime.py`.
- Preserve Copy2's tagged workflow cells. Keep model discovery and run archiving
  in notebook cells; save unique archives of candidates, errors, settings, data
  identity, source/package snapshots, and simulation parameters. Never overwrite
  prior archives or mislabel inferred provenance/training fits as validated.
- Keep only the Gauss preprocessing/recovery and Copy2 notebooks. Keep generated
  artifacts out of commits. See `README.md` and `PACE_TWO_RUNS.md` for workflows.
