# Agent instructions

- Commit each logical change.
- Programs run only on remote headless Georgia Tech PACE. Do not run scientific
  tests or pipelines locally; use static source, notebook JSON, and whitespace
  checks. State what was and was not validated.
- Both pipelines use the pinned external `PySPIDER`; leave it unmodified and
  do not reintroduce a vendored implementation. See `pyspider_runtime.py`.
- This branch preserves the feature notebooks, without main's newer discovery
  workflow. Keep only Gauss preprocessing/recovery and Copy2 notebooks.
- Preserve previous run archives and source provenance. Keep generated artifacts
  out of commits. See `README.md` and `CHEBYSHEV_GAUSS_RUN.md` for workflows.
