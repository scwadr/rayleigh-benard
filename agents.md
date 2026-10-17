# Agent instructions

- Commit every change.
- Programs run only on the remote headless Georgia Tech PACE cluster. Do not
  run scientific tests or pipelines locally; use static source, notebook JSON,
  and Git whitespace checks. State what was and was not validated.
- Keep only the native Gauss preprocessing/recovery and Copy2 notebooks.
- Native Gauss imports the externally pinned `PySPIDER` fork; leave that
  external package unmodified. Copy2 imports the repository's `SPIDER/` code.
  Do not confuse the pip dependency with the legacy vendor-sync script.
- Consult `README.md` for both workflows and `CHEBYSHEV_GAUSS_RUN.md` for Gauss
  diagnostics. Keep run logs and generated artifacts out of commits.
