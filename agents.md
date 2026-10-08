# RULES

1. Make a separate commit for every logical part of an implementation, rather than one commit per prompt.

2. These programs run only on a remote headless Linux cluster (Georgia Tech PACE), so testing cannot be performed in this development environment because the required packaging is too complex to support here.

3. Keep model-discovery implementation and run archiving in notebook cells. For thesis runs, save a unique archive containing all candidate equations and coefficients, errors, baseline choice, filtering/library/regression/integration settings, data identity, source revision and snapshots, and relevant Dedalus simulation parameters. Preserve previous archives; distinguish inferred source configuration from confirmed run provenance, and training fits from validated or thesis-selected models. Do not change unrelated existing cells.

# TODO

- Cleanup remaining filtered-equation issues:
  - Restore full `tau` plus ordinary filtered pressure.
  - Fix `dt` using `np.diff(t)` and remove the obsolete `*10` time-scale factor.
  - Fix filter-support trimming and use periodic filtering along `x`.
  - Omit `w` during equation recovery.
  - Remove zero and duplicate library columns numerically after constructing `Q`.
  - Directly evaluate the known momentum combination before regression.
