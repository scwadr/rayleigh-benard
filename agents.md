# RULES

1. Commit every change.

2. These programs run only on a remote headless Linux cluster (Georgia Tech PACE), so testing cannot be performed in this development environment because the required packaging is too complex to support here.

# TODO

- Cleanup remaining filtered-equation issues:
  - Restore full `tau` plus ordinary filtered pressure.
  - Fix `dt` using `np.diff(t)` and remove the obsolete `*10` time-scale factor.
  - Fix filter-support trimming and use periodic filtering along `x`.
  - Omit `w` during equation recovery.
  - Remove zero and duplicate library columns numerically after constructing `Q`.
  - Directly evaluate the known momentum combination before regression.