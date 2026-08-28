# Changelog

All notable public changes to LEAF-EB are documented here.

## 2026.08.27

- Clarified that scenarios can be defined directly in LEAF-EB or derived from external studies; the scenario does not need to exist before using the framework.

- Replaced the single public example with six progressive, commented examples.
- Added synthetic daily and hourly datasets for self-contained learning cases.
- Added deterministic balance, Monte Carlo, BESS, methane, constant-output
  nuclear and load-following nuclear examples.
- Allowed the electricity-balance engine to run when commodity production is
  not configured, enabling basic and BESS-only scenarios.
- Updated public tests and CI to validate and execute all six examples.
- Added `docs/EXAMPLES.md` and aligned the public documentation with the
  example sequence.

## 2026.08.25

- Prepared the first general-public GitHub distribution.
- Added explicit Monte Carlo control through
  `simulation.monte_carlo.preserve_annual_targets`.
- Retained explicit technology uncertainty and demand-balance settings.
- Retained automatic anchor resolution with reproducible resolved settings.
- Updated commodity technology data and references.
- Included a self-contained deterministic example and public smoke tests.
