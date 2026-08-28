# Public examples

The public repository contains six progressive examples. Run any file with
`python Runner.py <name>`; the `.yml` extension is optional.

| File | Purpose | Historical data |
| --- | --- | --- |
| `Example_01_basic.yml` | Deterministic balance | Daily synthetic |
| `Example_02_monte_carlo.yml` | Temporal Monte Carlo | Daily synthetic |
| `Example_03_bess.yml` | BESS operation | Hourly synthetic |
| `Example_04_commodities.yml` | CH4 production/storage/reconversion | Hourly synthetic |
| `Example_05_nuclear_constant.yml` | Must-run nuclear + calendar refueling | Hourly synthetic |
| `Example_06_nuclear_load_following.yml` | Load-following + burnup refueling | Hourly synthetic |

The YAML files are intentionally commented. In YAML, text after `#` is a
comment and is ignored by LEAF-EB. Comments identify units, explain important
switches and indicate what each example is meant to demonstrate.

The datasets in `Inputs/data/` are synthetic. They exist only to make the
examples reproducible without external downloads.

## Recommended learning order

Run the examples in numerical order. Each file adds one main capability while
keeping earlier assumptions as similar as practical. Examples 05 and 06 use
the same 300-MW nuclear deployment so that operating and refueling differences
can be compared directly.

## Monte Carlo annual energy

`simulation.monte_carlo.preserve_annual_targets` controls whether stochastic
energy series are rescaled to their deterministic annual totals. The public
Monte Carlo example sets it to `false`, so sampled historical conditions can
change annual energy as well as within-year timing.
