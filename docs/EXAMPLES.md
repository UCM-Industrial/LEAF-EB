# Public examples

The six public examples form a guided learning sequence. They use synthetic
historical data and require no external downloads.

## 01 - Basic deterministic balance

Run:

```bash
python Runner.py Example_01_basic
```

This is the minimum complete electricity-balance case. It defines demand, four
generation sources and a daily deterministic simulation. No storage or
commodity conversion is configured.

Use it to learn the `scenario`, `historical_data`, `projection`, `simulation`,
`demand`, `sources`, `commodities_input` and `output` blocks.

## 02 - Monte Carlo uncertainty

Run:

```bash
python Runner.py Example_02_monte_carlo
```

This example activates temporal variability and runs five Monte Carlo
simulations. The small sample count is for demonstration only. Research runs
normally require a sample-size assessment appropriate to the study.

Important fields:

- `simulations`: number of stochastic simulations;
- `seed`: random seed;
- `sources`: series to perturb;
- `preserve_annual_targets`: whether stochastic annual energy is forced back
  to the deterministic annual total;
- `technology_uncertainty`: independent switch for technology parameters.

## 03 - Battery storage

Run:

```bash
python Runner.py Example_03_bess
```

A 150-MW, 4-h BESS is added to the deterministic case. Surplus electricity
charges the battery and the battery discharges when additional generation is
needed. The dispatch-order lists contain only `BESS`.

## 04 - Methane chain

Run:

```bash
python Runner.py Example_04_commodities
```

The BESS remains first in the dispatch order. Surplus remaining after battery
charging can produce CH4, which is stored and later reconverted through a
CCGT. Technology parameters are read from `data/Database.yaml`.

The order is:

```text
surplus: BESS -> CH4 production
additional generation need: BESS -> CH4-to-power
```

## 05 - Constant-output nuclear

Run:

```bash
python Runner.py Example_05_nuclear_constant
```

The example adds one generic 300-MW SMR in 2025. It uses
`smr_300_calendar60`, operates as `must_run` and follows a fixed calendar
refueling schedule. In the current example, the first 30-day outage begins on
2027-01-26.

## 06 - Load-following nuclear

Run:

```bash
python Runner.py Example_06_nuclear_load_following
```

The deployment and non-nuclear assumptions are the same as Example 05. The
nuclear source instead uses `smr_300_burnup60` and `load_following`. Actual
operation accumulates equivalent full-power exposure, so the first refueling
outage occurs later. In the current example, it begins on 2027-07-13.

The example demonstrates why operating strategy and refueling basis should not
be treated as independent bookkeeping choices.

## Synthetic datasets

Examples 01-02 use `Inputs/data/example_history_daily.csv`. Examples 03-06 use
`Inputs/data/example_history_hourly.csv`. Both datasets are synthetic and
should not be interpreted as observations from a real electricity system.
