# LEAF-EB to ANICCA annual interface

Use `scripts/leaf_to_anicca.py` after a nuclear LEAF run has completed.

The generated workbook contains one worksheet per nuclear fleet and exactly
six columns:

`Year | Installed_Power | Burnup | Load_Factor | Reactors_In | Reactors_Out`

## Fixed burnup

When `fuel_cycle.target_burnup` is present, the converter passes that target
directly to ANICCA for every active year. The annual `Load_Factor` carries
changes in nuclear generation while refueling dates remain internal LEAF-EB
results.

## Fixed refueling calendar

A calendar-driven LEAF case produces physical discharge burnup only when a
batch is actually discharged. ANICCA, however, requires a burnup value in
every active annual time step. The converter therefore uses an annual-
equivalent burnup:

`BU_eq,y = LF_y * P_th * N_batches * T_calendar / (M_core * 1000)`

where `T_calendar` is the start-to-start interval between refueling outages:
the LEAF operating cycle, converted with LEAF's calendar rule, plus the
refueling outage duration.

This mapping makes the annual-equivalent fuel use consistent with a fixed
reload mass and fixed refueling calendar while retaining LEAF's actual annual
load factor. It must not be interpreted as the physical burnup of a batch
discharged in that specific calendar year. Physical batch burnups remain in
LEAF's nuclear fuel-event outputs.

For `smr_300_calendar60`, LEAF rounds the operating cycle to 755 days and uses
a 30-day outage. With 967.742 MWth, 36.523 tHM, and three batches, the
coefficient is about 62.40 GWd/tHM per unit load factor. At the corresponding
full-power calendar-average load factor, this reconstructs about
60.016 GWd/tHM, matching the LEAF reference after calendar-day rounding.

## Monte Carlo statistic

Choose the annual LEAF series with `--statistic`:

- `deterministic`
- `mean`
- `p025`
- `p50`
- `p975`

Use the same statistic for both `Load_Factor` and the calendar-driven annual-
equivalent `Burnup`.
