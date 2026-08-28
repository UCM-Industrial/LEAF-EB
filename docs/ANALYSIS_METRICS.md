# LEAF-EB analysis metrics

## Purpose

The public analysis layer is organized around residual load and quantities
that can be derived directly from the demand and generation chronology already
constructed by LEAF-EB. The metrics do not modify scenario construction,
stochastic sampling, dispatch, storage, commodity conversion, or nuclear
operation.

LEAF-EB distinguishes two stages:

- **Initial residual load**: demand minus represented generation before the
  configured balancing sequence.
- **Remaining residual load**: the requirement left after the configured
  storage, exchange, dispatch, conversion, and reconversion stages.

This distinction keeps the historical-data-based demand/generation chronology
separate from the effect of the flexibility options applied by the scenario.

## Sign convention and units

For a time step `t`, let `B(t)` be the LEAF electricity balance written as
represented supply minus demand. Residual load is therefore

`R(t) = -B(t) = demand - represented supply`.

Positive residual load indicates a supply requirement and negative residual
load indicates surplus. LEAF simulation flows are stored as energy per time
step. For power-based metrics the interval energy is divided by the time-step
duration `dt` in hours.

## Energy

The public energy metrics are:

- `Initial_Surplus_Energy`
- `Initial_Positive_Residual_Load_Energy`
- `Remaining_Positive_Residual_Load_Energy`

Positive residual-load energy is the sum of the positive part of residual load
over the selected period. Surplus energy is the magnitude of the negative part.
This follows the common use of positive and negative residual-load energy in
long-term flexibility analysis.

## Peak power

LEAF reports:

- `Peak_Initial_Residual_Load`
- `Peak_Remaining_Residual_Load`

Peak metrics are power quantities. For an interval energy `E(t)` and time-step
duration `dt`, the corresponding average power is `E(t) / dt`. This keeps the
metric dimensionally consistent for both hourly and daily simulations.

## Short-term change

Residual-load ramp rate is calculated from consecutive average-power values:

`r(t) = [R_P(t) - R_P(t-1)] / dt`,

where `R_P` is residual-load power and `dt` is the time between observations in
hours. Analysis tables retain signed upward and downward diagnostics. The public
comparison layer reports the magnitude distribution for both initial and
remaining residual load:

- `P95_Residual_Load_Ramp_Magnitude`
- `P99_Residual_Load_Ramp_Magnitude`
- `Maximum_Residual_Load_Ramp_Magnitude`

The percentiles characterize high changes without relying only on a single
maximum, while the maximum remains available as the extreme observed change.
For an hourly LEAF simulation these are one-hour ramp rates in MW/h. For a
different temporal resolution, `Time_Step_Hours` records the actual interval.

## Persistence

A positive-residual-load episode is a contiguous sequence of time steps with
positive residual load. LEAF reports episode count, active hours, total energy,
mean and maximum duration, P95 duration, mean and maximum episode energy, and
maximum peak power.

Analysis-level outputs distinguish:

- initial positive-residual-load episodes;
- remaining positive-residual-load episodes;
- remaining surplus episodes.

Frequency, duration, energy/severity, and peak power therefore describe
persistence separately from cumulative energy and the single annual peak.

## Residual-load duration curves

LEAF reports selected exceedance points for both the initial and remaining
residual-load duration curves. These curves are expressed in MW and show how
the configured flexibility sequence changes the distribution of residual load.
The chronological episode metrics remain separate because a duration curve does
not retain event ordering.

## Battery use

`Full_Equivalent_Cycles` is calculated from actual BESS energy withdrawal
relative to the available BESS energy capacity at each time step and accumulated
through the year. It is a throughput-based utilization metric. SOC, charging,
discharge, and low-SOC diagnostics remain separate because FEC alone does not
describe temporal availability or electrochemical degradation.

## Interpretation boundary

The residual-load metrics characterize temporal balance requirements derived
from the scenario and its historical-data-based time series. They are planning
and scenario-comparison quantities. Reliability indices, reserve requirements,
market outcomes, and network constraints require the corresponding additional
models or assumptions.

## Literature basis

The metric families follow established residual-load, flexibility, energy-drought,
and storage-utilization analyses. Particularly relevant references are:

- Huber, M., Dimkova, D., & Hamacher, T. (2014). *Integration of wind and
  solar power in Europe: Assessment of flexibility requirements*. Energy, 69,
  236-246. DOI: 10.1016/j.energy.2014.02.109. Uses net-load ramp distributions
  and 5-95, 1-99, and min-max ranges.
- Bauhofer, P., & Zoglauer, M. (2021). *Safeguarding Climate Targets:
  Hydropower Flexibility Facilities*. Chemie Ingenieur Technik, 93, 632-640.
  DOI: 10.1002/cite.202000157. Reports positive/negative residual-load energy,
  residual-load peaks, gradients, block durations, and block counts.
- Raynaud, D., Hingray, B., François, B., & Creutin, J.-D. (2018). *Energy
  droughts from variable renewable energy sources in European climates*.
  Renewable Energy, 125, 578-589. DOI: 10.1016/j.renene.2018.02.130. Uses
  uninterrupted high production-demand mismatch periods and characterizes
  their frequency and duration.
- Kucevic, D., et al. (2020). *Standard battery energy storage system
  profiles: Analysis of various applications for stationary energy storage
  systems using a holistic simulation framework*. Journal of Energy Storage,
  28, 101077. DOI: 10.1016/j.est.2019.101077. Uses full-equivalent cycles and
  other profile characteristics to describe BESS utilization.
