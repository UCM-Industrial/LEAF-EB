# Stochastic method

## 1. Historical residual

For a historical series `y(t)`, LEAF-EB first estimates a fitted local level
and calendar pattern `f(t)`. The residual used by the current empirical
bootstrap is relative to that same fitted value:

```text
r(t) = [y(t) - f(t)] / f(t)
```

Using the same fitted basis for pattern extraction and residual normalization
is important. Mixing a continuous pattern with a discontinuous annual-mean
residual basis creates artificial jumps at calendar-year boundaries.

## 2. Unified stochastic simulation

Monte Carlo operation is controlled globally through
`simulation.monte_carlo.simulations`. A positive value activates stochastic
treatment for the selected uncertainty components; separate energy-source
and commodity switches are not used. Simulation 0 remains deterministic.

For energy, `simulation.monte_carlo.sources` explicitly names the empirical
series to perturb. The list is required when Monte Carlo simulations are
requested, preventing LEAF from silently changing Biomasa, Geotermica, or any
other source that the user intends to keep deterministic. Selected series are
aligned on the same historical dates. Custom sources are prescribed by their
own operating rules and dispatchable sources are allocated against residual
demand, so they should not be selected for historical-residual perturbation.

For commodities, every range-valued electricity-consumption, mass-balance or
reconversion parameter in `Database.yaml` is sampled in each Monte Carlo
worker. Deterministic simulation 0 uses the mean of each configured range.
Energy residuals and commodity parameters use different stochastic models
appropriate to their data, but they are enabled by the same Monte Carlo run
and are reproducible from the same simulation seed.

This preserves observed contemporaneous relationships among energy series
while propagating process-parameter uncertainty through commodity conversion.

## 3. Automatic block length

`optimal_stationary_block_length()` estimates a mean block length for each
residual series using the automatic stationary-bootstrap selector. Because a
single multivariate index path is required, LEAF-EB combines the active-source
estimates using their upper quartile.

The block length is data driven. No source name changes the rule.

## 4. Calendar-compatible restarts

The stationary bootstrap has a geometric restart probability of
`1 / mean_block_length`.

For daily data, a restart chooses a historical row that matches:

- target calendar month; and
- target weekday.

For hourly residual histories, the corresponding generic rule is month and
hour. Once a block has started, consecutive historical rows are retained until
another restart occurs or the historical time axis is discontinuous.

A restart joins two historical blocks that were not necessarily adjacent in
the observed record. LEAF-EB therefore checks only that artificial seam against
the empirical one-step changes of the jointly sampled residual series. Restart
candidates are preferred when every residual change stays within the empirical
99.5th percentile of observed consecutive changes. If no candidate satisfies
that condition, the closest multivariate candidate is used. This guard does
not clip residual values and does not alter any extreme event occurring inside
a sampled historical block.

## 5. Interannual variability regime

The stationary bootstrap preserves observed residual sequences inside sampled
blocks, but a long simulated year can combine blocks from several historical
years. When the historical record contains years with different residual
variability, that mixing can make simulated annual variability too narrow.

LEAF therefore calculates, for each complete historical year and stochastic
series:

```text
annual_scale(year, series)
    = annual residual std / complete-history residual std
```

For each simulated calendar year, one historical annual-scale vector is
sampled and applied jointly to all stochastic series after the stationary
bootstrap. The vector comes from one real historical year, so cross-series
annual regimes remain linked. Residual values, block selection, calendar
matching and block-join safeguards are otherwise unchanged.

Partial calendar years are not used as annual regimes. If fewer than two
complete historical years are available, the scale is exactly one and LEAF
uses the stationary bootstrap without interannual modulation. A single year
of history can therefore be simulated normally, but its Monte Carlo ensemble
represents only the variability supported by that year; interannual diversity
cannot be inferred from information that is not present.

## 6. Optional annual-target preservation

The deterministic projection defines the central long-term scenario.  By
default, Monte Carlo residuals are allowed to change both the timing and the
annual energy of every selected stochastic series.  This permits historically
plausible favorable and unfavorable years to propagate into the electricity
balance.

Users who instead want uncertainty only in within-year timing can set:

```yaml
simulation:
  monte_carlo:
    preserve_annual_targets: true
```

With this option, each selected stochastic series is rescaled within every
calendar year to the deterministic annual total after stochastic perturbation.
Capacity limits remain binding, so the option never overrides a physical
generation ceiling.  The default is `false`.

## 7. Multiplicative reconstruction

For local-relative empirical residuals, the future stochastic value is:

```text
MC(t) = deterministic(t) * [1 + r*(t)]
```

where `r*(t)` is the sampled residual.

This makes stochastic amplitude scale with the local deterministic level and
avoids the excessive future dispersion produced by applying historical errors
with one global absolute scale.

Stochastic residuals are applied at full amplitude from the first forecast
time step. The empirical stationary bootstrap already selects a
calendar-compatible historical residual at the boundary, so forcing the
first shock to zero would artificially suppress variability and make every
Monte Carlo simulations begin from the deterministic scenario series.

## 8. Historical hourly profile

When the simulation is hourly but the projection is daily, the daily residual
and the intraday profile are linked.

If the stationary bootstrap selects historical day `d`, the hourly expansion
uses the normalized 24-hour shape observed on day `d` for every stochastic
source that has hourly historical data.

The same sampled day is shared across the stochastic variables. This retains
within-day cross-variable timing that is lost when each future day uses only a
calendar-average profile.

## 9. Capacity feasibility

Daily energy is checked against the installed capacity and the selected hourly
shape before expansion. Hourly generation is then checked against accepted
capacity, including any configured tolerance.

If a limit binds, the excess is clipped locally and reported. The removed
energy is not transferred to a later time step.

## 10. Paired scenario comparison

Monte Carlo simulation `i` uses the same base seed and therefore the same
bootstrap history path and annual variability-regime draws in every scenario.
Relative stochastic shocks are the same for corresponding stochastic series
unless the scenario itself changes the deterministic scale.

This paired design isolates the effect of scenario changes from Monte Carlo
sampling noise.

## 11. Gaussian fallback

Gaussian residual parameters and covariance matrices are generated as a
fallback when a requested stochastic source has no empirical bootstrap
history. The empirical local-relative bootstrap is the preferred current
method.

## 12. Direct-hourly mode

LEAF-EB can also operate with hourly projection, hourly residuals and hourly
Monte Carlo perturbation. In this mode the historical hourly series is first
separated from its continuous structural annual level. The empirical
hour-of-year pattern is estimated on the resulting dimensionless shape, and
local-relative residuals are bootstrapped directly at hourly resolution.

Very small fitted signals are excluded from multiplicative residual estimation
using a generic threshold equal to 2% of the P95 positive fitted shape. The
rule is applied identically to every source and demand series.

Direct-hourly perturbation is a supported alternative when the projection
itself is hourly. When historical data are hourly but the long-term projection
is daily, the daily-to-hourly formulation remains the natural formulation
because the daily stochastic path and the recovered intraday profile remain
explicitly linked.
