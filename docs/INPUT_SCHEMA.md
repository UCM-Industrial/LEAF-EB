# Units and public input schema

LEAF-EB separates the physical meaning of a parameter from the unit used to
enter or report its value.


## YAML comments

YAML uses `#` for comments. LEAF-EB ignores everything after `#` on the same
line. The public examples use comments to explain units, non-obvious switches
and the purpose of each configuration block. Comments do not change a
simulation and are not copied into runtime values.

```yaml
simulation:
  monte_carlo:
    simulations: 5                 # Small demonstration sample.
    seed: 12345                    # Reproducible random seed.
    preserve_annual_targets: false # Annual energy may vary.
```

## Public-input rule

A dimensional public YAML field carries its unit with the value:

```yaml
demand:
  target_production: "200 TWh/year"

sources:
  Nuclear_SMR300:
    unit_capacity: "300 MW"
    capacity_additions:
      '2035-01-01': "600 MW"
    fuel_cycle:
      thermal_power: "968 MW"
      core_fuel_mass: "36.5 tHM"
    refueling:
      operating_cycle: "24 month"
      outage_duration: "30 day"
      fuel_batches: 3
```


For burnup-driven nuclear refuelling, the calendar field is omitted and the
fuel-cycle block adds a burnup target:

```yaml
fuel_cycle:
  thermal_power: "968 MW"
  core_fuel_mass: "36.5 tHM"
  target_burnup: "60 GWd/tHM"
```

`operating_cycle` and `target_burnup` are mutually exclusive refuelling bases.

The parameter name identifies the physical quantity. The unit is not encoded
in the parameter name. For example, `thermal_power` and `reference_net_power`
are both power quantities and may both use `MW`; their physical meaning comes
from the field name.

Dimensionless fractions remain plain numbers on a 0-1 basis unless a field
explicitly documents another convention.

## Historical datasets

External historical time-series files are the deliberate exception. Demand and
all energy-generation columns in one dataset must use the same energy unit.
That unit is declared once:

```yaml
historical_data:
  file: Inputs/data/example_history_hourly.csv
  date_column: Date
  processing_resolution: hourly
  unit: MWh
```

LEAF-EB does not infer or mix units column by column.

## Accepted dimensional forms

The public parser currently supports:

| Quantity | Examples |
|---|---|
| Energy | `500 MWh`, `2 GWh`, `0.2 TWh` |
| Energy per period | `550 GWh/day`, `200 TWh/year` |
| Power | `300 MW`, `0.3 GW` |
| Duration | `7 h`, `30 day` |
| Calendar cycle | `18 month`, `1.5 year` |
| Heavy-metal mass | `36500 kgHM`, `36.5 tHM` |
| Burnup | `60 GWd/tHM`, `60000 MWd/tHM`, `60 MWd/kgHM` |
| Emission factor | `11 gCO2e/kWh`, `11 kgCO2e/MWh` |


Canonical accepted spellings are intentionally limited:

- Energy: `kWh`, `MWh`, `GWh`, `TWh`.
- Energy-period denominator: `hour`, `day`, `year` (common short aliases
  such as `h`, `d`, `yr` are also accepted).
- Power: `kW`, `MW`, `GW`; electrical or thermal qualifiers such as
  `MWe` and `MWth` are also recognized as unit spellings.
- Elapsed duration: hours or days. Calendar-cycle fields accept months or
  years.
- Heavy-metal mass: `kgHM`, `tHM`.
- Burnup: `MWd/tHM`, `GWd/tHM`, `MWd/kgHM`, `GWd/kgHM`.
- Emissions numerator: `gCO2e`, `kgCO2e`, `tCO2e`, divided by any
  supported energy unit.
- Percentage fields that carry units: `%`, `percent`, `pct`.
- Fractional rates: `/h`, `/hour`, `/day` (the equivalent `1/...` forms
  are also accepted).

New public inputs should normally use `MW`; fields such as `thermal_power`
or `reference_net_power` already provide the electrical or thermal meaning.

Ambiguous mass units such as `ton` or `tons` are not accepted for heavy-metal
mass. Use `kgHM` or `tHM`.

`demand.target_production` is an energy-per-period quantity. A value such as
`"500 TWh"` is rejected because the intended period is ambiguous. Use, for
example, `"500 TWh/year"` or an equivalent daily value. The scalar
`year -> day` conversion uses 365 days, consistent with the existing
long-horizon target convention; calendar patterning remains part of the
forecast model.

## Internal normalization

Units are parsed once while the YAML is loaded. The calculation engine then
uses canonical internal quantities, principally MWh for electrical energy, MW
for power, tHM for heavy-metal mass, GWd/tHM for burnup, and hours/days/months
where required by the corresponding algorithm.

Canonical field names are preserved when values enter the runtime mapping.
Dimensional values are parsed in place; the parser does not create a second
engine-facing name for the same quantity. Runtime fields and result columns
therefore use one vocabulary independent of the unit chosen in the YAML.

Only current schema names are accepted. This keeps the public schema,
calculation engine, tests, and documentation aligned around one canonical
name for each modeled quantity.

## Public outputs

Public indicators identify the physical quantity without adding the display
unit to the field name. Units remain in dedicated metadata or unit columns.
The compact public vocabulary includes names such as:

```text
Initial_Surplus_Energy
Initial_Positive_Residual_Load_Energy
Battery_Discharge
Remaining_Positive_Residual_Load_Energy
Peak_Initial_Residual_Load
Peak_Remaining_Residual_Load
Full_Equivalent_Cycles
P95_Residual_Load_Ramp_Magnitude
P99_Residual_Load_Ramp_Magnitude
Maximum_Residual_Load_Ramp_Magnitude
```


The public balance metrics are organized around residual load, defined with the
sign convention demand minus represented supply. Positive residual load is a
remaining supply requirement; negative residual load is surplus. Energy metrics
integrate the positive or negative parts over the represented time steps, while
peak residual load is reported as average power over the corresponding time
step. Ramp metrics are calculated from consecutive changes in residual-load
power and are reported as MW/h. LEAF-EB reports the 95th and 99th percentiles
of absolute ramp magnitude together with the maximum; signed upward and
downward diagnostics remain available in the analysis tables.

Positive-residual-load episodes are contiguous periods for which remaining
residual load is greater than zero. Their count, duration, energy, and peak
power describe persistence separately from annual energy or the single peak.

The output level is selected only through `output.level`, with the canonical
values `comparison`, `analysis`, and `detailed`. `Results.xlsx` contains the
compact scenario indicators, `Samples` retains annual Monte Carlo values, and
`Tallies.xlsx` contains analysis-level temporal and operational summaries.

For nuclear cases, `ANICCA_Input.xlsx` uses the fixed interface:

`Year | Installed_Power | Burnup | Load_Factor | Reactors_In | Reactors_Out`

No compatibility aliases are accepted for the removed output schema.
