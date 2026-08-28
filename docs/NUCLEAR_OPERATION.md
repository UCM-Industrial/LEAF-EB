# Nuclear fleets, refuelling, fuel use, and load-following in LEAF-EB

## Purpose

LEAF-EB represents each nuclear source as one homogeneous reactor fleet. A
fleet can grow or retire through dated capacity changes, while the reusable
technology template defines the unit size, fuel-cycle parameters, refuelling
assumptions, and supported operating rules.

The model separates quantities that should not be conflated:

1. installed fleet capacity;
2. capacity unavailable during offline refuelling;
3. actual hourly generation after the selected operating rule;
4. fuel exposure accumulated from actual operation; and
5. the mass and burnup of fuel discharged at each refuelling event.

The source name is arbitrary. `Nuclear_SMR300`, `PWR1000`, `HTR`, and other
names are valid. LEAF-EB does not require a source literally named `Nuclear`.

## One source, one homogeneous fleet

A scenario source represents a fleet of equivalent units. Dated additions do
not create new fleets when the reactor design and operating strategy are the
same:

```yaml
Nuclear_SMR300:
  technology_template:
    file: data/Nuclear_Technologies.yaml
    name: smr_300_reference
  hourly_operation: load_following
  dispatch_priority: 1
  custom_mode: replace
  capacity_additions:
    '2035-01-01': "600 MW"
    '2040-01-01': "600 MW"
    '2045-01-01': "600 MW"
  replaces:
    - [Eolica, Solar]
```

If `unit_capacity` is 300 MW, each 600-MW addition corresponds internally to
two equivalent units. The user does not enter the number of reactors
separately.

Different reactor designs or operating strategies can be represented by
separate sources. Each source then keeps its own unit size, fuel parameters,
refuelling history, availability, and fuel-discharge results.

## Minimum reactor and fuel input

The reusable technology contains the physical quantities required by the
fleet model:

```yaml
unit_capacity: "300 MW"

fuel_cycle:
  thermal_power: "967.742 MW"
  core_fuel_mass: "36.523 tHM"

refueling:
  operating_cycle: "24 month"
  outage_duration: "30 day"
  fuel_batches: 3
```

`unit_capacity` is a reactor property, not a refuelling property. LEAF uses it
to convert installed fleet capacity into equivalent units and to build fleet
availability.

`fuel_batches` is the number of equal-mass fuel batches used in the simplified
core-management representation. LEAF derives internally:

`reload fraction = 1 / fuel_batches`

and

`discharged mass per refuelling = core_fuel_mass / fuel_batches`.

For a 36.523-tHM core with three batches, one refuelling therefore discharges
about 12.174 tHM. The derived reload fraction is internal.

## Calendar-driven refuelling

With `operating_cycle`, the calendar controls the outage and burnup is a
result:

```yaml
refueling:
  operating_cycle: "24 month"
  outage_duration: "30 day"
  fuel_batches: 3
```

`operating_cycle` is the planned operating time after one refuelling outage
ends and before the next outage begins. The outage duration is additional to
that operating period.

By default, each equivalent unit keeps its own refuelling clock. LEAF does not
move an outage to avoid overlap with another unit. Units commissioned on the
same modeled date therefore keep the same refuelling dates unless the scenario
explicitly requests staggering.

Optional non-overlapping staggering remains available with:

```yaml
refueling:
  schedule: staggered
```

When this option is absent, every reported `Schedule_Shift` is zero and the
calendar follows only commissioning date, operating cycle, and outage
duration.

The calendar is identical for must-run and load-following cases when they use
the same technology and deployment. Their fuel burnup can nevertheless differ
because actual operation differs.

## Burnup-driven refuelling

When the fuel exposure itself should determine the next outage, omit
`operating_cycle` and define `target_burnup`:

```yaml
unit_capacity: "300 MW"

fuel_cycle:
  thermal_power: "967.742 MW"
  core_fuel_mass: "36.523 tHM"
  target_burnup: "60 GWd/tHM"

refueling:
  outage_duration: "30 day"
  fuel_batches: 3
```

LEAF derives the full fuel residence in equivalent full-power days (EFPD):

`residence EFPD = target burnup * 1000 * core fuel mass / thermal power`

and the batch-refuelling threshold:

`cycle EFPD = residence EFPD / fuel batches`.

EFPD accumulates from actual generation. A day at 50% average output adds
about 0.5 EFPD. A load-following fleet can therefore take longer in calendar
time to reach the same target burnup.

The public input defines exactly one refuelling basis:

- `operating_cycle` for calendar-driven refuelling; or
- `target_burnup` for burnup-driven refuelling.

Providing both, or neither for an offline-refuelled fleet, is rejected.

## Internal fuel-batch accounting

Fuel batches are internal bookkeeping, not user-defined reactor objects or
fuel assemblies. Each equivalent unit contains `fuel_batches` equal-mass
batches. During operation, all fuel resident in an available equivalent unit
receives the core-average burnup increment implied by that unit's utilization.

For calendar-driven operation, LEAF obtains the daily fleet EFPD from actual
generation relative to available capacity. Generation is shared uniformly
among equivalent available units, consistent with a homogeneous fleet.

At a refuelling event LEAF:

1. removes the oldest fuel batch;
2. records its discharged mass and accumulated burnup;
3. loads an equal mass of fresh fuel; and
4. continues the operating history with the same fleet-level representation.

This is a fleet-level fuel-use model. It does not model individual assemblies,
spatial power distributions, isotopic depletion, or detailed core management.
Those calculations belong downstream in a fuel-cycle or reactor-physics model.

A newly commissioned core starts with fresh fuel batches. Consequently, the
first discharge contains fuel exposed for one cycle, the second can contain
fuel exposed for two cycles, and the equilibrium discharge burnup is reached
after the initial batches have progressed through all modeled cycles. LEAF
retains this startup effect rather than assigning equilibrium burnup to a new
core artificially.

## Must-run and load-following

The operating strategy is independent of the refuelling basis.

For must-run operation:

```yaml
hourly_operation: must_run
```

available nuclear capacity generates at the configured must-run power
fraction. `dispatch_priority` is not needed because the source does not respond
to the hourly residual demand.

For load-following operation:

```yaml
hourly_operation: load_following
dispatch_priority: 1
```

nuclear generation responds to the residual after non-controlled generation,
subject to the technology's minimum and maximum power fractions, ramp limits,
deep-reduction rules, and refuelling availability.

Calendar-driven must-run and load-following cases can therefore use exactly the
same reactor and refuelling input. The comparison then isolates the effect of
operation on load factor and fuel burnup without moving the planned refuelling
calendar.

## Public fuel outputs

At analysis and detailed output levels, calendar-driven nuclear fleets retain
both annual fuel use and individual discharge events after Monte Carlo
consolidation. `Results.xlsx` contains the annual `Nuclear_Fuel` statistics.
`Tallies.xlsx` contains `Nuclear_Fuel_Annual` and `Nuclear_Fuel_Events`, including
deterministic values and Monte Carlo statistics. The machine-readable `Samples`
table retains the corresponding per-simulation records under the families
`Nuclear_Fuel_Annual` and `Nuclear_Fuel_Discharge`. These outputs do not depend
on `keep_analysis_temp`.

Each discharge record contains:

`Simulation | Source | Unit | Refueling_Number | Discharge_Date |`
`Discharged_Fuel | Fuel_Mass_Unit | Discharge_Burnup | Burnup_Unit |`
`Cycle_EFPD | Fuel_Batches`

The `Unit` identifier is an internal equivalent-unit label used for scheduling
and traceability; it does not imply detailed reactor physics.

Each annual record contains:

`Simulation | Year | Source | Installed_Power | Installed_Units |`
`Load_Factor | Refueling_Events | Discharged_Fuel | Fuel_Mass_Unit |`
`Mean_Discharge_Burnup | Burnup_Unit`

The annual mean discharge burnup is mass weighted. If no fuel is discharged in
a year, discharged mass is zero and mean discharge burnup is left missing. LEAF
does not insert zero, carry forward an earlier burnup, or invent a fleet-average
discharge value.

## ANICCA interface

`ANICCA_Input.xlsx` retains the existing annual six-column structure for each
nuclear fleet:

`Year | Installed_Power | Burnup | Load_Factor | Reactors_In | Reactors_Out`

For burnup-driven operation, the configured `target_burnup` remains available
as the burnup parameter. For calendar-driven operation, LEAF now calculates
physical discharge burnup independently, but no annual ANICCA mapping is
assumed yet. The `Burnup` field is therefore left blank for calendar-driven
fleets until the downstream ANICCA interpretation is defined explicitly.

This deliberately separates two tasks: LEAF first reports what fuel was
actually discharged and at what burnup; the annual representation expected by
ANICCA can then be selected without altering the physical LEAF calculation.

## Multiple fleets

Different technologies remain ordinary custom sources:

```yaml
sources:
  Nuclear_SMR300:
    technology_template:
      file: data/Nuclear_Technologies.yaml
      name: smr_300_reference
    ...

  Nuclear_PWR1000:
    technology_template:
      file: data/Nuclear_Technologies.yaml
      name: pwr_1000_reference
    ...
```

Each fleet has independent installed capacity, equivalent unit count,
refuelling calendar, operating history, and fuel-discharge accounting.

The broader redesign in which equal `dispatch_priority` values form one joint
dispatch group is separate from the fuel-cycle change described here. The
current fuel and refuelling implementation does not require that dispatch
redesign to calculate calendar-driven burnup correctly.
