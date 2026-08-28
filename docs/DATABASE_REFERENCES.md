# Database.yaml technical review — 2026-08-24

## Scope

`data/Database.yaml` was technically reviewed on 2026-08-24. The objective
was to remove inherited, weakly documented parameter ranges and to make the
electricity, material and carbon boundaries explicit.

The database follows three rules:

- `[value]`: fixed referenced scalar;
- `[min, mode, max]`: referenced triangular technology uncertainty;
- `[min, max]`: legacy uniform range, supported only for compatibility.

For a triangular parameter, deterministic runs use the mode. The distribution
is sampled only when `monte_carlo.technology_uncertainty: true`. A sampled
technology parameter is drawn once per Monte Carlo simulation and reused
consistently wherever that parameter appears.

## Reconstructed electricity parameters

| Parameter | Database value | Basis |
| --- | ---: | --- |
| DAC, electricity | 0.400 kWh/kg CO2 | Concawe Table 77: 1.44 MJ/kg CO2 |
| MED, electricity | 0.00225 kWh/L | NREL 2024: 1.5–3 kWh/m3; fixed midpoint because no mode is reported |
| N2 PSA | [0.310, 0.365, 0.630] kWh/kg N2 | Published PSA range plus 0.365 kWh/kg design value |
| Alkaline electrolysis | [47, 52, 66] kWh/kg H2 | Commercial range; IRENA central value 52 kWh/kg |
| Haber-Bosch | 0.600 kWh/kg NH3 | Concawe Table 7: 2.16 MJ/kg NH3 |
| Methanation | 0.31944 kWh/kg CH4 | Concawe Table 2: 1.15 MJ/kg CH4 |

The previous values for `DACC`, `A`, `MFD`, `IO` and `PEM` were removed because
they did not have a complete, traceable reference basis in the inherited
database. Removing them is safer
than retaining unsupported optional technologies.

## Material balances

### Hydrogen

Alkaline-electrolysis make-up water is set to 10.5 L/kg H2, the midpoint of the
10–11 L/kg range reported for commercial alkaline and PEM systems in the 2026
water-consumption review. Cooling-water withdrawal is not included in this
material input.

### Ammonia

The NH3 feed fractions are stoichiometric:

- H2: 0.1776 kg/kg NH3;
- N2: 0.8224 kg/kg NH3.

The water input, 1.8648 L/kg NH3, follows the H2 requirement and the 10.5 L/kg
H2 make-up-water assumption.

### Methane

The integrated e-methane route is based on the Concawe balance:

- CO2 process feed: 3.00 kg/kg CH4;
- H2 feed: 0.50 kg/kg CH4;
- methanation water production: about 2.25 kg/kg CH4.

Using the 10.5 L/kg H2 make-up-water value adopted for alkaline electrolysis,
0.50 kg H2 requires 5.25 L of make-up water. Crediting the 2.25 kg of water
produced by methanation gives 3.00 L/kg CH4 net make-up water. This replaces the
inherited 7.495 L/kg gross flow and avoids charging desalination electricity to
water returned by the integrated process.

## CH4 carbon accounting

The old database stored 2.75 kg CO2/kg CH4 under `Production_Inputs`, while a
gross process-feed calculation uses 3.00 kg CO2/kg CH4. Concawe Table 2 shows
that these numbers describe different quantities:

- gross CO2 feed: 3.00 kg/kg CH4;
- CO2 released during methane synthesis: 0.25 kg/kg CH4;
- CO2-equivalent retained in CH4: 2.75 kg/kg CH4.

LEAF now stores these three quantities separately under `Carbon_Accounting`.
`Atmospheric_CO2_Captured` remains the net amount incorporated in methane,
while two additional outputs expose gross DAC capture and synthesis release.
This closes the carbon ledger without treating the process feed as the carbon
content of the final methane.

## Non-electric energy boundary

Two important heat requirements are now explicit metadata rather than hidden
inside electricity coefficients:

- DAC heat: 1.600 kWh_th/kg CO2, from Concawe Table 77;
- MED heat: 0.1825 kWh_th/L, the midpoint of the NREL 2024 range.

LEAF-EB currently balances electricity, so these heat requirements are not
converted silently into electricity. The Concawe integrated e-methane pathway
uses process heat integration for DAC. Any scenario that supplies these heat
loads electrically would require an explicit future heat-to-electricity model.

## Commodity-to-electricity conversion

Energy densities are consistently on an LHV basis:

- H2: 33.32 kWh/kg;
- NH3: 5.19 kWh/kg;
- CH4: 13.89 kWh/kg.

The fixed conversion efficiencies are:

- H2 PEMFC: 0.50;
- NH3 direct-ammonia SOFC: 0.60;
- CH4 CCGT: 0.585, NETL full-load net LHV efficiency.

The CCGT value is a rated fixed efficiency. NETL reports lower efficiency at
part load, but LEAF does not yet implement a load-dependent CCGT efficiency
curve. Randomizing that operational dependence as technology uncertainty would
be methodologically incorrect, so it is not represented by a triangle.

## Deterministic reference calculations

With technology uncertainty disabled, the revised database gives:

| Commodity | Production electricity | Reconversion electricity |
| --- | ---: | ---: |
| H2 | 52.023625 kWh/kg | 16.660000 kWh/kg |
| NH3 | 10.1395718 kWh/kg | 3.114000 kWh/kg |
| CH4 | 27.5261944444 kWh/kg | 8.125650 kWh/kg |

For the 100 MWh synthetic CH4 verification case, the revised database therefore
produces approximately 3,632.903 kg CH4 when BESS and interconnections are
disabled and all surplus is assigned to methane. The old 3,904.48 kg value is
not retained because it depended on the inherited database.

## Main references

- Concawe and Aramco (2022), *E-Fuels: A techno-economic assessment of European
  domestic production and imports towards 2050*, Report 17/22.
- Fthenakis et al. (2024), *Progress in Energy* 6, 032004, desalination review.
- IRENA (2022), *Renewable Technology Innovation Indicators*.
- Nayak-Luke, Banares-Alcantara and Wilkinson (2018), *Industrial & Engineering
  Chemistry Research*, DOI 10.1021/acs.iecr.8b02447.
- Liu, Elgowainy and Wang (2020), *Green Chemistry* 22, 5751–5761,
  DOI 10.1039/D0GC02301A.
- Luo et al. (2022), *Applied Energy* 307, 118158,
  DOI 10.1016/j.apenergy.2021.118158.
- NETL, *Cost and Performance Baseline for Fossil Energy Plants, Volume 5*.
- Sanchez et al. (2026), *WIREs Energy and Environment*,
  DOI 10.1002/wene.70035.

Exact URLs and parameter-to-reference mappings are embedded in
`data/Database.yaml`.
