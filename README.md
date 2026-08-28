# LEAF-EB

**LEAF-EB (Long-term Energy Analysis Framework - Electricity Balance)**
constructs and evaluates long-term electricity scenarios while preserving the
temporal characteristics that affect operation at daily or hourly resolution.

A scenario is specified through user inputs that combine historical data with
dated assumptions for demand, generation, capacity changes, technology
parameters and operating rules. These assumptions may come from planning or
optimization studies, policy targets or published scenarios, or they may be
defined directly by the user in LEAF-EB.

LEAF-EB constructs the corresponding time series and evaluates the electricity
balance sequentially. The framework can represent battery storage, commodity
production and reconversion, nuclear operation and refueling, emissions and
Monte Carlo sampling. It does **not** optimize the future generation mix and
does not perform network-constrained unit commitment or power-flow
calculations.

## Main capabilities

- daily and hourly scenario simulation;
- deterministic and Monte Carlo analyses;
- historical temporal-pattern reconstruction;
- battery storage and configurable dispatch order;
- hydrogen, ammonia and methane conversion chains;
- constant-output and load-following nuclear operation;
- unit-level nuclear refueling and fuel-use calculations;
- emissions accounting;
- annual export for ANICCA fuel-cycle analyses.

## Requirements

LEAF-EB requires **CPython 3.11 or newer**.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository is intended to run directly from source. A PyPI package is not
required.

## Quick start

Six commented, self-contained examples are included. Start with:

```bash
python Runner.py Example_01_basic
```

List all available examples with:

```bash
python Runner.py --list
```

The examples are progressive:

| Example | New concept introduced |
| --- | --- |
| `Example_01_basic` | Deterministic electricity balance |
| `Example_02_monte_carlo` | Temporal Monte Carlo uncertainty |
| `Example_03_bess` | Battery storage |
| `Example_04_commodities` | Methane production, storage and reconversion |
| `Example_05_nuclear_constant` | Constant-output nuclear and calendar refueling |
| `Example_06_nuclear_load_following` | Load-following nuclear and burnup refueling |

Every example YAML contains `#` comments explaining the main parameters. The
historical datasets are synthetic and are included only for learning and
software verification.

## Repository structure

```text
LEAF-EB/
├── Runner.py
├── Inputs/
│   ├── Example_01_basic.yml
│   ├── ...
│   ├── Example_06_nuclear_load_following.yml
│   └── data/
├── data/
│   ├── Database.yaml
│   └── Nuclear_Technologies.yaml
├── docs/
├── scripts/
│   └── leaf_to_anicca.py
├── src/
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

`src/` contains the runtime calculation modules. `data/Database.yaml` stores
commodity technology parameters and `data/Nuclear_Technologies.yaml` contains
reusable nuclear templates.

## Configuration

Scenario assumptions are defined in the input YAML files. The six public
examples are commented with `#` so users can see the meaning, units and role
of the main parameters directly in the files. Detailed configuration rules
are documented in `docs/INPUT_SCHEMA.md`.

LEAF-EB records the effective settings of each run in `Resolved_Config.yaml`
for reproducibility. Temporal Monte Carlo uncertainty and technology-parameter
uncertainty are controlled independently.

## Documentation

- `docs/LEAF-EB_User_Manual_2026-08-27.docx` - user-friendly manual linked
  directly to the six public examples;
- `docs/EXAMPLES.md` - guided tour through the six public examples;
- `docs/INPUT_SCHEMA.md` - input structure and units;
- `docs/STOCHASTIC_METHOD.md` - temporal Monte Carlo method;
- `docs/NUCLEAR_OPERATION.md` - nuclear operation and refueling;
- `docs/ANALYSIS_METRICS.md` - reported indicators;
- `docs/DATABASE_REFERENCES.md` - technology-parameter references;
- `docs/LEAF_TO_ANICCA.md` - optional annual ANICCA interface.

## Verification

The public repository includes automated tests and a GitHub Actions workflow.
The workflow validates and executes all six public examples on Python 3.11 and
3.12.

Run the tests locally with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Study-specific datasets, large simulation outputs and publication result
archives are intentionally not included in the repository.

## Citation

If LEAF-EB contributes to published work, cite the software using the metadata
in `CITATION.cff`. A paper citation can be added there when the methodological
article is published.

## License

LEAF-EB is released under the MIT License. See `LICENSE`.
