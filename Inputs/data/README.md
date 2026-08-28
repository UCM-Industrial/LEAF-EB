# Synthetic example datasets

The files in this directory are synthetic and are provided only to make the
public examples self-contained. They do not represent a national electricity
system and should not be used as empirical data.

- `example_history_daily.csv`: daily demand and generation for 2018-2021.
  Used by Examples 01-02.
- `example_history_hourly.csv`: hourly demand and generation for 2019-2021.
  Used by Examples 03-06 for hourly storage, commodity and nuclear
  operation.

All demand and generation columns use MWh for each represented time step. The
corresponding YAML file declares that shared unit under `historical_data.unit`.
