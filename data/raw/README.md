# Local research data

CSV files in this directory are ignored by Git and must not be committed.

The study expects the latest file matching each pattern:

- `msci_acwi_grtr_*.csv`: MSCI ACWI (`892400`), USD Gross Total Return; columns `DATE`, `LEVEL`.
- `msci_acwi_netr_*.csv`: MSCI ACWI (`892400`), USD Net Total Return; columns `DATE`, `LEVEL`.
- `fred_dgs3mo_*.csv`: FRED DGS3MO; columns `observation_date`, `DGS3MO`.

Use an authorized MSCI source or export covered by your license. DGS3MO is
available from https://fred.stlouisfed.org/series/DGS3MO.
