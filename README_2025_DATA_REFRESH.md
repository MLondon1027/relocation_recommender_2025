# Relocation recommender data refresh

`final_data.csv` uses the newest nationwide ZIP Code Tabulation Area data available for this model:

- 2024 ACS 5-year subject tables: S0101, S1501, S2302, S1601, S2101, S0801, S2503, S1201, S2504, and S2502.
- 2025 FHFA annual five-digit ZIP House Price Index.
- 2024 Census Gazetteer land area, used with ACS population to calculate population density in residents per square kilometer (the same unit as the original dataset).

The ACS portion is correctly described as **2024 ACS 5-year**, not 2025 ACS. The 2025 ACS 5-year release was not available when this refresh was built.

## Output decisions

- ZIP/ZCTA identifiers remain five-character text, so values such as `02108` retain the leading zero.
- Areas with fewer than 2,000 residents are excluded, matching the original notebook.
- Missing ACS, HPI, or land-area values are imputed with each feature's national median after the population filter.
- `HPI_Available` marks rows whose 2025 HPI and annual-change values were both originally present.
- Current Census category names replace obsolete buckets: structures use `2020_Plus` and `2010_2019`; move-in year uses `2023_Plus`, `2020_2022`, and `2010_2019`.

## Rebuild

Install `pandas`, `numpy`, and `openpyxl`. Then run:

```bash
python build_2025_relocation_data.py \
  --census-dir /path/to/data_census_exports \
  --output final_data.csv
```

The Census directory must contain the ten data.census.gov ZIP exports named like `ACSST5Y2024.S0101_*.zip`. Alternatively, set `CENSUS_API_KEY`; the script will query any missing group through the Census API. FHFA HPI and Census Gazetteer files download automatically unless local files are supplied with `--hpi-file` and `--gazetteer-file`.

## Sources

- Census ACS 2024 5-year subject tables: https://data.census.gov/
- Census Gazetteer files: https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
- FHFA HPI datasets: https://www.fhfa.gov/data/hpi/datasets
