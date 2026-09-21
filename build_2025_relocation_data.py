#!/usr/bin/env python3
"""Build Streamlit-ready relocation features from 2024 ACS and 2025 FHFA data."""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ACS_YEAR = 2024
HPI_YEAR = 2025
GROUPS = ("S0101", "S1501", "S2302", "S1601", "S2101", "S0801", "S2503", "S1201", "S2504", "S2502")
HPI_URL = "https://www.fhfa.gov/hpi/download/annual/hpi_at_zip5.xlsx"
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_zcta_national.zip"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "relocation-data-refresh/1.0"})
    with urlopen(request, timeout=180) as response:
        return response.read()


def read_census_export(path: Path, variables: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        data_name = next(name for name in archive.namelist() if name.endswith("-Data.csv"))
        with archive.open(data_name) as source:
            frame = pd.read_csv(
                source,
                dtype=str,
                skiprows=[1],
                usecols=["GEO_ID", *variables],
                low_memory=False,
            )
    frame["index"] = frame["GEO_ID"].str[-5:].str.zfill(5)
    return frame.set_index("index")


def read_census_api(group: str, variables: list[str], api_key: str) -> pd.DataFrame:
    chunks = []
    for start in range(0, len(variables), 45):
        selected = variables[start : start + 45]
        query = urlencode(
            {
                "get": "NAME," + ",".join(selected),
                "for": "zip code tabulation area:*",
                "key": api_key,
            }
        )
        rows = pd.read_json(f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5/subject?{query}")
        rows.columns = rows.iloc[0]
        rows = rows.iloc[1:].copy()
        rows["index"] = rows["zip code tabulation area"].astype(str).str.zfill(5)
        chunks.append(rows.set_index("index")[selected])
    return pd.concat(chunks, axis=1)


def load_group(group: str, variables: list[str], census_dir: Path, api_key: str | None) -> pd.DataFrame:
    matches = sorted(census_dir.glob(f"ACSST5Y{ACS_YEAR}.{group}_*.zip"))
    if matches:
        return read_census_export(matches[-1], variables)
    if api_key:
        return read_census_api(group, variables, api_key)
    raise FileNotFoundError(
        f"Missing {group} Census export. Put its data.census.gov ZIP in {census_dir} "
        "or set CENSUS_API_KEY."
    )


AGE_NAMES = [
    "Percent_Under_5", "Percent_5_to_9", "Percent_10_to_14", "Percent_15_to_19",
    "Percent_20_to_24", "Percent_25_to_29", "Percent_30_to_34", "Percent_35_to_39",
    "Percent_40_to_44", "Percent_45_to_49", "Percent_50_to_54", "Percent_55_to_59",
    "Percent_60_to_64", "Percent_65_to_69", "Percent_70_to_74", "Percent_75_to_79",
    "Percent_80_to_84", "Percent_85_Plus",
]

EDUCATION = {
    "Percent_Bachelors": "S1501_C02_012E",
    "Percent_Graduate_Professional": "S1501_C02_013E",
    "Percent_High_School_Or_Higher": "S1501_C02_014E",
    "Percent_Some_College_No_Degree": "S1501_C02_010E",
    "Percent_Male_Bachelors": "S1501_C04_012E",
    "Percent_Female_Bachelors": "S1501_C06_012E",
    "Percent_Male_Graduate_Professional": "S1501_C04_013E",
    "Percent_Female_Graduate_Professional": "S1501_C06_013E",
    "Percent_Male_High_School_Or_Higher": "S1501_C04_014E",
    "Percent_Female_High_School_Or_Higher": "S1501_C06_014E",
}

EMPLOYMENT = {
    "Percent_Families_With_Children": "S2302_C04_001E",
    "Percent_Families_Husband_LF_Wife_Not": "S2302_C02_004E",
    "Percent_Children_Families_Husband_LF_Wife_Not": "S2302_C04_004E",
    "Percent_Families_Wife_LF_Husband_Not": "S2302_C02_005E",
    "Percent_Children_Families_Wife_LF_Husband_Not": "S2302_C04_005E",
    "Percent_Families_Both_Spouses_Not_LF": "S2302_C02_006E",
    "Percent_Children_Families_Both_Spouses_Not_LF": "S2302_C04_006E",
    "Percent_Families_Female_No_Spouse": "S2302_C02_008E",
    "Percent_Children_Families_Female_No_Spouse": "S2302_C04_008E",
    "Percent_Families_Female_No_Spouse_In_LF": "S2302_C02_009E",
    "Percent_Children_Families_Female_No_Spouse_In_LF": "S2302_C04_009E",
    "Percent_Families_Male_No_Spouse": "S2302_C02_011E",
    "Percent_Children_Families_Male_No_Spouse": "S2302_C04_011E",
    "Percent_Families_Male_No_Spouse_In_LF": "S2302_C02_012E",
    "Percent_Children_Families_Male_No_Spouse_In_LF": "S2302_C04_012E",
    "Percent_Families_Male_No_Spouse_Not_LF": "S2302_C02_013E",
    "Percent_Families_1_Worker": "S2302_C02_016E",
    "Percent_Families_2_Plus_Workers": "S2302_C02_017E",
    "Percent_Children_Families_No_Workers": "S2302_C04_015E",
    "Percent_Children_Families_1_Worker": "S2302_C04_016E",
    "Percent_Families_No_Workers": "S2302_C02_015E",
}

FEATURE_MAP = {
    "age": {"total_pop": ("S0101", "S0101_C01_001E"), **{name: ("S0101", f"S0101_C02_{n:03d}E") for n, name in enumerate(AGE_NAMES, 2)}},
    "education": {name: ("S1501", var) for name, var in EDUCATION.items()},
    "employment": {name: ("S2302", var) for name, var in EMPLOYMENT.items()},
    "language": {
        "Speaks_Spanish": ("S1601", "S1601_C02_023E"),
        "Speaks_Other_Language": ("S1601", "S1601_C02_003E"),
        "Speak_Only_English": ("S1601", "S1601_C02_002E"),
        "Poor_English_Speaking": ("S1601", "S1601_C06_001E"),
    },
    "veteran": {"Percent_Veterans": ("S2101", "S2101_C04_001E")},
    "transportation": {
        "Percent_Private_Vehicle": ("S0801", "S0801_C01_002E"),
        "Percent_Public_Trans": ("S0801", "S0801_C01_009E"),
        "Percent_Walked": ("S0801", "S0801_C01_010E"),
        "Percent_Remote_Work": ("S0801", "S0801_C01_013E"),
        "Mean_Commute_Time": ("S0801", "S0801_C01_046E"),
    },
    "income": {
        "Median_HH_Income": ("S2503", "S2503_C02_013E"),
        "Median_Owner_HH_Income": ("S2503", "S2503_C04_013E"),
        "Median_Renter_HH_Income": ("S2503", "S2503_C06_013E"),
    },
    "marital": {
        "Percent_Now_Married": ("S1201", "S1201_C02_001E"),
        "Percent_Widowed": ("S1201", "S1201_C03_001E"),
        "Percent_Divorced": ("S1201", "S1201_C04_001E"),
        "Percent_Separated": ("S1201", "S1201_C05_001E"),
        "Percent_Never_Married": ("S1201", "S1201_C06_001E"),
    },
    "housing": {
        "Percent_SF_Houses_Det": ("S2504", "S2504_C02_002E"),
        "Percent_SF_Houses_Att": ("S2504", "S2504_C02_003E"),
        "Percent_2F_Apts": ("S2504", "S2504_C02_004E"),
        "Percent_3_4F_Apts": ("S2504", "S2504_C02_005E"),
        "Percent_5_9F_Apts": ("S2504", "S2504_C02_006E"),
        "Percent_10_PlusF_Apts": ("S2504", "S2504_C02_007E"),
        "Percent_Mobile_Home": ("S2504", "S2504_C02_008E"),
        "Percent_Yr_Built_2020_Plus": ("S2504", "S2504_C02_009E"),
        "Percent_Yr_Built_2010_2019": ("S2504", "S2504_C02_010E"),
        "Percent_Yr_Built_2000_2009": ("S2504", "S2504_C02_011E"),
        "Percent_Yr_Built_1980_1999": ("S2504", "S2504_C02_012E"),
        "Percent_Yr_Built_1960_1979": ("S2504", "S2504_C02_013E"),
        "Percent_Yr_Built_1940_1959": ("S2504", "S2504_C02_014E"),
        "Percent_Yr_Built_1939_Prior": ("S2504", "S2504_C02_015E"),
    },
    "moved": {
        "Percent_Moved_In_2023_Plus": ("S2502", "S2502_C02_022E"),
        "Percent_Moved_In_2020_2022": ("S2502", "S2502_C02_023E"),
        "Percent_Moved_In_2010_2019": ("S2502", "S2502_C02_024E"),
        "Percent_Moved_In_2000_2009": ("S2502", "S2502_C02_025E"),
        "Percent_Moved_In_1990_1999": ("S2502", "S2502_C02_026E"),
        "Percent_Moved_In_1989_Prior": ("S2502", "S2502_C02_027E"),
    },
}


def numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series.replace({"(X)": np.nan, "-": np.nan, "N": np.nan}), errors="coerce")
    return values.mask(values <= -1_000_000)


def build_acs(census_dir: Path, api_key: str | None) -> pd.DataFrame:
    flat_map = {
        output_name: source
        for section in FEATURE_MAP.values()
        for output_name, source in section.items()
    }
    features: dict[str, pd.Series] = {}
    owner = occupied = None
    for group in GROUPS:
        group_map = {name: var for name, (source_group, var) in flat_map.items() if source_group == group}
        variables = sorted(set(group_map.values()))
        if group == "S2302":
            variables.append("S2302_C02_001E")
        if group == "S2503":
            variables.extend(["S2503_C03_001E", "S2503_C01_001E"])
        table = load_group(group, sorted(set(variables)), census_dir, api_key)
        for output_name, variable in group_map.items():
            features[output_name] = numeric(table[variable])
        if group == "S2302":
            families_with_children = numeric(table["S2302_C04_001E"])
            all_families = numeric(table["S2302_C02_001E"])
            features["Percent_Families_With_Children"] = families_with_children.div(all_families).mul(100)
        if group == "S2503":
            owner = numeric(table["S2503_C03_001E"])
            occupied = numeric(table["S2503_C01_001E"])
    result = pd.DataFrame(features)
    assert owner is not None and occupied is not None
    result["Percent_Owner_Occupied"] = owner.div(occupied).mul(100)
    return result


def load_hpi(path: Path | None) -> pd.DataFrame:
    source = path if path else io.BytesIO(fetch(HPI_URL))
    hpi = pd.read_excel(source, header=5, dtype={"Five-Digit ZIP Code": str})
    hpi["index"] = hpi["Five-Digit ZIP Code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    hpi = hpi[pd.to_numeric(hpi["Year"], errors="coerce") == HPI_YEAR].copy()
    hpi = hpi.set_index("index")[["Annual Change (%)", "HPI"]]
    return hpi.rename(columns={"Annual Change (%)": "HPI_%_Annual_Change"}).apply(pd.to_numeric, errors="coerce")


def load_land_area_sq_km(path: Path | None) -> pd.Series:
    raw = path.read_bytes() if path else fetch(GAZETTEER_URL)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as source:
            gaz = pd.read_csv(source, sep="\t", dtype={"GEOID": str})
    gaz.columns = gaz.columns.str.strip()
    gaz["index"] = gaz["GEOID"].astype(str).str.zfill(5)
    square_meters = pd.to_numeric(gaz.set_index("index")["ALAND"], errors="coerce")
    return square_meters / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census-dir", type=Path, default=Path("."))
    parser.add_argument("--hpi-file", type=Path)
    parser.add_argument("--gazetteer-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("final_data.csv"))
    parser.add_argument("--minimum-population", type=int, default=2000)
    args = parser.parse_args()

    api_key = os.getenv("CENSUS_API_KEY")
    final = build_acs(args.census_dir, api_key)
    final = final[final["total_pop"] >= args.minimum_population].copy()

    hpi = load_hpi(args.hpi_file)
    final = final.join(hpi, how="left")
    final["HPI_Available"] = final[["HPI_%_Annual_Change", "HPI"]].notna().all(axis=1).astype(float)

    land_area = load_land_area_sq_km(args.gazetteer_file)
    final["density"] = final["total_pop"].div(land_area.reindex(final.index).replace(0, np.nan))

    missing_before = final.isna().sum()
    for column in final.columns:
        if final[column].isna().any():
            final[column] = final[column].fillna(final[column].median())

    final.index = final.index.astype(str).str.zfill(5)
    final.index.name = "index"
    final = final.sort_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, float_format="%.6g")

    print(f"Wrote {len(final):,} ZIP/ZCTA rows and {len(final.columns):,} features to {args.output}")
    print("Columns imputed with national medians:")
    for name, count in missing_before[missing_before > 0].sort_values(ascending=False).items():
        print(f"  {name}: {int(count):,}")


if __name__ == "__main__":
    main()
