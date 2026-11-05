"""Published planning-area income bands, without fictitious building incomes."""
import numpy as np
import pandas as pd


def income_features(planning_areas, origins, table, available_at="2022-08-01"):
    values = table.copy()
    values["key"] = values.Number.str.strip().str.lower()
    if values.key.duplicated().any():
        raise ValueError("Ambiguous planning-area income table")
    values = values.set_index("key")
    numeric = values.drop(columns="Number").apply(pd.to_numeric, errors="coerce")
    total = numeric.Total.replace(0, np.nan)
    result = pd.DataFrame(index=planning_areas.index)
    rates = {
        "CENSUS_INCOME_NO_EMPLOYED_SHARE": numeric.NoEmployedPerson / total,
        "CENSUS_INCOME_BELOW3000_SHARE": numeric[["Below_1_000", "1_000_1_999", "2_000_2_999"]].sum(axis=1, min_count=3) / total,
        "CENSUS_INCOME_GE10000_SHARE": numeric[[c for c in numeric if c[0].isdigit() and int(c.split("_")[0]) >= 10]].sum(axis=1, min_count=8) / total,
    }
    for name, rate in rates.items():
        result[name] = planning_areas.astype(str).str.lower().map(rate)
    cutoff = pd.PeriodIndex(origins, freq="M").to_timestamp(how="end")
    result.loc[cutoff.isna() | (cutoff < pd.Timestamp(available_at)), :] = np.nan
    result["CENSUS_INCOME_UNAVAILABLE"] = result.isna().any(axis=1).astype(float)
    return result
