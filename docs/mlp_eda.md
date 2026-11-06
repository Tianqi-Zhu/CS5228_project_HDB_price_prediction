[Chinese version](mlp_eda.zh-CN.md)

# MLP data analysis and cleaning

This branch predicts monthly HDB rent in SGD from property information. EDA distinguishes formatting errors, possible transaction differences and temporal leakage, then evaluates cleaning on identical validation records.

**Current evidence does not justify treating every lower rent at an address as erroneous.** Removing obvious low-rent cases slightly lowers independent MLP ensemble RMSE from 471.80 to 471.26; retaining only high rents raises blend RMSE from 467.03 to 476.25. Cleaning shifts predictions upward; eliminating input-label conflicts does not establish predictive improvement.

**2026-11-05 full-data submission update:** Applying the same obvious-low-rent rule removes 2,017 of 150,000 labels and retains 147,983 for retraining the pure MLP ensemble. The user-provided Kaggle screenshot shows 439.38193 uncleaned versus 437.34455 cleaned, a decrease of 2.03738. This supports a limited increment for this configuration, not proof that every removed record is a partial rental or misreport. See the [prediction report](mlp_prediction_report.md).

## Data and actual fitting scope

| Data | Count | Dates and purpose |
| --- | ---: | --- |
| Original `data/train.csv` | 150,000 | 2021-01 through 2025-03, including fitting and chronological validation records |
| Original `data/test.csv` | 50,000 | 2025-03 through 2026-07; no rent labels |
| Original interval-model fitting data | 105,837 | 2021-01 through 2023-12 |
| Independent interval calibration | 9,005 | 2024-01 through 2024-03 |
| Complete outer validation | 35,158 | 2024-04 through 2025-03 |
| New full-data uncleaned MLP fitting set | 150,000 | 2021-01 through 2025-03, including former calibration/validation records |
| New full-data cleaned MLP fitting set | 147,983 | Same period; removes 2,017 records using the obvious-low-rent rule |

The original training file is not necessarily one model's fitting set. Original interval-model cleaning affects only 105,837 fitting records; calibration and outer validation retain all rows. Later known labels enter fitting only for full-data submission. Full-data models cannot treat former validation as an independent test. Neither stage overwrites raw CSVs.

Existing [full EDA](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/eda/report.md) covers distributions, temporal changes, spatial features and auxiliary tables. This page adds findings relevant to MLP training.

## Field usage

| Field or issue | Treatment and purpose |
| --- | --- |
| Variants such as `4 room` and `4-room` | Normalize flat-type categories so formatting does not become a property difference |
| Town, Street, Block | Normalize join keys, generate street/block IDs and use category embeddings; street information is included |
| Area, time, age and amenities | Learn numerical preprocessing on fitting data only; the network uses property, spatial and rental-history features |
| One record with lease year after approval year | Set derived negative age to missing and handle through fit-only preprocessing; do not guess the correct year |
| `FURNISHED=yes`, `FEE=0` | Constant in the current main data; excluded from main predictors |
| `NIL` and `-1` in building attributes | Treat as missing placeholders; highest floor describes the building, not the rented unit |
| Historical rents | Fitting rows use earlier months; calibration, validation and test histories freeze at the fitting cutoff |
| Stocks and COE | Used in separate RNN/LSTM experiments; the main MLP/TabM ensemble here uses neither those sequences nor latent vectors |

A raw table without nulls can still produce missing features through unseen categories, absent history or auxiliary placeholders. Static amenities also need not match transaction-time facilities.

## Duplicates and different rents for identical inputs

The full raw training file has 425 additional exact duplicates; the original interval experiment's 2021-2023 fitting subset has only 26. Do not mix counts from different periods.

Grouping all normalized raw non-target fields yields 12,763 original fitting groups with multiple rent values, involving 27,906 records. Without unit or contract IDs, identical block/type/area need not identify one transaction; differences are not automatically input errors.

For example, three 2023-06 records at 1C Cantonment Road, 4-room, Type S1, 94㎡, lease year 2011 have rents 1,500, 5,000 and 5,200. Original file lines 86944, 86598 and 85958 all entered fitting.

See [fitting cases with identical inputs and different rents](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/training_cases/training_similar_inputs_different_rents.csv). `ORIGINAL_CSV_LINE` includes the first-row header; `SOURCE_ROW` is the zero-based source index.

## Large flats with low rents: evidence and interpretation

The original fitting subset contains 231 records with area at least 90㎡ and rent at most S$1,300. One is 25 Telok Blangah Crescent in 2023-11: 114㎡, S$600, original line 102593. See [full fitting-period list](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/training_cases/training_large_flats_low_rents.csv). These are audit candidates, not confirmed partial rentals.

On 2026-11-03, we checked several later validation cases against the HDB source: S$650 at 112A Depot Road, S$950/1,300 at 654B Jurong West Street 61 and S$900 for an executive flat in Choa Chu Kang all appear upstream. This confirms provenance, not amount accuracy or rental scope. [Official query records](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/eda/official_low_rent_source_checks.json) retain restricted-month results without hidden test labels.

[HDB's data notes](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view) state that owners declare rents and HDB does not verify their accuracy. The source has only month, town, block, street, flat type and rent; it cannot verify unit, floor area or contract conditions.

In the full main data, 13,758 repeated address/type groups have constant area, flat model and lease year. This does not prove incorrect attributes, but these fields cannot distinguish units within a group. Special contracts, reporting conventions, amount errors or rental scope could explain differences; individual causes remain unknown. [Attribute-consistency statistics](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/eda/enriched_attribute_consistency.json)

## Cleaning rules and matched results

Rules were fixed before retraining without tuning thresholds on outer rents. Both variants retain architecture, seeds 42/43/44, three MLP epochs, four TabM epochs and identical blending.

**Obvious-low-rent cleaning** removes a fitting row if any condition holds: area≥90㎡ and rent≤1,300; rent below 50% of its same-month/town/type median with at least 20 reference records; or rent below 60% of the maximum in an identical-input group with at least two records.

**Strict-high-rent cleaning** keeps the first two rules but retains only the maximum rent in each identical-input group, including ties. Groups match month, town, block, street, type, model, area and lease year; they do not combine different years or types.

| Action | Removed | Retained | MLP RMSE | TabM RMSE | Blend RMSE | Blend MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncleaned | 0 | 105,837 | 471.80 | 469.74 | 467.03 | 343.66 |
| Obvious low rents | 1,332 | 104,505 | 471.26 | 477.72 | 468.79 | 337.00 |
| Strict high rents | 15,537 | 90,300 | 474.98 | 493.28 | 476.25 | 339.09 |

Obvious-low-rent cleaning reduces absolute error for about 55.1% of validation records and improves MAE, but worsening a few large errors raises RMSE. Strict cleaning removes fitting conflicts while shifting toward high rents: the S$1,500/5,000/5,200 group retains only 5,200, also discarding 5,000.

These results support further audits of low-rent definitions, not wholesale deletion of lower amounts. They apply to fixed configurations; cleaned models have not reselected epochs or undergone systematic tuning. Full [metrics](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/comparison_metrics.csv), [removal list](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/obvious_low/removed_training_rows.csv) and [strict-removal list](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/highest_only/removed_training_rows.csv) preserve row numbers and reasons.

## Temporal and evaluation boundaries

Rental-history features must not read the current rent, other same-month targets or future labels. Current rent, log rent and rent/area are not direct inputs. Vocabularies, imputation, scaling and numerical bins fit on the fitting partition only.

Interval calibration, tuning and final validation have distinct roles. Low-rent groups are post-hoc diagnostics, not inference-time routing based on true rent. Removing difficult outer cases cannot establish complete-test improvement. Previously inspected validation months are development data; conclusions need new origins.
