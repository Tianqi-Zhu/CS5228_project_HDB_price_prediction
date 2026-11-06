[Chinese version](report.zh-CN.md)

# Auxiliary-data EDA and preprocessing recommendations

This report analyzes six raw CSVs under `data/auxiliary/` and checks building-table matches against `train.csv` and `test.csv`. Run `python -m analysis.eda.run --scope auxiliary` to regenerate the [machine-readable summary](summary.json) and three figures. Implementation: `analysis/eda/audits/auxiliary.py`. The script only reads raw data; statistics and proposed features involve no model training.

## 1. File size and quality

| File | Records | Missing cells | Exact duplicate rows | Main findings |
| --- | ---: | ---: | ---: | --- |
| `sg-hdb-block.csv` | 9,464 | 0 | 0 | 285 postal codes are `NIL`; 11 maximum-floor and completion-year entries are `-1`. These placeholders must not be treated as valid values. |
| `sg-mrt-stations.csv` | 243 | 0 | 0 | 231 `open` and 12 `planned` records; only 192 distinct coordinate locations. |
| `sg-shopping-malls.csv` | 113 | 0 | 0 | 113 distinct names and coordinates. |
| `sg-schools.csv` | 337 | 0 | 0 | 182 records have `MAINLEVEL_CODE` containing `primary` or `p1`; four share coordinates with another school. |
| `sg-coe-prices.csv` | 539 | 0 | 0 | Covers 2021-01 to 2026-08; the final month, 2026-08, has only three records and is incomplete. |
| `sg-stock-prices.csv` | 91,642 | 510 | 0 | 77 stock symbols; 85 rows lack all six market fields, 2,623 have zero volume and 72 have anomalous OHLC ranges. |

All coordinates in the four geographic files parse and lie within valid latitude/longitude bounds. Missing-cell counts include parsed CSV nulls only, excluding placeholders such as `NIL` and `-1`.

## 2. Building address matching

After normalizing case, surrounding spaces and repeated whitespace in `TOWN + BLOCK + STREET`, the building table has no duplicate keys. Many-to-one matching on this composite key gives:

| Main dataset | Records | Matched | Match rate | Postal code `NIL` | Maximum floor / completion year `-1` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train.csv` | 150,000 | 150,000 | 100% | 5,259 | 159 |
| `test.csv` | 50,000 | 50,000 | 100% | 1,739 | 0 |

Building coordinates are therefore suitable first auxiliary features; postal code, maximum floor and completion year need separate placeholder handling. Valid completion years span 1949–2021 and maximum floors 2–50. High match coverage shows feasible joins, not proven predictive improvement.

No known completion year is later than rental approval year. Completion year differs from `LEASE_COMMENCE_DATE` by more than five years in 5,865 training and 1,932 test records. These years are not interchangeable; test derived ages separately. All postal codes except `NIL` have six digits.

## 3. Locations and distances

The [location map](01_auxiliary_locations.png) shows coordinates of 9,464 HDB buildings, MRT stations, malls and primary schools. The [nearest-facility distance plot](02_nearest_distances.png) measures each building's spherical straight-line distance to the nearest location, not walking distance.

| Nearest facility | Distinct locations used | Median distance | 90th-percentile distance | Buildings within 1 km |
| --- | ---: | ---: | ---: | ---: |
| MRT currently marked `open` | 192 | 0.568 km | 1.039 km | 88.46% |
| Malls | 113 | 0.747 km | 1.522 km | 68.97% |
| Primary schools, including mixed-level schools with `p1` | 180 | 0.366 km | 0.710 km | 97.23% |

The 243 MRT rows represent line/station codes, not 243 independent locations; all 12 `planned` records share coordinates with an `open` record. Deduplicate physical locations when calculating distances or counts to avoid counting interchanges repeatedly. Opening dates are unavailable, so current `STATUS` cannot establish operation during 2021–2025. Mall and school files also lack historical opening dates. The table describes a **current location snapshot**, not accessibility at historical approval times.

## 4. Economic-data coverage

The [monthly coverage plot](03_economic_coverage.png) shows COE record counts and symbols with valid market data; the dashed line marks the last training month, 2025-03.

- All COE `YEAR + MONTH` values parse, covering 68 consecutive months from 2021-01 to 2026-08. Categories are `a`, `b`, `c` and `e`, usually with two rounds per category per month. `YEAR + MONTH + CATEGORY + ROUND` has no duplicates. `QUOTA_PREMIUM`, `PREVAILING_QUOTA_PREMIUM`, `QUOTA` and `BIDS_RECEIVED` parse numerically after removing `$` and thousands separators; `QUOTA_PREMIUM` ranges from S$35,900 to S$158,004. All 134 category-`e` `PREVAILING_QUOTA_PREMIUM` entries are literal `$0` values and must not arbitrarily become missing. The three 2026-08 records do not form a complete monthly summary.
- All stock dates parse, covering 67 consecutive months from 2021-01 to 2026-07. `SYMBOL + DATE` has no duplicates, but symbols have only 113–1,403 rows each; monthly quoted-symbol coverage grows from 53 to 77. In 85 rows, OHLC, adjusted close and volume are all missing and must not become zero-price quotes. The 2,623 zero-volume rows should not automatically become missing either. Adjusted close has no nonpositive values.
- Of 77 symbols, 43 end in `.SI`, four in `.HK`, one in `.AS` and 29 have no exchange suffix. Adjusted prices span 0.092–6,544.8, so raw price levels should not be averaged across symbols. Another 72 rows have open or close outside `[LOW, HIGH]`, concentrated in four `.HK` symbols; inspect or filter these before using OHLC features.

Economic-feature availability depends on prediction time. **Monthly approval-time** estimation may use indicators published before that month. A **single fixed-origin forecast of every future month** at the training cutoff (2025-03) cannot know later COE or stock values. Their presence in auxiliary CSVs does not support claims of strict prospective forecasting. Define the scenario first and apply identical rules in validation and final prediction.

## 5. Recommended preprocessing and experiment order

| Data | Recommended treatment | Main risk / validation |
| --- | --- | --- |
| Building table | Left-join on normalized `TOWN + BLOCK + STREET`. Keep `POSTAL_CODE` as a string, map `NIL` to missing, and map `-1` in `MAX_FLOOR` and `YEAR_COMPLETED` to missing. Compare coordinates, Subzone, maximum floor and approval year minus completion year. | Preserve raw columns and fit missing-value handling on training data. Check completion years against approval years and compare completion-derived age with age from `LEASE_COMMENCE_DATE` for redundancy. |
| MRT | Calculate nearest-station straight-line distance and distinct locations within 1 km from matched coordinates. Check coordinates/station names and deduplicate first. | `STATUS` is current status. Claims of operation at approval time require opening dates; otherwise describe distance as a current geographic proxy and state the limitation. |
| Malls and schools | Compute nearest distances and nearby counts separately; select primary and mixed-primary schools using `MAINLEVEL_CODE` first. | Counts of colocated schools differ from counts of locations; fix the definition first. Without opening dates, historical existence cannot be asserted. |
| COE | Retain category and round, parse currency, quotas and bids, and define category-specific monthly aggregation before lagged joins to approval dates. | Do not average categories without explanation; preserve the meaning of category-`e` `$0`. Exclude prices unavailable at prediction time and exclude/flag incomplete 2026-08. |
| Stocks | Filter entirely missing market rows; compute returns per `SYMBOL` from `ADJUSTED_CLOSE`, then aggregate a small set of lagged monthly indicators. | Stock histories, exchanges and price scales differ. Check anomalous OHLC and zero volume before use, avoid averaging raw price levels, and never fill missing quotes with zero or use quotes unavailable at prediction time. |

Compare the current model with building attributes first, then add MRT, mall/school distances and finally COE/stock indicators separately. Keep temporal splits matched and record join coverage, RMSE, MAE, mean bias and monthly errors. Longer temporal validation is needed for genuine future forecasting. This EDA trains no auxiliary-data models and does not establish predictive gains from these features.
