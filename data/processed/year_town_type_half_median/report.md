# Annual town and flat-type low-rent audit

Source: `data/train.csv`; SHA-256 `2fe38ce9e0d8a02e9786f066d4a10d575d91f34be1d9111f0142e3f43fbfc8a5`.

The requested strict rule flags **1,121 / 150,000 rows (0.747%)** and retains **148,879 rows**.

## Method

Group by approval calendar year, normalized town and normalized flat type. Merge aliases such as `4 room` / `4-room`; keep executive separate. Calculate one median from all original labeled rows in each group, including the row being assessed. Remove only rent < 0.5 x that median. There is no extra minimum group size, flat-type restriction, fallback, or repeated filtering.

Exactly half the median is retained: 189 rows. Of 624 observed groups, 147 have fewer than 30 rows; 42 flagged rows come from these small groups. The data cover 2021-01 through 2025-03; 2025 contains January-March only.

## By year

| YEAR | total_rows | removed_rows | equal_to_half_median_retained | retained_rows | removed_pct |
| --- | --- | --- | --- | --- | --- |
| 2021 | 37619 | 29 | 12 | 37590 | 0.08 |
| 2022 | 32564 | 121 | 27 | 32443 | 0.37 |
| 2023 | 35654 | 473 | 72 | 35181 | 1.33 |
| 2024 | 35239 | 414 | 61 | 34825 | 1.17 |
| 2025 | 8924 | 84 | 17 | 8840 | 0.94 |

## By flat type

| NORMALIZED_FLAT_TYPE | total_rows | removed_rows | equal_to_half_median_retained | retained_rows | removed_pct |
| --- | --- | --- | --- | --- | --- |
| 1-room | 42 | 1 | 0 | 41 | 2.38 |
| 2-room | 3098 | 153 | 14 | 2945 | 4.94 |
| 3-room | 49024 | 409 | 53 | 48615 | 0.83 |
| 4-room | 54079 | 344 | 83 | 53735 | 0.64 |
| 5-room | 35523 | 169 | 30 | 35354 | 0.48 |
| executive | 8234 | 45 | 9 | 8189 | 0.55 |

## Towns with most flagged rows

| NORMALIZED_TOWN | total_rows | removed_rows | equal_to_half_median_retained | retained_rows | removed_pct |
| --- | --- | --- | --- | --- | --- |
| jurong west | 10482 | 108 | 17 | 10374 | 1.03 |
| woodlands | 7577 | 74 | 10 | 7503 | 0.98 |
| ang mo kio | 8283 | 70 | 4 | 8213 | 0.85 |
| bukit merah | 8270 | 68 | 11 | 8202 | 0.82 |
| yishun | 7913 | 61 | 12 | 7852 | 0.77 |
| bedok | 8907 | 59 | 7 | 8848 | 0.66 |
| tampines | 9942 | 59 | 8 | 9883 | 0.59 |
| toa payoh | 5988 | 52 | 6 | 5936 | 0.87 |
| kallang/whampoa | 5689 | 52 | 10 | 5637 | 0.91 |
| hougang | 7115 | 50 | 10 | 7065 | 0.70 |

## Comparison with previous rules

| previous_rule | previous_removed | both_removed | annual_rule_only | previous_rule_only |
| --- | --- | --- | --- | --- |
| 3-room-or-larger rent <= 1500 | 3768 | 857 | 264 | 2911 |
| BROAD_LOW_RENT_FLAG | 921 | 868 | 253 | 53 |
| CONSERVATIVE_LOW_RENT_FLAG | 305 | 301 | 820 | 4 |

The fixed S$1,500 rule is recomputed directly on the current source. Earlier broad/conservative flags, if present, are read from their existing aligned audit file; they are not part of this rule.

## Lowest rent relative to its group median

| SOURCE_CSV_LINE | RENT_APPROVAL_DATE | TOWN | FLAT_TYPE | MONTHLY_RENT | GROUP_MEDIAN_RENT | HALF_MEDIAN_CUTOFF | RENT_TO_GROUP_MEDIAN | GROUP_N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 102593 | 2023-11 | bukit merah | 5-room | 600 | 4150.00 | 2075.00 | 0.14 | 353 |
| 48009 | 2022-04 | punggol | 2-room | 300 | 2000.00 | 1000.00 | 0.15 | 51 |
| 113982 | 2024-03 | bishan | 5-room | 600 | 4000.00 | 2000.00 | 0.15 | 166 |
| 147332 | 2025-03 | bukit merah | 5-room | 650 | 4200.00 | 2100.00 | 0.15 | 106 |
| 85506 | 2023-06 | clementi | 4-room | 600 | 3700.00 | 1850.00 | 0.16 | 377 |
| 142878 | 2025-01 | hougang | 2-room | 400 | 2400.00 | 1200.00 | 0.17 | 11 |
| 144888 | 2025-02 | jurong west | 5-room | 600 | 3600.00 | 1800.00 | 0.17 | 230 |
| 111306 | 2024-02 | ang mo kio | 5-room | 600 | 3575.00 | 1787.50 | 0.17 | 174 |
| 111366 | 2024-02 | clementi | 5-room | 700 | 4150.00 | 2075.00 | 0.17 | 110 |
| 118437 | 2024-05 | tampines | 5-room | 600 | 3550.00 | 1775.00 | 0.17 | 565 |
| 83842 | 2023-05 | tampines | 5-room | 600 | 3505.00 | 1752.50 | 0.17 | 638 |
| 119320 | 2024-05 | bishan | 4-room | 600 | 3500.00 | 1750.00 | 0.17 | 386 |
| 117451 | 2024-04 | clementi | 4-room | 650 | 3700.00 | 1850.00 | 0.18 | 425 |
| 76263 | 2023-03 | jurong west | 5-room | 600 | 3400.00 | 1700.00 | 0.18 | 894 |
| 81845 | 2023-04 | jurong west | 2-room | 400 | 2250.00 | 1125.00 | 0.18 | 38 |

## What the flags establish

These rents are unusually low for the requested comparison group. As a supplementary check, 289 flagged rows have at least five observations with the same year, town, flat type, street, block, flat model and floor area; 229 of these also fall below half that more local median. This supports reviewing those records, but does not establish an incorrect label or a bedroom-only lease: the data do not identify individual units or contractual terms. The supplementary check uses raw address/model labels and does not alter selection.

Small groups can have unstable medians or miss anomalies when cheap rents dominate. Annual medians can also flag genuine discounted leases and do not account for within-year market changes. No external source verification or model retraining was performed.

This is a retrospective audit using labels from the whole observed year. For model evaluation, recompute the filter using only each training fold and leave validation targets unchanged. Do not calculate training cutoffs using held-out labels. A row audit alone does not establish that removal improves prediction.

## Outputs and reproduction

- `train_filtered.csv`: original columns and original order for retained rows.
- `removed_below_half_median.csv`: original row values, source CSV line (header is line 1), group statistics, and supporting block comparison; sorted by rent/median ascending.
- `all_row_flags.csv`: one row per original record, in original order, with grouping keys, medians, cutoffs and flags.
- `group_medians.csv`: every observed year/town/type median, cutoff, count and removed count.
- `summary_by_year.csv`, `summary_by_flat_type.csv`, `summary_by_town.csv`, `comparison_previous_filters.csv`, `summary.json`: aggregate audit results.

Run from repository root: `.venv/Scripts/python.exe analysis/filter_rent_below_half_median.py`.

The original training file and existing filters/submissions are preserved.
