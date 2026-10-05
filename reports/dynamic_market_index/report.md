# Dynamic market-index and full-history tree experiment

## Decision

The selected prediction model is **CatBoost with the existing static spatial and strictly prior rental-history features**. Its pooled RMSE across the three disjoint forward folds is **S$539.29**, versus **S$543.10** for the prior HistGradientBoosting historical model. The separate, overlapping 16-month check is **S$485.86** versus **S$489.60**. These are modest but consistent improvements; CatBoost was not extensively tuned.

The forecast market-index correction **does not pass the plan's grid/PCA gate**. It helps during the 2023 rising-market fold but fails to improve all forward folds. A low-cost **16 × 16 surface** proof of concept also worsened every fold. We subsequently ran the core PCA, state-space factor forecasting, and CatBoost hybrid experiment requested in the original longer plan. Its residual hybrid gains only S$1.23 pooled and worsens two of the three disjoint folds; see the [factored rental-surface report](../factored_rental_surface/report.md). This additional evaluation supports retaining full-history CatBoost as the selected model.

## Full-feature row-model comparison

Each model uses the same property, HDB-block, amenity, and earlier-month history features. The training rows and fixed-origin validation rows are identical. Settings were fixed before scoring these folds; there was no broad search. Lower RMSE is better.

| Model | Pooled RMSE, three disjoint folds | 2023-01, 6 months | 2023-07, 9 months | 2024-04, 12 months | 2023-12, 16-month check |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical HistGradientBoosting | 543.10 | 639.02 | 563.35 | 469.61 | 489.60 |
| Full-history LightGBM | 545.83 | 643.27 | 567.26 | 470.29 | 492.77 |
| **Full-history CatBoost** | **539.29** | **637.41** | **557.35** | **466.23** | **485.86** |

The 16-month fold overlaps the disjoint folds and is never pooled with them. The detailed scores, monthly errors, predictions, settings, and fit times are in `row_model_scores.csv`, `fold_scores.csv`, `monthly_scores.csv`, `validation_predictions.csv`, `results.json`, and `fit_durations.csv`.

CatBoost has lower squared error than HistGradientBoosting in **25 of the 27 validation months**. Resampling those months 10,000 times with seed 42 gives a descriptive 95% interval of **S$2.24–S$5.17** for CatBoost's pooled RMSE gain (observed gain: S$3.81). Months are few and adjacent, so this interval does not turn the examined folds into an untouched test.

## Forecast market-index test

A Ridge model fit on 2021 property, flat-model, size, age, and town information defines a fixed composition reference. Monthly residuals from that reference form flat-type and shrunken planning-area × flat-type indices. At each validation origin, the index forecast sees only fitting-period rents. Persistence, a recent-three-month mean, and a damped trend are compared without updating from validation labels. Each non-persistence forecast changes a CatBoost row prediction by its forecast index movement since the origin.

| CatBoost variant | Pooled RMSE | 2023-01 | 2023-07 | 2024-04 | 16-month check |
| --- | ---: | ---: | ---: | ---: | ---: |
| No market correction | 539.29 | 637.41 | 557.35 | 466.23 | 485.86 |
| Flat-type, damped trend | **537.44** | **625.96** | 560.32 | 466.50 | **485.27** |
| Area × flat-type, damped trend | 537.58 | 626.54 | 559.63 | 467.10 | 485.63 |
| Flat-type, recent-three mean | 544.08 | 654.30 | 558.11 | 466.37 | 485.34 |
| Area × flat-type, recent-three mean | 543.91 | 654.22 | 557.05 | 466.93 | 485.66 |

The best pooled index adjustment gains S$1.85 against CatBoost, below the prespecified roughly 1% practical threshold of S$5.39. It worsens the second and latest disjoint folds. The market-index forecast itself also fails to beat persistence in every fold; see `index_forecast_scores.csv`. Thus a fine rental grid is unlikely to justify its much larger search and imputation cost as the next performance experiment.

## Recent rental-history ablation

The existing historical features use expanding means, counts, last-month means, and recency. The ablation adds strictly earlier 3-, 6-, and 12-month block × flat-type and area × flat-type means/counts, plus a block mean shrunk toward the area mean. For validation, these windows are frozen at the fold origin.

| CatBoost variant | Pooled RMSE | 2023-01 | 2023-07 | 2024-04 | 16-month check |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing history | 539.29 | 637.41 | **557.35** | **466.23** | 485.86 |
| + recent windows | **539.20** | **632.24** | 559.43 | 467.64 | **485.72** |

The S$0.09 pooled difference is too small and inconsistent to justify keeping the extra features in the selected model. Full fold and monthly results are in `recent_history_fold_scores.csv`, `recent_history_monthly_scores.csv`, and `recent_history_predictions.csv`.

## Coarse rental-surface proof of concept

The 16 × 16 grid is fixed from the supplied HDB-block coordinate bounds. It has **115 HDB-supported cells**. For each pre-origin month, cell, and flat type, the experiment forms a premium from rent residuals relative to the contemporaneous flat-type index. A shrunken slope between the last two three-month windows forecasts a change in that premium; validation rents never enter either window. This is added to CatBoost predictions, with and without the national flat-type damped trend. The surface is one coarse, composition-adjusted representation; it is not a test of all grid sizes or a learned PCA surface.

| Variant | Pooled RMSE | 2023-01 | 2023-07 | 2024-04 | 16-month check |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost | **539.29** | **637.41** | **557.35** | **466.23** | **485.86** |
| + local 16 × 16 change | 543.88 | 641.40 | 562.85 | 470.50 | 494.37 |
| + flat-type damped trend | 537.44 | 625.96 | 560.32 | 466.50 | 485.27 |
| + flat-type trend and local change | 542.07 | 629.88 | 565.98 | 470.74 | 493.72 |

About **27%** of supported cell × common-flat-type × month slots are empty; among observed slots the median is only **5–6 transactions**. Two prior three-month windows cover roughly 99% of validation rows, but that high coverage does not make local slopes reliable. The consistent regression and limited per-cell sample size are reasons to stop before a larger grid/PCA search. See `surface_occupancy.csv`, `surface_fold_scores.csv`, `surface_pooled_scores.csv`, and `surface_predictions.csv`.

## Limits and reproduction

The tree settings, the 2021 composition reference, and the index forecast constants are deliberately simple. All validation periods have already been examined during this project; they are development evidence rather than an untouched final test. The supplied amenity files are current snapshots without historical opening dates. The strict candidate submission in `submission_strict_feb2025.csv` uses **146,968** rentals through February 2025 for **all 50,000** test predictions, excluding March 2025 training labels because test also starts in March. It is a candidate prediction file, not a measured test-set score.

Run from the repository root, using an environment with `requirements-model.txt` installed:

```powershell
$env:OMP_NUM_THREADS = '1'
python analysis/compare_dynamic_market_index.py
python analysis/compare_recent_history.py
python analysis/compare_coarse_rental_surface.py
python analysis/train_dynamic_market_submission.py
python -m unittest discover -s tests -q
```

The single-thread setting avoids a Windows joblib pipe restriction encountered in this environment; LightGBM and CatBoost still use their explicitly configured threads. Raw CSV files and earlier reports are unchanged.
