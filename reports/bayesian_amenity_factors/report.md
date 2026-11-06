# Bayesian amenity and location factors

Gaussian hierarchical additive model with proper grouped priors, a monthly random-walk effect, and an exact conditional posterior. Lower fixed-origin RMSE is better. The 16-month fold overlaps the earlier validation windows and is a separate horizon check.

| Validation start | Historical tree | Property/location/time | + amenities | + town MRT slope | + history |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023-01 | 639.02 | 661.17 | 660.34 | 660.28 | 644.59 |
| 2023-07 | 563.35 | 580.48 | 579.28 | 579.11 | 573.30 |
| 2024-04 | 469.61 | 493.85 | 492.54 | 492.24 | 484.64 |
| 2023-12 | 489.60 | 513.97 | 512.87 | 512.71 | 501.30 |

## Model and interpretation

Adding amenities improves this Bayesian specification by S$1.31 at most across the four folds, but its best 12-month variant remains S$15.03 above the historical tree RMSE. The history variant's nominal 90% intervals cover only 69.6% of sampled rows in the 2023-01 rising-market fold, indicating poor calibration there.

Property terms include flat type/model, floor area, age, and block height. Town and subzone intercepts have shrinkage priors. MRT and mall distance have separate piecewise-linear curves, along with 1 km facility counts. The town MRT variant adds pooled town-specific slopes. The history variant includes earlier-month rent summaries and recency.

A monthly random-walk prior pools neighboring months and carries the last fitted market level forward. Seasonality is explicit. All preprocessing and posterior fitting use only pre-origin data. Posterior-mean rents are scored in SGD; 90% predictive interval coverage uses a reproducible sample of 3,000 validation rows per fold.

The amenity contrast compares fitted rent at 0.25 km and 1.0 km while holding other predictors fixed; for the town-slope variant it reports the shared MRT curve without a particular town deviation. It is an association rather than a causal effect; facility locations are current snapshots without historical opening dates. The Gaussian observation model treats rows as conditionally independent, so posterior intervals may understate repeated-block dependence. Prior strengths are fixed before validation, and uncertainty does not integrate over them. The forward-fold coverage numbers take precedence over nominal interval labels when judging reliability.

## Amenity contrasts (SGD; near minus far)

| Validation start | Variant | Factor | Estimate | 90% interval |
| --- | --- | --- | ---: | ---: |
| 2023-01 | plus_amenities | mrt | +78.0 | [+66.1, +90.0] |
| 2023-01 | plus_amenities | mall | +55.4 | [+43.1, +67.8] |
| 2023-01 | plus_town_mrt_slope | mrt | +70.9 | [+43.0, +98.8] |
| 2023-01 | plus_town_mrt_slope | mall | +58.9 | [+46.2, +71.6] |
| 2023-01 | plus_history | mrt | +79.0 | [+67.0, +90.9] |
| 2023-01 | plus_history | mall | +53.3 | [+41.0, +65.5] |
| 2023-07 | plus_amenities | mrt | +75.0 | [+62.6, +87.4] |
| 2023-07 | plus_amenities | mall | +60.0 | [+47.2, +72.8] |
| 2023-07 | plus_town_mrt_slope | mrt | +72.4 | [+41.1, +103.8] |
| 2023-07 | plus_town_mrt_slope | mall | +65.4 | [+52.2, +78.6] |
| 2023-07 | plus_history | mrt | +77.8 | [+65.4, +90.1] |
| 2023-07 | plus_history | mall | +56.9 | [+44.2, +69.6] |
| 2024-04 | plus_amenities | mrt | +76.6 | [+64.9, +88.4] |
| 2024-04 | plus_amenities | mall | +66.3 | [+54.2, +78.3] |
| 2024-04 | plus_town_mrt_slope | mrt | +73.8 | [+41.3, +106.3] |
| 2024-04 | plus_town_mrt_slope | mall | +73.4 | [+61.0, +85.9] |
| 2024-04 | plus_history | mrt | +80.0 | [+68.3, +91.7] |
| 2024-04 | plus_history | mall | +63.2 | [+51.3, +75.2] |
| 2023-12 | plus_amenities | mrt | +77.0 | [+64.8, +89.2] |
| 2023-12 | plus_amenities | mall | +63.8 | [+51.3, +76.3] |
| 2023-12 | plus_town_mrt_slope | mrt | +76.5 | [+44.2, +108.8] |
| 2023-12 | plus_town_mrt_slope | mall | +69.9 | [+57.0, +82.9] |
| 2023-12 | plus_history | mrt | +80.0 | [+67.9, +92.1] |
| 2023-12 | plus_history | mall | +60.8 | [+48.4, +73.2] |

See `fold_metrics.csv`, `monthly_metrics.csv`, `amenity_effects.csv`, and `results.json` for complete results and priors. Run `python analysis/compare_bayesian_amenity_factors.py` from the repository root.
