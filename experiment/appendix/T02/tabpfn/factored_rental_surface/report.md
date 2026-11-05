# Factored dynamic rental-surface experiment

This experiment implements the architecture requested in the longer project plan: **monthly HDB-supported rental surfaces → PCA → temporal factor forecasting → predicted local market rent → CatBoost direct or residual model**. Static HDB-block, MRT, mall, school, and planning features enter CatBoost; they are not PCA channels. Code: [`factored_rental_surface.py`](../../analysis/experiments/factored_rental_surface.py).

## Setup and leakage control

- A **16 × 16** grid over supplied HDB-block coordinates has **115 supported cells**. Four channels represent 3-room, 4-room, 5-room, and executive flats. Rare 1- and 2-room flats use a type-specific 2021 anchor and the common-market movement.
- A Ridge hedonic reference fitted on **2021 only** adjusts rent for flat type/model, floor area, age, and town. For each month and channel, residual rents are smoothed across nearby supported cells with a 1 km Gaussian kernel and a 10-observation flat-type prior. Unobserved cells are never assigned SGD 0.
- PCA is fitted only on months before each forecast origin. We test **3, 5, and 8 factors** with persistence, Ridge AR(1), and an independent damped local-linear-trend Kalman filter. The factors contain only dynamic rental surfaces.
- Hybrid training rows from **January 2022 onward** receive a one-month-ahead PCA/state-space market forecast fitted on strictly earlier months. All validation months receive a multi-step forecast fixed at the fold origin. Validation targets are used only in error calculations and the diagnostic observed-surface plot.
- The hybrid comparison uses **5 PCA factors and the state-space forecast**, a prespecified interpretable configuration. Both direct and residual CatBoost use property, static spatial, and earlier-rent features; the residual model targets `rent - forecast_market_rent`. A CatBoost trained on the same 2022+ subset controls for the loss of 2021 hybrid training rows.

## Surface forecasting

The table averages monthly RMSE on observed cell × flat-type entries with at least five rentals. It is a market-surface diagnostic, not transaction-level RMSE. Across the four fixed-origin windows there are 43 forecast months, including the overlapping 16-month check.

| PCA factors | Explained variance, mean | Reconstruction RMSE, mean | Persistence forecast RMSE | AR(1) forecast RMSE | State-space forecast RMSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 98% | S$49.87 | S$176.72 | S$288.93 | **S$175.24** |
| 5 | 99% | S$44.92 | S$177.58 | S$312.70 | **S$176.17** |
| 8 | 99% | S$38.32 | S$180.22 | S$255.42 | **S$177.99** |

PCA reconstructs smoothed past surfaces well, but better reconstruction with more factors does **not** improve future surface forecasts. The simple AR(1) extrapolates poorly over long horizons. The state-space model improves only slightly over persistence. The full monthly data are in `surface_forecast_scores.csv`.

## Transaction-level results

| Model | Pooled RMSE, three disjoint folds | 2023-01, 6 months | 2023-07, 9 months | 2024-04, 12 months | 2023-12, overlapping 16 months |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full-history CatBoost | 539.29 | 637.41 | **557.35** | 466.23 | 485.86 |
| CatBoost on the hybrid's 2022+ rows | 548.44 | 669.90 | 558.03 | 466.83 | 486.52 |
| CatBoost + forecast market feature | 546.55 | 663.63 | 558.18 | **466.19** | 487.52 |
| **Forecast market + CatBoost residual** | **538.06** | **626.25** | 560.40 | 467.84 | **484.93** |

The residual hybrid improves pooled RMSE over full-history CatBoost by **S$1.23**, but worsens the second and latest disjoint folds. Its largest gain is in the rising-market 2023-01 fold. The direct-feature hybrid is worse pooled. The 16-month fold overlaps the others and is not included in the pooled score. The selected submission candidate remains the simpler full-history CatBoost model because its improvement is more consistent and the residual hybrid's gain is small and regime-dependent.

The hybrid's 2022+ training restriction is material: its matched subset CatBoost is S$9.15 worse pooled than the all-history CatBoost. This comparison isolates some of that restriction, but does not prove how a different reference model or warm-up length would behave. Likewise, one smoothing bandwidth and one grid resolution cannot rule out every spatial representation.

This run covers the longer instruction's core surface/PCA/temporal forecast/transaction hybrid comparison. It does **not** complete its full experiment matrix: 32 × 32 and 48 × 48 grids, every static amenity ablation on these same folds, residual spatial diagnostics, and a hybrid-generated test submission remain unrun. Earlier repository reports contain separate baseline and auxiliary-feature comparisons, but their settings and folds are not all identical to this run. The existing strict submission comes from full-history CatBoost, the selected model, rather than the residual hybrid.

See `hybrid_fold_scores.csv`, `hybrid_pooled_scores.csv`, `hybrid_predictions.csv`, `fit_times.csv`, and `results.json` for exact results and parameters. The figures `pca_4room_component_maps.png`, `pca_factor_paths.png`, and `april_2024_surface_forecast.png` show the learned factors and an observed-versus-forecast 4-room surface; the observed April 2024 surface was used **only** for evaluation.

## Reproduce

```powershell
$env:OMP_NUM_THREADS = '1'
python analysis/compare_factored_rental_surface.py
python analysis/plot_factored_rental_surface.py
python -m unittest discover -s tests -q
```

The training CSV and existing earlier reports remain unchanged. These folds have already been examined repeatedly, so the numbers are development evidence rather than an untouched final test.
