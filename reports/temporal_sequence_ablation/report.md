# Rental, stock-index and COE temporal processing ablation

The earlier national-index study did compare Singapore mean-rent changes with overall stock returns: same-month correlation 0.005; prior-month correlation -0.253 (95% month-block interval -0.517 to 0.067). The worksite study instead concerns local relative rents. Neither earlier study used LSTM.

## Earlier-selected models and later monthly-mean results

| target | scenario | selection | model | development_normalized_mse | outer_rmse_sgd | delta_rmse_sgd | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adjusted_mean_rent | fixed_origin | overall | persistence__rent__w12 | 1.000 | 45.388 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | fixed_origin | rent_only | persistence__rent__w12 | 1.000 | 45.388 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | fixed_origin | joint | ridge_lags__rent_stock_coe__w12 | 8.740 | 41.974 | -3.414 | -13.528 | 8.228 |
| adjusted_mean_rent | fixed_origin | joint_lstm | lstm__rent_stock_coe__w3 | 20.116 | 31.828 | -13.560 | -17.748 | -0.587 |
| adjusted_mean_rent | rolling | overall | persistence__rent__w12 | 1.000 | 37.746 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | rolling | rent_only | persistence__rent__w12 | 1.000 | 37.746 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | rolling | joint | ridge_lags__rent_stock_coe__w12 | 2.712 | 79.764 | 42.018 | 12.544 | 75.525 |
| adjusted_mean_rent | rolling | joint_lstm | lstm__rent_stock_coe__w3 | 10.314 | 61.503 | 23.757 | -0.271 | 40.743 |
| mean_rent | fixed_origin | overall | persistence__rent__w12 | 1.000 | 42.642 | 0.000 | 0.000 | 0.000 |
| mean_rent | fixed_origin | rent_only | persistence__rent__w12 | 1.000 | 42.642 | 0.000 | 0.000 | 0.000 |
| mean_rent | fixed_origin | joint | ridge_lags__rent_stock_coe__w12 | 10.031 | 42.432 | -0.210 | -9.372 | 15.024 |
| mean_rent | fixed_origin | joint_lstm | lstm__rent_stock_coe__w6 | 21.530 | 38.571 | -4.071 | -13.410 | 13.825 |
| mean_rent | rolling | overall | persistence__rent__w12 | 1.000 | 39.546 | 0.000 | 0.000 | 0.000 |
| mean_rent | rolling | rent_only | persistence__rent__w12 | 1.000 | 39.546 | 0.000 | 0.000 | 0.000 |
| mean_rent | rolling | joint | ridge_lags__rent_stock_coe__w12 | 3.139 | 82.513 | 42.968 | 13.839 | 74.657 |
| mean_rent | rolling | joint_lstm | lstm__rent_stock_coe__w3 | 11.224 | 67.721 | 28.175 | 3.673 | 43.747 |

RMSE and deltas are SGD errors of the monthly arithmetic mean, not individual rental errors. Deltas compare persistence on identical cases. Rolling results pool 1/3/6-month horizons with equal target-month weight; fixed-origin results cover ten target months from May 2024 information. Development ranking gives each horizon equal weight after normalizing MSE to persistence. Every model/window is in all_scores.csv and rankings.csv; selection uses only development errors.

## Which information and processing help?

### mean_rent, rolling

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| ewma | 1.820 | nan | nan | nan |
| lstm | 7.337 | 1.464 | 9.097 | 2.391 |
| mlp | 24.964 | 3.775 | 24.197 | 1.643 |
| momentum3 | 2.233 | nan | nan | nan |
| persistence | 1.000 | nan | nan | nan |
| ridge_lags | 24.756 | 3.897 | 25.338 | 3.946 |
| ridge_summary | 16.807 | 3.998 | 9.919 | 2.594 |

### mean_rent, fixed_origin

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| ewma | 0.199 | nan | nan | nan |
| lstm | 91.474 | 0.737 | 29.607 | 1.484 |
| mlp | 204.064 | 19.738 | 263.101 | 5.217 |
| momentum3 | 0.252 | nan | nan | nan |
| persistence | 1.000 | nan | nan | nan |
| ridge_lags | 225.116 | 2.001 | 215.941 | 2.264 |
| ridge_summary | 132.089 | 18.425 | 66.950 | 11.691 |

### adjusted_mean_rent, rolling

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| ewma | 1.663 | nan | nan | nan |
| lstm | 7.943 | 1.334 | 7.449 | 2.173 |
| mlp | 23.195 | 3.914 | 23.021 | 1.274 |
| momentum3 | 1.906 | nan | nan | nan |
| persistence | 1.000 | nan | nan | nan |
| ridge_lags | 24.443 | 4.219 | 24.838 | 4.263 |
| ridge_summary | 15.390 | 3.745 | 8.833 | 2.334 |

### adjusted_mean_rent, fixed_origin

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| ewma | 0.171 | nan | nan | nan |
| lstm | 44.645 | 0.191 | 10.902 | 1.278 |
| mlp | 116.772 | 8.445 | 147.138 | 1.457 |
| momentum3 | 0.167 | nan | nan | nan |
| persistence | 1.000 | nan | nan | nan |
| ridge_lags | 139.215 | 1.057 | 130.696 | 1.246 |
| ridge_summary | 71.124 | 11.332 | 39.470 | 7.781 |

Values above are later MSE / persistence MSE (below 1 improves persistence), after choosing each cell's window on earlier data. Compare both within a processor and against the rent-only control; a forced best joint model need not beat a simple rent baseline.

## Transfer to individual-rental prediction

| scenario | candidate | model | gamma | baseline_rmse_sgd | corrected_rmse_sgd | delta_rmse_sgd | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_origin | joint | ridge_lags__rent_stock_coe__w12 | 0.318 | 471.631 | 470.375 | -1.256 | -2.402 | 0.191 |
| fixed_origin | joint_lstm | lstm__rent_stock_coe__w3 | 0.117 | 471.631 | 470.905 | -0.726 | -0.979 | -0.493 |
| fixed_origin | persistence | persistence__rent__w12 | 0.000 | 471.631 | 471.631 | 0.000 | 0.000 | 0.000 |
| fixed_origin | rent_only | persistence__rent__w12 | 0.000 | 471.631 | 471.631 | 0.000 | 0.000 | 0.000 |
| rolling | joint | ridge_lags__rent_stock_coe__w12 | 0.494 | 471.631 | 470.049 | -1.582 | -2.553 | -0.662 |
| rolling | joint_lstm | lstm__rent_stock_coe__w3 | 0.411 | 471.631 | 470.271 | -1.360 | -1.699 | -0.981 |
| rolling | persistence | persistence__rent__w12 | 0.754 | 471.631 | 469.618 | -2.013 | -2.829 | -1.038 |
| rolling | rent_only | persistence__rent__w12 | 0.754 | 471.631 | 469.618 | -2.013 | -2.829 | -1.038 |

The same 29-feature HistGradientBoosting baseline is frozen in each row fold. An additive correction is gamma times the predicted adjusted national-index movement since the fold cutoff. Gamma is constrained to [0,1] and fitted on July 2023–May 2024 row errors only. Two earlier row folds precede the June 2024–March 2025 comparison. The input processor is selected by earlier aggregate scores. This is a matched tree transfer, not a measured TabPFN improvement. Monthly updates may use rents observed after the tree fit; fixed-origin paths cannot.

## Locked-model sensitivity checks

| target | scenario | check | model | outer_rmse_sgd |
| --- | --- | --- | --- | --- |
| adjusted_mean_rent | fixed_origin | seed43 | lstm__rent_stock_coe__w3 | 25.920 |
| adjusted_mean_rent | fixed_origin | seed44 | lstm__rent_stock_coe__w3 | 22.350 |
| adjusted_mean_rent | fixed_origin | equal_weight_index | lstm__rent_stock_coe__w3 | 43.636 |
| adjusted_mean_rent | fixed_origin | extra_month_delay | lstm__rent_stock_coe__w3 | 99.351 |
| adjusted_mean_rent | rolling | seed43 | lstm__rent_stock_coe__w3 | 62.521 |
| adjusted_mean_rent | rolling | seed44 | lstm__rent_stock_coe__w3 | 60.278 |
| adjusted_mean_rent | rolling | equal_weight_index | lstm__rent_stock_coe__w3 | 69.844 |
| adjusted_mean_rent | rolling | extra_month_delay | lstm__rent_stock_coe__w3 | 73.703 |
| mean_rent | fixed_origin | seed43 | lstm__rent_stock_coe__w6 | 25.192 |
| mean_rent | fixed_origin | seed44 | lstm__rent_stock_coe__w6 | 25.539 |
| mean_rent | fixed_origin | equal_weight_index | lstm__rent_stock_coe__w6 | 34.352 |
| mean_rent | fixed_origin | extra_month_delay | lstm__rent_stock_coe__w6 | 49.099 |
| mean_rent | rolling | seed43 | lstm__rent_stock_coe__w3 | 72.037 |
| mean_rent | rolling | seed44 | lstm__rent_stock_coe__w3 | 69.661 |
| mean_rent | rolling | equal_weight_index | lstm__rent_stock_coe__w3 | 74.614 |
| mean_rent | rolling | extra_month_delay | lstm__rent_stock_coe__w3 | 78.578 |

Seeds 43/44, replacing cap-seeded returns with equal-weight matched returns, and one extra month of input delay are checks of the earlier-selected joint LSTM; no new window is selected. The delay changes training and query information cutoffs and anchors to the last available rent. It approximates a full-month publication lag rather than a precise release calendar.

## Design, availability and limits

- 51 labeled months (January 2021–March 2025), from all 150,000 rental rows. Raw arithmetic means and a 2021-only fixed-property-model adjustment are separate responses. The reference model sees no later rents and is held fixed. Changing property premiums can remain.
- Primary stock series is the existing historical-cap-seeded 22-name Singapore return proxy, not official STI. Matched stocks represent 81.56% of the source ETF portfolio. Later adjusted-price snapshots and fixed constituents can contain survivorship/revision effects.
- COE A/C premium log levels and log changes; stocks use monthly log returns; rent uses log changes. Common known calendar features and training-origin start January 2022. Information sets are rent, rent+stock, rent+COE and all three. Windows 3/6/12 apply to lag regression, MLP and LSTM.
- All 43 specifications run on the same evaluation cases per target/scenario. Training labels are masked until their endpoint is observed. Neural targets are mean monthly log growth for ten direct horizons; loss averages masked squared standardized errors equally across horizons. Outputs share an encoder, not independent new market observations. Ridge uses a separate mature-label head per horizon. No realized future exogenous path is supplied to a multi-step forecast.
- Small MLP/LSTM: hidden size 8, 100 fixed epochs, AdamW .01, weight decay .05, gradient norm 1, seed 42, CPU deterministic execution. MLP uses the same ordered values and calendar inputs. This is a bounded architecture ablation, not exhaustive neural architecture search.
- Development: rolling target months July 2023–May 2024; fixed ten-month paths from Jan/Apr/Jul 2023. Later comparison: June 2024–March 2025. Earlier fixed paths overlap. This later period has already been examined elsewhere; it is not an untouched test. Few independent months and many candidates mean rankings can be unstable. No result establishes causality or global optimality.
- Paired three-month moving-block intervals retain target-month dependence, and row intervals retain rental counts. They are conditional on fitted/selected models, omit fitting/selection uncertainty, and are not adjusted for model searches. Multi-horizon rolling errors also overlap beyond three months.
- Test rents remain unavailable. A ten-month backtest cannot validate every forecast through July 2026. Existing submissions are unchanged. No neural or macro feature is promoted automatically.

## Reproduce

`.venv/Scripts/python.exe analysis/compare_temporal_sequence_ablation.py`

Use `--resume` for hash-checked candidate caches, `--skip-row-transfer` for aggregate analysis only, or `--smoke` with a different output directory for a quick execution check. results.json contains versions, settings, hashes, elapsed times, source coverage and cutoff checks. forecast_predictions.csv records every origin, endpoint and latest permissible input/label date.

References: [previous national-index study](../stock_index_regions/report.md), [PyTorch LSTM](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html), [rolling-origin evaluation](https://otexts.com/fpp3/tscv.html).
