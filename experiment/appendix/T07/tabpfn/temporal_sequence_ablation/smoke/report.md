# Rental, stock-index and COE temporal processing ablation

The earlier national-index study did compare Singapore mean-rent changes with overall stock returns: same-month correlation 0.005; prior-month correlation -0.253 (95% month-block interval -0.517 to 0.067). The worksite study instead concerns local relative rents. Neither earlier study used LSTM.

## Earlier-selected models and later monthly-mean results

| target | scenario | selection | model | development_normalized_mse | outer_rmse_sgd | delta_rmse_sgd | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adjusted_mean_rent | fixed_origin | overall | persistence__rent__w12 | 1.000 | 45.388 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | fixed_origin | rent_only | persistence__rent__w12 | 1.000 | 45.388 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | fixed_origin | joint | ridge_summary__rent_stock_coe__w12 | 13.406 | 115.199 | 69.811 | 9.150 | 110.861 |
| adjusted_mean_rent | fixed_origin | joint_lstm | lstm__rent_stock_coe__w3 | 19.466 | 224.054 | 178.666 | 73.875 | 250.777 |
| adjusted_mean_rent | rolling | overall | persistence__rent__w12 | 1.000 | 37.746 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | rolling | rent_only | persistence__rent__w12 | 1.000 | 37.746 | 0.000 | 0.000 | 0.000 |
| adjusted_mean_rent | rolling | joint | ridge_summary__rent_stock_coe__w12 | 6.329 | 67.321 | 29.575 | 8.062 | 57.324 |
| adjusted_mean_rent | rolling | joint_lstm | lstm__rent_stock_coe__w3 | 14.511 | 139.769 | 102.023 | 89.200 | 119.496 |
| mean_rent | fixed_origin | overall | persistence__rent__w12 | 1.000 | 42.642 | 0.000 | 0.000 | 0.000 |
| mean_rent | fixed_origin | rent_only | persistence__rent__w12 | 1.000 | 42.642 | 0.000 | 0.000 | 0.000 |
| mean_rent | fixed_origin | joint | ridge_summary__rent_stock_coe__w12 | 14.516 | 120.926 | 78.284 | 14.566 | 121.420 |
| mean_rent | fixed_origin | joint_lstm | lstm__rent_stock_coe__w3 | 20.788 | 231.113 | 188.471 | 81.292 | 263.978 |
| mean_rent | rolling | overall | persistence__rent__w12 | 1.000 | 39.546 | 0.000 | 0.000 | 0.000 |
| mean_rent | rolling | rent_only | persistence__rent__w12 | 1.000 | 39.546 | 0.000 | 0.000 | 0.000 |
| mean_rent | rolling | joint | ridge_summary__rent_stock_coe__w12 | 6.029 | 72.575 | 33.030 | 12.168 | 60.672 |
| mean_rent | rolling | joint_lstm | lstm__rent_stock_coe__w3 | 13.632 | 143.221 | 103.675 | 89.128 | 124.604 |

RMSE and deltas are SGD errors of the monthly arithmetic mean, not individual rental errors. Deltas compare persistence on identical cases. Rolling results pool 1/3/6-month horizons with equal target-month weight; fixed-origin results cover ten target months from May 2024 information. Development ranking gives each horizon equal weight after normalizing MSE to persistence. Every model/window is in all_scores.csv and rankings.csv; selection uses only development errors.

## Which information and processing help?

### mean_rent, rolling

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| lstm | nan | nan | nan | 10.463 |
| mlp | nan | nan | nan | 8.617 |
| persistence | 1.000 | nan | nan | nan |
| ridge_summary | 16.807 | 3.998 | 9.919 | 2.594 |

### mean_rent, fixed_origin

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| lstm | nan | nan | nan | 50.500 |
| mlp | nan | nan | nan | 55.550 |
| persistence | 1.000 | nan | nan | nan |
| ridge_summary | 132.089 | 18.425 | 66.950 | 11.691 |

### adjusted_mean_rent, rolling

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| lstm | nan | nan | nan | 10.963 |
| mlp | nan | nan | nan | 8.848 |
| persistence | 1.000 | nan | nan | nan |
| ridge_summary | 15.390 | 3.745 | 8.833 | 2.334 |

### adjusted_mean_rent, fixed_origin

| method | rent | rent_coe | rent_stock | rent_stock_coe |
| --- | --- | --- | --- | --- |
| lstm | nan | nan | nan | 33.133 |
| mlp | nan | nan | nan | 34.064 |
| persistence | 1.000 | nan | nan | nan |
| ridge_summary | 71.124 | 11.332 | 39.470 | 7.781 |

Values above are later MSE / persistence MSE (below 1 improves persistence), after choosing each cell's window on earlier data. Compare both within a processor and against the rent-only control; a forced best joint model need not beat a simple rent baseline.

## Transfer to individual-rental prediction

## Locked-model sensitivity checks

| target | scenario | check | model | outer_rmse_sgd |
| --- | --- | --- | --- | --- |
| adjusted_mean_rent | fixed_origin | seed43 | lstm__rent_stock_coe__w3 | 234.320 |
| adjusted_mean_rent | fixed_origin | seed44 | lstm__rent_stock_coe__w3 | 225.410 |
| adjusted_mean_rent | fixed_origin | equal_weight_index | lstm__rent_stock_coe__w3 | 223.798 |
| adjusted_mean_rent | fixed_origin | extra_month_delay | lstm__rent_stock_coe__w3 | 271.071 |
| adjusted_mean_rent | rolling | seed43 | lstm__rent_stock_coe__w3 | 140.003 |
| adjusted_mean_rent | rolling | seed44 | lstm__rent_stock_coe__w3 | 131.438 |
| adjusted_mean_rent | rolling | equal_weight_index | lstm__rent_stock_coe__w3 | 139.543 |
| adjusted_mean_rent | rolling | extra_month_delay | lstm__rent_stock_coe__w3 | 179.686 |
| mean_rent | fixed_origin | seed43 | lstm__rent_stock_coe__w3 | 239.927 |
| mean_rent | fixed_origin | seed44 | lstm__rent_stock_coe__w3 | 233.396 |
| mean_rent | fixed_origin | equal_weight_index | lstm__rent_stock_coe__w3 | 230.849 |
| mean_rent | fixed_origin | extra_month_delay | lstm__rent_stock_coe__w3 | 273.825 |
| mean_rent | rolling | seed43 | lstm__rent_stock_coe__w3 | 141.839 |
| mean_rent | rolling | seed44 | lstm__rent_stock_coe__w3 | 133.920 |
| mean_rent | rolling | equal_weight_index | lstm__rent_stock_coe__w3 | 142.677 |
| mean_rent | rolling | extra_month_delay | lstm__rent_stock_coe__w3 | 183.580 |

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
