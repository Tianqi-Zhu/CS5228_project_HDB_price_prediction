# Focused temporal and macro feature ensembles

Four new frozen TabPFN views were defined before scoring. All retain the original raw targets, residual baselines, contexts, chronology and native-default two-estimator inference. No fine-tuning occurs.

## Feature definitions

- **relative_history**: Replace19 absolute historical/forecast rent levels by level minus original per-row residual baseline;77 features.
- **horizon_dynamics**: Retain77 plus log horizon, type momentum x log horizon, horizon-decayed local/block premiums, area momentum and block support;83 features.
- **coe_origin**: Retain77 plus category A/C log premiums,3-month log changes,bid pressures,two damped-horizon momentum interactions,missing flag;86 features.
- **stocks_origin**: Retain77 plus .SI basket1/3-month returns,3-month volatility,coverage,damped-horizon momentum,missing flag;83 features.

COE/stock inputs freeze at each original row's forecast origin with one extra month of publication buffer. A June2024 query therefore reads market data through April2024 even for March2025 targets. No realized target-month macro data enters. Stocks are the supplied .SI median-return basket, not the STI. Startup NaN and explicit missing flags are retained; snapshot membership/adjusted-close vintages remain limitations.

Relative histories use the original per-row forecast baseline, never observed rent. Horizon interactions are deterministic transforms of existing causal features; they add representation, not new observations.

## Main development results

| model | rmse_sgd | mae_sgd | bias_sgd | rmse_difference_sgd | rmse_improvement_pct |
| --- | --- | --- | --- | --- | --- |
| baseline | 455.3959 | 327.7937 | 10.7922 | 0.0000 | 0.0000 |
| relative_history | 455.3697 | 329.0421 | 0.8419 | -0.0262 | 0.0058 |
| horizon_dynamics | 457.0372 | 327.6829 | 20.9593 | 1.6413 | -0.3604 |
| coe_origin | 458.8325 | 326.0308 | 51.5737 | 3.4365 | -0.7546 |
| stocks_origin | 456.7165 | 326.1449 | 32.7512 | 1.3205 | -0.2900 |
| dynamics_optimized | 455.3697 | 329.0421 | 0.8419 | -0.0262 | 0.0058 |
| macro_optimized | 455.3553 | 327.2495 | 14.5348 | -0.0406 | 0.0089 |
| all_optimized | 455.3697 | 329.0421 | 0.8419 | -0.0262 | 0.0058 |
| blend_relative_history | 455.3697 | 329.0421 | 0.8419 | -0.0262 | 0.0058 |
| blend_horizon_dynamics | 455.3959 | 327.7937 | 10.7922 | 0.0000 | 0.0000 |
| blend_coe_origin | 455.3705 | 327.3929 | 13.7160 | -0.0255 | 0.0056 |
| blend_stocks_origin | 455.3577 | 327.4267 | 13.1585 | -0.0382 | 0.0084 |
| dynamics_equal | 455.2272 | 327.4871 | 10.8645 | -0.1687 | 0.0371 |
| macro_equal | 456.0978 | 325.7602 | 31.7057 | 0.7018 | -0.1541 |
| all_equal | 455.5452 | 326.2327 | 23.3837 | 0.1493 | -0.0328 |
| prior_temporal | 455.8041 | 329.2373 | 3.5155 | 0.4082 | -0.0896 |
| prior_standard6 | 455.2152 | 327.0914 | 15.1509 | -0.1808 | 0.0397 |
| prior_ensemble | 455.1719 | 328.2022 | 5.7634 | -0.2241 | 0.0492 |

## Earlier-only ensemble weights

| origin | model | fit_label_max | baseline | relative_history | horizon_dynamics | coe_origin | stocks_origin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-06 | dynamics_optimized | 2024-04 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2024-06 | macro_optimized | 2024-04 | 0.8625 | 0.0000 | 0.0000 | 0.0384 | 0.0991 |
| 2024-06 | all_optimized | 2024-04 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2024-06 | blend_relative_history | 2024-04 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2024-06 | blend_horizon_dynamics | 2024-04 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2024-06 | blend_coe_origin | 2024-04 | 0.9283 | 0.0000 | 0.0000 | 0.0717 | 0.0000 |
| 2024-06 | blend_stocks_origin | 2024-04 | 0.8922 | 0.0000 | 0.0000 | 0.0000 | 0.1078 |

Weights minimize original-SGD MSE with nonnegative sum-to-one constraints. Main weights use the two disjoint earlier2000-query windows, through April2024. September2022 uses baseline-only cold start; July2023 uses only the September2022 window. No outer labels select weights or a winning specification.

## Chronological transfer

| origin | model | rmse_sgd | mae_sgd | rmse_difference_sgd |
| --- | --- | --- | --- | --- |
| 2022-09 | baseline | 635.0860 | 512.0524 | 0.0000 |
| 2022-09 | dynamics_optimized | 635.0860 | 512.0524 | 0.0000 |
| 2022-09 | macro_optimized | 635.0860 | 512.0524 | 0.0000 |
| 2022-09 | all_optimized | 635.0860 | 512.0524 | 0.0000 |
| 2022-09 | prior_standard6 | 634.6037 | 511.3751 | -0.4823 |
| 2023-07 | baseline | 569.4795 | 430.4276 | 0.0000 |
| 2023-07 | dynamics_optimized | 565.7814 | 426.4944 | -3.6981 |
| 2023-07 | macro_optimized | 578.4417 | 432.3002 | 8.9622 |
| 2023-07 | all_optimized | 576.6363 | 431.2290 | 7.1568 |
| 2023-07 | prior_standard6 | 563.3067 | 426.3940 | -6.1728 |
| 2024-06 | baseline | 455.3959 | 327.7937 | 0.0000 |
| 2024-06 | dynamics_optimized | 455.3697 | 329.0421 | -0.0262 |
| 2024-06 | macro_optimized | 455.3553 | 327.2495 | -0.0406 |
| 2024-06 | all_optimized | 455.3697 | 329.0421 | -0.0262 |
| 2024-06 | prior_standard6 | 455.2152 | 327.0914 | -0.1808 |

## Paired descriptive uncertainty

| cluster | model | reference | replicates | lower95_sgd | upper95_sgd |
| --- | --- | --- | --- | --- | --- |
| month | dynamics_optimized | baseline | 3000 | -0.7337 | 0.6845 |
| month | dynamics_optimized | prior_standard6 | 3000 | -0.6628 | 1.1824 |
| month | macro_optimized | baseline | 3000 | -0.2035 | 0.0924 |
| month | macro_optimized | prior_standard6 | 3000 | -0.1974 | 0.4967 |
| month | all_optimized | baseline | 3000 | -0.7337 | 0.6845 |
| month | all_optimized | prior_standard6 | 3000 | -0.6628 | 1.1824 |
| building | dynamics_optimized | baseline | 3000 | -0.5357 | 0.4740 |
| building | dynamics_optimized | prior_standard6 | 3000 | -0.3501 | 0.6502 |
| building | macro_optimized | baseline | 3000 | -0.1158 | 0.0339 |
| building | macro_optimized | prior_standard6 | 3000 | -0.0860 | 0.3780 |
| building | all_optimized | baseline | 3000 | -0.5357 | 0.4740 |
| building | all_optimized | prior_standard6 | 3000 | -0.3501 | 0.6502 |

3000 paired bootstrap replicates, seed42, separately by month/building. Ten adjacent month clusters, repeated development inspection and unaccounted weight-estimation uncertainty limit interpretation.

## Inference cost

| origin | model | fit_seconds | predict_seconds | peak_vram_gib | internal_estimators |
| --- | --- | --- | --- | --- | --- |
| 2024-06 | baseline | 25.7085 | 115.5391 | 9.1236 | 2 |
| 2024-06 | relative_history | 26.0459 | 116.5893 | 9.1236 | 2 |
| 2024-06 | horizon_dynamics | 26.7626 | 116.6427 | 9.1275 | 2 |
| 2024-06 | coe_origin | 27.1998 | 117.6294 | 9.1296 | 2 |
| 2024-06 | stocks_origin | 26.9988 | 118.1173 | 9.1275 | 2 |
| 2024-06 | dynamics_optimized | 26.0459 | 116.5893 | 9.1236 | 2 |
| 2024-06 | macro_optimized | 79.9071 | 351.2858 | 9.1296 | 6 |
| 2024-06 | all_optimized | 26.0459 | 116.5893 | 9.1236 | 2 |
| 2024-06 | blend_relative_history | 26.0459 | 116.5893 | 9.1236 | 2 |
| 2024-06 | blend_horizon_dynamics | 25.7085 | 115.5391 | 9.1236 | 2 |
| 2024-06 | blend_coe_origin | 52.9082 | 233.1685 | 9.1296 | 4 |
| 2024-06 | blend_stocks_origin | 52.7073 | 233.6564 | 9.1275 | 4 |
| 2024-06 | dynamics_equal | 78.5170 | 348.7711 | 9.1275 | 6 |
| 2024-06 | macro_equal | 79.9071 | 351.2858 | 9.1296 | 6 |
| 2024-06 | all_equal | 132.7156 | 584.5178 | 9.1296 | 10 |
| 2024-06 | prior_temporal | 26.4960 | 116.7174 | 9.1272 | 2 |
| 2024-06 | prior_standard6 | 77.2196 | 346.8563 | 9.1236 | 6 |

Dynamics and macro triples each have at most six views, comparable to the reused same-feature seeds42/43/44 six-view control. The combined five-member ensemble uses up to ten views and has no exact ten-view compute control. Timings exclude feature preparation; reused controls were measured in a different session. Costs sum sequential active-member runs, and VRAM is their maximum.

## Decision

```json
{
  "dynamics_optimized": {
    "main_rmse_sgd": 455.36972746220016,
    "earlier_nested_transfer_improves": true,
    "candidate_for_independent_confirmation": false
  },
  "macro_optimized": {
    "main_rmse_sgd": 455.3553416650742,
    "earlier_nested_transfer_improves": false,
    "candidate_for_independent_confirmation": false
  },
  "all_optimized": {
    "main_rmse_sgd": 455.36972746220016,
    "earlier_nested_transfer_improves": false,
    "candidate_for_independent_confirmation": false
  }
}
```

All results are development backtests on previously examined labels, not an untouched test. Inherited static geography/MRT timing limitations remain. No submission is generated or replaced by this experiment.

Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_temporal_view_ensembles.py --stage all`.
