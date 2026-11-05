# TabPFN: reliability and forecast-derived COE input ablation

Earlier-only input selection chooses **rent_forecast**. All added-feature variants are reported below. This is a matched frozen TabPFN study; the previously measured MSE-only DiagScale control uses a different adapter/view setting. The already examined later period is development evidence, not a new test.

## Main individual-rental results

| member | features | rows | rmse_sgd | mae_sgd | bias_sgd | delta_vs_baseline_sgd |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 77 | 28995 | 455.396 | 327.794 | 10.792 | 0.000 |
| reliability | 82 | 28995 | 455.962 | 328.283 | 13.540 | 0.566 |
| rent_forecast | 84 | 28995 | 456.707 | 328.417 | 15.287 | 1.311 |
| coe_increment | 85 | 28995 | 456.905 | 329.586 | 7.542 | 1.509 |

## Earlier chronology and selection

| origin | member | rmse_sgd | delta_vs_baseline_sgd |
| --- | --- | --- | --- |
| 2022-09 | baseline | 635.086 | 0.000 |
| 2022-09 | reliability | 647.344 | 12.258 |
| 2022-09 | rent_forecast | 633.977 | -1.109 |
| 2022-09 | coe_increment | 644.224 | 9.138 |
| 2023-07 | baseline | 569.479 | 0.000 |
| 2023-07 | reliability | 581.510 | 12.030 |
| 2023-07 | rent_forecast | 559.404 | -10.076 |
| 2023-07 | coe_increment | 571.616 | 2.137 |

Pooled earlier weighted RMSE: {"baseline": 601.9336991618188, "reliability": 614.0620126940327, "rent_forecast": 596.4432560007637, "coe_increment": 607.6292929219736}. The July2023 nested choice, using only September2022 predictions, is rent_forecast. No later target selected a feature set or a macro architecture/window.

## Incremental feature effects

| origin | candidate | reference | delta_rmse_sgd | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| 2024-06 | reliability | baseline | 0.566 | 0.197 | 1.002 |
| 2024-06 | rent_forecast | reliability | 0.746 | 0.277 | 1.125 |
| 2024-06 | coe_increment | rent_forecast | 0.198 | -0.043 | 0.462 |
| 2024-06 | rent_forecast | baseline | 1.311 | 0.917 | 1.614 |
| 2024-06 | coe_increment | baseline | 1.509 | 1.210 | 1.739 |

Intervals use paired contiguous three-month blocks and retain population sampling weights. They condition on fitted models and omit fitting/selection uncertainty; the ten later months are a small dependent sample.

## Feature definitions and availability

- reliability82: original77 plus missing signature/block history, block3m count<2, block12m count<5, and unknown BLOCK_ID flags.
- rent_forecast84: reliability82 plus predicted national rent-only log growth and a macro-unavailable indicator.
- coe_increment85: rent_forecast84 plus (rent+COE forecast growth minus matched rent-only forecast growth).
- Both macro LSTMs have12-month windows,8 hidden units,100 fixed epochs,seed42 and ten direct heads. Heads require3 mature monthly labels; early unsupported forecasts remain NaN. All neural loss targets stop at their source cutoff.
- Macro data stop at each original per-row feature origin minus one month. The extra buffer can make the macro forecast distance eleven months; the growth feature then stays at head10. Actual calendar/horizon and all original inputs stay unchanged.
- The national index uses the old2021-only property adjustment, and is never exposed as a forecast before2021 ends. There are only51 monthly rent observations in total. Source snapshots do not reproduce exact historical release vintages.
- All original contexts, native categorical/missing handling, residual targets and additive rental baselines are unchanged. Two internal estimators,seed42,full context,native batching and batch200 match the audited frozen control. Control predictions are reused only with hashes, IDs, labels and settings checked.

## DiagScale transfer confirmation

| model | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- |
| original77 | 455.071 | 330.790 | -8.428 |
| rent_forecast | 456.720 | 332.238 | -7.703 |

The existing preDecember2023 MSE-only adapter is held fixed with one inference view. It is not retrained for the appended inputs; this measures feature transfer to that adapter, not an optimized new adapter.

## Separate long-horizon baseline experiment

Earlier 11–16-month cell-mean errors select **plateau10**.

| phase | band | policy | cells | unique_target_months | rental_case_weights | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 1-10 | damped16 | 4465 | 21 | 105806 | 218.389 | 157.354 | 50.494 |
| development | 1-10 | persistence | 4465 | 21 | 105806 | 284.683 | 207.741 | -173.874 |
| development | 1-10 | plateau10 | 4465 | 21 | 105806 | 218.389 | 157.354 | 50.494 |
| development | 11-16 | damped16 | 1612 | 11 | 39001 | 425.428 | 345.597 | 307.759 |
| development | 11-16 | persistence | 1612 | 11 | 39001 | 439.776 | 374.465 | -368.090 |
| development | 11-16 | plateau10 | 1612 | 11 | 39001 | 321.752 | 252.744 | 179.039 |
| later_reused | 1-10 | damped16 | 494 | 4 | 11797 | 204.204 | 145.689 | -107.424 |
| later_reused | 1-10 | persistence | 494 | 4 | 11797 | 165.417 | 107.844 | -63.294 |
| later_reused | 1-10 | plateau10 | 494 | 4 | 11797 | 204.204 | 145.689 | -107.424 |
| later_reused | 11-16 | damped16 | 1364 | 10 | 31658 | 244.276 | 189.498 | 3.704 |
| later_reused | 11-16 | persistence | 1364 | 10 | 31658 | 175.516 | 125.919 | -94.745 |
| later_reused | 11-16 | plateau10 | 1364 | 10 | 31658 | 212.894 | 160.948 | -17.655 |

These are planning-area/flat-type cell-mean forecasts, weighted by rental counts, from four historical origins. Their SGD RMSE is not individual-rental TabPFN RMSE. Forecast cases overlap; rental_case_weights can count the same observed rental under different origins. Native month10 plateau is compared with continued damped trend/local-premium decay and persistence of the latest shrunken local level. Only targets beforeJune2024 select a policy. This does not refit TabPFN residual targets or change the submission baseline.

## Reproduce and artifacts

`.venv/Scripts/python.exe analysis/compare_tabpfn_reliability_macro.py --stage all`

settings.json records source/code hashes and settings; features_*.json and feature_coverage.csv describe matrices; national_forecasts.csv records every causal macro cutoff/head; causality_audit.json records future-value poisoning; predictions_*.csv and timing_*.json record fixed-model inference. Stage one resumes completed hash-checked runs. Existing reports, raw CSVs, adapters and both final submission files remain unchanged.
