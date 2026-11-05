# Findings: TabPFN reliability and forecast-derived COE inputs

All appended input sets worsen later RMSE. Retain the original77-feature inputs and current submission. The rent-only forecast won earlier selection but failed to transfer to the later fold.

The frozen two-estimator TabPFN control, complete context rows, original rental histories and additive residual baseline are fixed. This isolates nested input additions. The numerical-best MSE-only DiagScale model is a separate architecture/view control, not the frozen baseline in this table.

## Individual-rental results

June2024–March2025, all28,995 rentals; SGD RMSE. Negative changes improve the frozen77-feature control.

| member | features | rmse_sgd | delta_vs_baseline_sgd |
| --- | --- | --- | --- |
| baseline | 77 | 455.396 | 0.000 |
| reliability | 82 | 455.962 | 0.566 |
| rent_forecast | 84 | 456.707 | 1.311 |
| coe_increment | 85 | 456.905 | 1.509 |

Inputs: reliability adds five missing/sparse-history/unknown-block flags; rent_forecast adds a national rent-only LSTM growth forecast and unavailable flag; coe_increment adds the difference between matched rent+COE and rent-only forecast growth. Both LSTMs use12 months,8 hidden units,100 fixed epochs and seed42.

## Does each addition help?

| candidate | reference | delta_rmse_sgd | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| reliability | baseline | 0.566 | 0.197 | 1.002 |
| rent_forecast | reliability | 0.746 | 0.277 | 1.125 |
| coe_increment | rent_forecast | 0.198 | -0.043 | 0.462 |
| rent_forecast | baseline | 1.311 | 0.917 | 1.614 |
| coe_increment | baseline | 1.509 | 1.210 | 1.739 |

Intervals resample paired three-month blocks. They omit fitting/selection uncertainty and multiple-search adjustment. The ten later months have been used by many earlier experiments. Neither a small favorable point estimate nor an interval is untouched confirmation.

## Earlier selection and sparse-history coverage

| member | pooled_earlier_rmse_sgd |
| --- | --- |
| baseline | 601.934 |
| reliability | 614.062 |
| rent_forecast | 596.443 |
| coe_increment | 607.629 |

The nested July2023 choice based on September2022 alone is **rent_forecast**. Only earlier query targets enter these selections. The specification was informed by prior development studies; the old historical folds are not independent discoveries.

| origin | split | missing | rows |
| --- | --- | --- | --- |
| 2022-09 | train | 59398 | 60079 |
| 2022-09 | query | 1400 | 2000 |
| 2023-07 | train | 75742 | 87937 |
| 2023-07 | query | 0 | 2000 |
| 2024-06 | train | 79046 | 121005 |
| 2024-06 | query | 0 | 28995 |

Missing macro forecasts are deliberately retained with an availability flag. A direct head requires three mature monthly labels. Early matrices have very few distinct forecast values; the unchanged native TabPFN preprocessor can infer a low-cardinality forecast column as categorical and warn when incoming values differ. The matrices themselves retain identical float64 appended-column dtypes. This is a real startup limitation of the tested pipeline, not evidence that a richer-history numerical forecast can never help. The unavailable flag and forecast value are added together, so their separate contributions are not isolated.

Each input is generated at the row's existing historical forecast origin with an additional one-month publication buffer, for both rents and COE. Future-value poisoning refits replayed exactly. Only source observations available by that cutoff train an LSTM forecast. The reference adjustment itself uses2021 only and is applied to forecasts after that year ends. Forecast growth caps at head10 if the publication buffer creates an eleven-month macro distance.

## DiagScale confirmation

| inputs | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- |
| original77 | 455.071 | 330.790 | -8.428 |
| rent_forecast | 456.720 | 332.238 | -7.703 |

This keeps the original preDecember2023 adapter fixed with one view; it does not retrain an adapter on the augmented input matrices. No later target selected this input set.

## Separate 11–16-month baseline check

Earlier 11–16-month area/type mean-rent errors select **plateau10**.

| band | policy | unique_target_months | rmse_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| 1-10 | damped16 | 4 | 204.204 | -107.424 |
| 1-10 | persistence | 4 | 165.417 | -63.294 |
| 1-10 | plateau10 | 4 | 204.204 | -107.424 |
| 11-16 | damped16 | 10 | 244.276 | 3.704 |
| 11-16 | persistence | 10 | 175.516 | -94.745 |
| 11-16 | plateau10 | 10 | 212.894 | -17.655 |

Plateau10 holds the temporal forecast at its ten-month level. Damped16 extends the existing damped trend and local-premium decay. Persistence carries the latest shrunken local level. All policies fit only data available at each historical cutoff. Later cases reuse June2024–March2025 targets, and overlapping origins can reuse the same rentals. These are count-weighted area/type cell-mean errors; they must not be compared numerically with individual-rental TabPFN RMSE. Changing the baseline inside a refitted residual TabPFN remains a separate, unmeasured experiment.

## Files and reproduction

- [Full methods and all folds](report.md)
- [Feature coverage](feature_coverage.csv)
- [Causality replay](causality_audit.json)
- [Verification](verification.json)

Run `.venv/Scripts/python.exe analysis/compare_tabpfn_reliability_macro.py --stage all`, then `.venv/Scripts/python.exe analysis/summarize_tabpfn_reliability_macro.py`. Completed inference jobs resume using checked source, feature and prediction hashes. No raw data, previous experiment artifacts, pretrained weights, existing adapters or final submission CSVs are overwritten.
