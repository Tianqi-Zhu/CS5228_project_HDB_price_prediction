# Strictly local correction of saved TabPFN ensemble backtests

Earlier calibration selects **sector_local**, with a fixed Ridge alpha of 1000. This compares an unchanged saved TabPFN ensemble, a localized sector/FX/bias correction, and the same correction with one local KLIC return term. These are postprocessors, not a newly fitted TabPFN feature ablation.

Fit on the 2022-09-origin out-of-time predictions and choose on the 2023-07-origin queries. Refit using both earlier prediction sets before evaluating the already examined 2024-06-origin period. All calibration rent labels mature before the next origin. Population query weights are retained. Signals use each query month's historical t-6 observations. The stock inputs update monthly; rental histories in the saved TabPFN predictions stay frozen at their original fold cutoffs.

All correction inputs are multiplied by the fixed 1 km Gaussian exposure, truncated at 3 km. Ridge uses weighted RMS scaling, no centering and no intercept. Consequently, corrections are exactly zero beyond 3 km; changing the fitted backbone or affecting unrelated locations cannot explain a local gain here. This is a stricter locality check than the CatBoost feature ablation, where tree topology can change globally.

| phase | variant | subset | rows | rmse_sgd |
| --- | --- | --- | --- | --- |
| earlier_calibration | sector_local | all_rentals | 2000 | 569.798 |
| earlier_calibration | sector_local | klic_within_1km | 33 | 559.465 |
| earlier_calibration | sector_local | outside_3km | 1766 | 563.501 |
| earlier_calibration | company_local | all_rentals | 2000 | 569.819 |
| earlier_calibration | company_local | klic_within_1km | 33 | 563.197 |
| earlier_calibration | company_local | outside_3km | 1766 | 563.501 |
| earlier_calibration | uncorrected | all_rentals | 2000 | 570.480 |
| earlier_calibration | uncorrected | klic_within_1km | 33 | 578.885 |
| earlier_calibration | uncorrected | outside_3km | 1766 | 563.501 |
| later_reused | sector_local | all_rentals | 28995 | 455.151 |
| later_reused | sector_local | klic_within_1km | 418 | 405.552 |
| later_reused | sector_local | outside_3km | 25229 | 461.329 |
| later_reused | company_local | all_rentals | 28995 | 455.155 |
| later_reused | company_local | klic_within_1km | 418 | 405.844 |
| later_reused | company_local | outside_3km | 25229 | 461.329 |
| later_reused | uncorrected | all_rentals | 28995 | 455.172 |
| later_reused | uncorrected | klic_within_1km | 418 | 397.688 |
| later_reused | uncorrected | outside_3km | 25229 | 461.329 |

| subset | candidate | baseline | earlier_selected | control_rmse | candidate_rmse | delta_rmse | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_rentals | company_local | sector_local | False | 455.151 | 455.155 | 0.004 | -0.006 | 0.011 |
| all_rentals | company_local | uncorrected | False | 455.172 | 455.155 | -0.017 | -0.169 | 0.130 |
| all_rentals | sector_local | uncorrected | True | 455.172 | 455.151 | -0.021 | -0.169 | 0.126 |
| klic_within_1km | company_local | sector_local | False | 405.552 | 405.844 | 0.292 | -0.579 | 0.976 |
| klic_within_1km | company_local | uncorrected | False | 397.688 | 405.844 | 8.156 | -2.610 | 18.751 |
| klic_within_1km | sector_local | uncorrected | True | 397.688 | 405.552 | 7.863 | -2.539 | 17.604 |
| outside_3km | company_local | sector_local | False | 461.329 | 461.329 | 0.000 | 0.000 | 0.000 |
| outside_3km | company_local | uncorrected | False | 461.329 | 461.329 | 0.000 | 0.000 | 0.000 |
| outside_3km | sector_local | uncorrected | True | 461.329 | 461.329 | 0.000 | 0.000 | 0.000 |

Conditional intervals resample paired three-month blocks. Calibration is sparse near K&S and the later period was already inspected, so this cannot establish causality or fresh confirmation. It uses the saved feature-ensemble backtests, not the all-150k final submission model. No submission file or trained TabPFN backbone was modified.
