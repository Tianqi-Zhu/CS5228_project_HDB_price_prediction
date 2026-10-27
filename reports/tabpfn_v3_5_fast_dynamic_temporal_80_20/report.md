# Comprehensive temporal model with TabPFN-3.5-Fast

The broader candidate scores **S$457.97 RMSE**, compared with **S$455.86** for the previous aligned residual model: S$2.10 (0.46%) worse. It worsens all ten validation months and its MAE rises from S$329.98 to S$332.37. Retain the previous aligned residual setup. This experiment does not support adding the entire new temporal feature block.

The same chronological 80/20 split uses 121,005 context rows (January 2021-May 2024) and 28,995 validation rows (June 2024-March 2025). All context rows, aligned history columns, property inputs, and source labels are retained.

A property reference fitted on the first twelve months removes changes in transaction mix from the monthly market index. It is used only at origins where that reference is already available. Earlier origins use the previous causal baseline. The market model pools planning-area and flat-type information and learns damped local trends with sparse-group shrinkage.

Additional direct dynamic regressions consider 3/12-month levels and trends, annual changes, acceleration/regime shifts, annual Fourier seasonality, rental-market volatility, observation counts, activity momentum, common/type-specific movement, and delayed category-A COE momentum. Future COE values are never inputs. Regularization is scaled to distinct market months, all coefficients including the intercept are penalized, and corrections are conservatively shrunk. Each origin tests structural-only and COE variants against its simple baseline using earlier forecast origins; a new correction must improve error by at least 1% to be used.

Final-origin economic correction selected: `None` (`null` means no learned correction). Twenty temporal outputs replace the previous nine, giving 88 total inputs. TabPFN predicts a residual around the resulting baseline and reconstructs rent by adding that baseline exactly once.

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| tabpfn_dynamic_temporal_residual | 28995 | 457.967 | 332.367 | -8.828 |
| previous_best_residual | 28995 | 455.863 | 329.984 | -2.428 |
| previous_aligned_features | 28995 | 457.788 | 332.431 | -6.699 |
| original_fast | 28995 | 459.812 | 336.604 | -10.284 |

## Runtime

| fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | architecture |
| --- | --- | --- | --- | --- | --- | --- |
| 16.306 | 816.041 | 7.125 | 121005 | False | True | TabPFNV3p5 |

These are development results on a repeatedly examined ten-month holdout, not an untouched test. The supplied test has a seventeen-month horizon. The model covers observable temporal signals in this dataset; it cannot account for unobserved economic drivers or future shocks.

Model-ready data: `.cache/tabpfn_v3_features/holdout_80_20_2024-06_dynamic_temporal.joblib`. Fit with `y_train - train_baseline` and reconstruct with `predicted_residual + query_baseline`. Component choices at every origin are recorded in `temporal_model_audit.json`.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_dynamic_temporal_80_20.py`.

## Interpretation and verification

The standalone property-adjusted baseline improves individual-rent RMSE from S$499.17 for the previous local baseline to S$487.46. That improvement does not carry through to TabPFN. The final-origin inner backtest scores the simple adjusted baseline at S$251.57 group-mean RMSE, compared with S$260.70 for the best structural correction and S$260.88 for the best COE correction. Both optional corrections are therefore disabled at the final origin. Group-mean inner scores and outer individual-rent scores measure different prediction tasks.

Plausible explanations are redundant signals already represented by the existing rental histories, a changed residual target, and limited independent temporal observations. The fixed first-year property reference leaves only 29 later monthly target periods for correction training. To preserve causality, adjusted-index features are missing for 31.1% of context rows during reference warmup, while they are complete for validation. This availability shift may make those inputs less useful to TabPFN. These are hypotheses: this combined experiment does not isolate each component causally.

Nine temporal tests passed. Future rents and future COE values cannot alter an earlier forecast, candidate selection, or correction coefficients; the first-year reference is unavailable before its training period ends. Saved feature checks verify unchanged original 68 inputs and raw labels, strictly earlier origins, and finite residual baselines. All 28,995 predictions have unique IDs and finite values; saved metrics were independently recomputed. Prediction used all 121,005 context rows with no subsampling. GPU fitting took 16.3 seconds and prediction 816.0 seconds; peak PyTorch allocation was 7.13 GiB on the RTX 4080, with no out-of-memory retry.

A useful next experiment would change one element at a time, starting with the property-adjusted market index while retaining the previous residual target and other features. Selection should use earlier origins and include a 17-month forecast window before claiming an improvement for production. Increasing temporal model complexity is not supported by this result.

Additional audits: `verification.json`, `monthly_comparison.csv`, `temporal_component_summary.csv`, and `temporal_model_audit.json`.
