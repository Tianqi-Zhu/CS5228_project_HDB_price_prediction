# Improved temporal training and usage with TabPFN-3.5-Fast

The chronological 80/20 split is January 2021-May 2024 context (121,005 rows) and June 2024-March 2025 validation (28,995 rows). All source context rows remain included.

All 37 rental-history inputs are regenerated at the same earlier origin as the temporal forecasts. Historical context rows cover 1-10 month horizons, matching the evaluation window. The previous version combined stale temporal forecasts with preceding-transaction-month local histories; this experiment aligns those information sets. Static inputs and rental labels are preserved.

A pooled local model selects flat-type trend damping, planning-area shrinkage, and local-premium decay from rolling forecasts whose target labels are already observed at each origin. It uses the last twelve observed target months, gives available horizons equal weight, and prefers a simpler candidate within 1% of the best error. Forecast error and local pooling weight are also inputs. No economic observations are used.

Both new variants use the same 77 inputs. The feature variant predicts raw rent. The residual variant predicts rent minus the local temporal baseline and adds that baseline back exactly once. Rows without any earlier history use additive baseline zero and keep missing temporal inputs.

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| tabpfn_aligned_features | 28995 | 457.788 | 332.431 | -6.699 |
| tabpfn_aligned_residual | 28995 | 455.863 | 329.984 | -2.428 |
| previous_temporal_levels | 28995 | 457.827 | 332.582 | -0.583 |
| original_fast | 28995 | 459.812 | 336.604 | -10.284 |

## Runtime and batch checks

| model | fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | batch_size | batch_agreement | batch_max_difference_sgd | target_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn_aligned_features | 14.956 | 1181.874 | 17.722 | 121005 | False | True | 200 | False | 0.731 | raw |
| tabpfn_aligned_residual | 16.139 | 1010.371 | 17.720 | 121005 | False | True | 200 | False | 0.478 | residual |

The larger prediction batch is checked against 200-row batches on the same fitted model without using labels. If numerical agreement fails, inference uses 200-row batches. Memory retries retain all context rows. All models use one estimator, seed 42, and the same Fast checkpoint.

This is a repeatedly examined ten-month development holdout. Temporal parameters use earlier data, but model design has evolved using development results. The supplied test spans seventeen months and has no rent labels. The new method should be confirmed at another forecast origin.

Model-ready data: `.cache/tabpfn_v3_features/holdout_80_20_2024-06_aligned_temporal.joblib`. Origin metadata and new numeric inputs are in `temporal_features_train.csv` and `temporal_features_query.csv`.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_aligned_temporal_80_20.py`.

## Result interpretation and reuse

The residual model reduces RMSE from S$457.827 to S$455.863 (0.43%) and improves 9 of 10 months. The feature-only score of S$457.788 is effectively tied with the previous best. Both new variants share the same 77 inputs, isolating the effect of residual target usage.

The runtime table peaks include the rejected 1,000-row batch probe and describe PyTorch allocations, not measured resident physical VRAM. A separate production-only probe using the complete 121,005-row context and two 200-row prediction batches peaked at 7.219 GiB allocated. Its first 400 reconstructed rent predictions match the saved residual results to less than 0.000001 SGD. See `production_memory_probe.json`.

For the residual model, fit using `bundle['y_train'] - bundle['train_baseline']`; reconstruct rents with `predicted_residual + bundle['query_baseline']`. The bundle keeps original rent labels for auditing and for the raw-target control. Use the experiment runner to reproduce the complete fitting and inference configuration.
