# TabPFN-3.5-Fast temporal feature ablation

The same chronological 80/20 split uses January 2021-May 2024 as context (121,005 rows) and June 2024-March 2025 as validation (28,995 rows). Every context row is retained. All Fast models use one estimator, seed 42, the same checkpoint, and 200-row prediction batches.

The levels/trend variant adds six predictors to the original 68: `TEMP_FLAT_RECENT3_MEAN`, `TEMP_FLAT_RECENT12_MEAN`, `TEMP_FLAT_DAMPED_FORECAST`, `TEMP_FORECAST_HORIZON`, `TEMP_FLAT_HISTORY_COUNT_3M`, `TEMP_FLAT_LAST_OBSERVATION_AGE`. Its causal forecast origins match the full temporal-feature experiment. All temporal values use only rent observations available at their own origin; no COE inputs are included. The learned forecast and its delta are removed to diagnose the eight-feature run's higher RMSE.

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| tabpfn_fast_temporal_levels | 28995 | 457.827 | 332.582 | -0.583 |
| tabpfn_fast_original | 28995 | 459.812 | 336.604 | -10.284 |
| tabpfn_fast_temporal_full | 28995 | 460.716 | 331.254 | 34.169 |
| catboost_original_68_features | 28995 | 460.782 | 338.909 | -28.262 |

## Runtime

| model | fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn_fast_temporal_levels | 14.549 | 755.624 | 7.175 | 121005 | False | True | TabPFNV3p5 |
| tabpfn_fast_original | 15.231 | 588.211 | 7.250 | 121005 | False | True | TabPFNV3p5 |
| tabpfn_fast_temporal_full | 15.654 | 877.633 | 7.156 | 121005 | False | True | TabPFNV3p5 |

This is a development ablation selected after observing the full feature run. It is not an independent final test. The holdout covers ten months; the supplied unlabelled test covers seventeen. The CatBoost reference uses the original 68 features.

Enriched model-ready data: `.cache/tabpfn_v3_features/holdout_80_20_2024-06_temporal_levels.joblib`. Added training and validation columns are exported separately as `temporal_features_train.csv` and `temporal_features_validation.csv`, with source row IDs and origin metadata.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_fast_temporal_levels_80_20.py`.
