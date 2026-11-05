# TabPFN-3.5-Fast with temporal forecast features

Context: January 2021-May 2024, 121,005 rows (80.67%). Validation: June 2024-March 2025, 28,995 rows. One estimator, seed 42, raw monthly rent target, 200-row prediction batches, all context rows retained.

Eight numeric inputs are added to the original 68 features: `TEMP_FLAT_RECENT3_MEAN`, `TEMP_FLAT_RECENT12_MEAN`, `TEMP_FLAT_DAMPED_FORECAST`, `TEMP_FLAT_MODEL_FORECAST`, `TEMP_FLAT_MODEL_DELTA`, `TEMP_FORECAST_HORIZON`, `TEMP_FLAT_HISTORY_COUNT_3M`, `TEMP_FLAT_LAST_OBSERVATION_AGE`.

Context transactions receive deterministic forecasts from earlier origins spanning 1-17 months. Each temporal model and its regularization are fit/selected using only outcomes available at its own origin. The learned forecaster requires 18 history months; earlier rows retain missing learned forecasts. Simple levels use whatever strictly earlier rental history is available. Validation forecasts all originate at May 2024. COE is excluded following the standalone economic-model ablation.

The original 68 inputs retain their causal definitions, including preceding-month histories on context rows. The new forecast columns carry their separate origin and horizon. They are predictor inputs; the model still predicts raw rent, so no forecast is added to its output a second time.

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| tabpfn_fast_temporal | 28995 | 460.716 | 331.254 | 34.169 |
| tabpfn_fast_original | 28995 | 459.812 | 336.604 | -10.284 |
| catboost_original_68_features | 28995 | 460.782 | 338.909 | -28.262 |

## Runtime

| model | fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn_fast_temporal | 15.654 | 877.633 | 7.156 | 121005 | False | True | TabPFNV3p5 |
| tabpfn_fast_original | 15.231 | 588.211 | 7.250 | 121005 | False | True | TabPFNV3p5 |

The CatBoost reference uses the original 68 features; the direct feature ablation is against `tabpfn_fast_original` on exactly the same validation rows. Baseline predictions are hash-checked. All scores are development results on a ten-month horizon, shorter than the unlabelled test's 17 months.

Model-ready enriched data: `.cache/tabpfn_v3_features/holdout_80_20_2024-06_temporal.joblib`. The `temporal_features_train.csv` and `temporal_features_validation.csv` exports contain source row IDs and only the added features plus origin metadata. Original rental labels remain separate in the joblib bundle.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_fast_temporal_80_20.py`.
