# TabPFN-3.5-Fast full-context 80/20 experiment

Context: January 2021-May 2024, 121,005 rows (80.67%). Validation: June 2024-March 2025, 28,995 rows (19.33%). Whole months remain together.

TabPFN-3.5-Fast uses its separate Fast checkpoint, one estimator, seed 42, and every context row. Row subsampling is disabled and checked after fitting. All models share the exact saved 68 features and validation row IDs. Training histories use earlier months; validation histories freeze at May 2024. The TabPFN-3 and CatBoost predictions are reused after source, feature and row-label checks.

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| tabpfn_3_5_fast_full_context | 28995 | 459.812 | 336.604 | -10.284 |
| tabpfn_3_full_context | 28995 | 465.410 | 339.114 | 15.751 |
| catboost_matched | 28995 | 460.782 | 338.909 | -28.262 |

## Runtime

| model | fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | architecture | training_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn_3_5_fast_full_context | 15.231 | 588.211 | 7.250 | 121005.000 | False | True | TabPFNV3p5 |  |
| tabpfn_3_full_context | 16.427 | 90.728 | 6.248 | 121005.000 | False | True |  |  |
| catboost_matched | 93.092 | 0.103 |  |  |  |  |  | 121005.000 |

GPU figures are peak PyTorch allocation. Control timings come from the previous run on the same GPU. This is a ten-month development holdout; the unlabelled test spans 17 months.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_v3_5_fast_80_20.py`.
