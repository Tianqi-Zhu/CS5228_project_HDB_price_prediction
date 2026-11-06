# TabPFN-3 chronological 80/20 experiment

Context: January 2021–May 2024, 121,005 rows (80.67%). Validation: June 2024–March 2025, 28,995 rows. Whole months remain together.

All models use the same 68 feature columns. Training histories use preceding months only; validation histories freeze at May 2024. TabPFN uses one estimator with no row subsampling. The recent-year control uses June 2023–May 2024 as direct context, while retaining the same pre-origin historical features. CatBoost fits all context rows. Validation labels are never fed back.

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| tabpfn_full_context | 28995 | 465.410 | 339.114 | 15.751 |
| tabpfn_recent_year | 28995 | 468.600 | 356.050 | -93.701 |
| catboost_matched | 28995 | 460.782 | 338.909 | -28.262 |

## Runtime

| model | fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | training_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tabpfn_full_context | 16.427 | 90.728 | 6.248 | 121005.000 | False | True |  |
| tabpfn_recent_year | 2.595 | 32.639 | 5.865 | 36103.000 | False | True |  |
| catboost_matched | 93.092 | 0.103 |  |  |  |  | 121005.000 |

This is a ten-month development holdout, shorter than the supplied test's 17-month horizon. Its months have been examined in previous project experiments, so it is not an untouched final test.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_v3_80_20.py`.
