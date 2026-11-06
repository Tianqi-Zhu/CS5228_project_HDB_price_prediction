# TabPFN fixed-origin rental comparison

Selected candidate: **catboost**. Promotion requires at least 1% lower pooled RMSE and improvement on every disjoint fold. The forward blend may tie its uncalibrated first fold, but must improve both later folds. This exception was set before TabPFN scoring.

## Configuration

Local v3.5 regressor, 4 estimators, 20000 rows per estimator, seed 42, `fit_with_cache`. GPU: NVIDIA GeForce RTX 4080. Target: raw monthly rent in SGD; predictive mean. Each variant has the same complete fitting population; context subsampling is internal to TabPFN. No COE or stock inputs are used.

Spatial features reuse the existing property, HDB-block, planning and amenity inputs. The history variant adds strictly earlier signature, block × flat-type and area × flat-type means/counts/last means/recency. Training histories exclude the entire current month. Validation histories freeze at the origin, with recency advancing by forecast month. Constant furnished/fee fields and current-row target transformations are excluded.

## Pooled transaction scores

| model | n | rmse_sgd | mae_sgd | bias_sgd | delta_rmse_vs_catboost_sgd |
| --- | --- | --- | --- | --- | --- |
| catboost | 79817 | 539.288 | 413.605 | -26.736 | 0.000 |
| catboost_tabpfn_forward_blend | 79817 | 540.437 | 412.648 | -11.835 | 1.149 |
| tabpfn_history | 79817 | 545.762 | 415.920 | 31.266 | 6.474 |
| tabpfn_spatial | 79817 | 547.222 | 417.353 | -2.306 | 7.934 |

## Fold RMSE (SGD)

| model | 2023-01 | 2023-07 | 2024-04 |
| --- | --- | --- | --- |
| catboost | 637.408 | 557.353 | 466.228 |
| catboost_tabpfn_forward_blend | 637.408 | 561.101 | 465.807 |
| tabpfn_history | 628.029 | 574.711 | 473.572 |
| tabpfn_spatial | 636.426 | 569.920 | 476.160 |

## Forward-only blend

The TabPFN-history weight minimizes squared error on all earlier folds, constrained to [0, 1]. The first fold defaults to CatBoost (zero calibration rows). Later validation labels never choose their own weight. Consequently the pooled blend includes a CatBoost-only first fold; it is not a globally tuned blend score.

| validation_start | tabpfn_weight | calibration_rows |
| --- | --- | --- |
| 2023-01 | 0.000 | 0 |
| 2023-07 | 0.604 | 17754 |
| 2024-04 | 0.415 | 44659 |

## Timing and limits

| validation_start | model | fit_predict_seconds | fit_seconds | predict_seconds | peak_vram_gib |
| --- | --- | --- | --- | --- | --- |
| 2023-01 | catboost | 49.212 | — | — | — |
| 2023-01 | tabpfn_spatial | — | 5.774 | 38.921 | 7.170 |
| 2023-01 | tabpfn_history | — | 5.758 | 39.481 | 7.181 |
| 2023-07 | catboost | 58.408 | — | — | — |
| 2023-07 | tabpfn_spatial | — | 5.564 | 58.819 | 7.169 |
| 2023-07 | tabpfn_history | — | 6.059 | 60.404 | 7.182 |
| 2024-04 | catboost | 67.327 | — | — | — |
| 2024-04 | tabpfn_spatial | — | 5.320 | 77.445 | 7.170 |
| 2024-04 | tabpfn_history | — | 5.844 | 78.016 | 7.182 |

The three validation windows are disjoint. Their months have been examined previously in this project, so these are development results rather than an untouched final test. Supplied amenity tables are current snapshots without historical opening dates. Model/package versions, source hashes, settings and checkpoint paths are recorded in `results.json`. All predictions retain original source row IDs.

Run: `.venv/Scripts/python.exe analysis/compare_tabpfn.py --output reports/tabpfn --model-version v3.5 --context-rows 20000 --n-estimators 4 --batch-size 500 --fit-mode fit_with_cache`. Use `--resume` to reuse matching completed folds. Model downloads may require authentication; data stays local.

## Runtime notes

CatBoost timings combine fitting and prediction; TabPFN reports them separately. Peak memory is PyTorch maximum allocated memory; driver reservations and display memory are additional.

An initial 2,000-row prediction batch exhausted GPU memory. The 500-row configuration passed a memory probe without changing contexts or ensemble size; see `memory_probe.json` and `attempt_batch2000/`.

## Diagnostic figures

![Pooled and fold comparisons](model_comparison.png)

![Monthly RMSE and mean bias](monthly_errors.png)

Regenerate with `.venv/Scripts/python.exe analysis/plot_tabpfn.py --input reports/tabpfn`.
