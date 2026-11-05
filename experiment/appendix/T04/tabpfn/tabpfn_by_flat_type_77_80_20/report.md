# Separate TabPFN inference by flat type

All six flat types are inferred independently using only their same-type context rows. The complete chronological 80/20 holdout is retained.

Original 77 inputs, their categorical dtypes, pre-origin histories and temporal residual baselines are unchanged. This isolates context partitioning; feature histories and the temporal baseline were not rebuilt separately. No tree-cover additions, external scaling, parameter optimization, query-label use or category selection is introduced. Each fresh process uses the same pretrained TabPFN-3.5 Fast checkpoint, one estimator, seed 42, native within-context preprocessing, 200-query batches and every available context row of its flat type.

Context: 121,005 January 2021–May 2024 rows across the six categories. Query: every one of 28,995 June 2024–March 2025 rows. This is the previously examined development holdout, not an untouched test. The unlabelled competition test is not inferred.

## Combined result

| Model | Queries | RMSE (SGD) | MAE (SGD) | Bias (SGD) | Delta RMSE (SGD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pooled_77 | 28995 | 455.814 | 329.748 | 0.768 | +0.000 |
| separate_flat_type_77 | 28995 | 456.445 | 330.283 | -0.331 | +0.631 |

## Per-flat-type result

| Flat type | Context | Queries | Pooled RMSE | Separate RMSE | Delta RMSE | Pooled MAE | Separate MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-room | 32 | 10 | 393.299 | 438.918 | +45.619 | 351.892 | 411.372 |
| 2-room | 2310 | 788 | 434.327 | 424.938 | -9.390 | 291.989 | 296.560 |
| 3-room | 39443 | 9581 | 431.290 | 431.251 | -0.038 | 321.284 | 320.193 |
| 4-room | 43535 | 10544 | 457.198 | 457.355 | +0.157 | 330.459 | 332.254 |
| 5-room | 28926 | 6597 | 467.614 | 470.023 | +2.409 | 330.487 | 329.380 |
| executive | 6759 | 1475 | 549.437 | 553.482 | +4.045 | 396.370 | 403.243 |

The separate-context model does not improve pooled holdout RMSE. No significance claim or uncertainty interval is attached to this comparison. One-room results have only 32 context rows and 10 query rows and should not be treated as a stable estimate.

Total fit time: 14.39 s; total prediction time: 22.73 s; maximum category peak PyTorch-allocated VRAM: 6.794 GiB. Fresh-process startup and input-loading time are excluded.

Inference results: `predictions_combined.csv` and one `predictions_<flat_type>.csv` per category in this report directory. These contain row IDs, months, actual rents, separate predictions and pooled-model reference predictions, all in SGD. Row-level CSVs are reproducible local artifacts and excluded from Git.

Audit: [settings.json](settings.json), [category_counts.csv](category_counts.csv), [scores.csv](scores.csv), [scores_by_flat_type.csv](scores_by_flat_type.csv), [monthly_scores.csv](monthly_scores.csv), [verification.json](verification.json).

Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_by_flat_type.py --stage all`. Summarize existing completed runs with `--stage summarize`.
