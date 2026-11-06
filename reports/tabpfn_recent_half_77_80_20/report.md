# Recent-half TabPFN context experiment

Use only the latest 21 of 41 training calendar months: **September 2022–May 2024**, 60,926 context rows. The original 80% training set spans January 2021–May 2024 (121,005 rows). For the odd 41-month horizon, retain the middle month in full; selection is by time, not half the row count.

Validation is unchanged: all 28,995 June 2024–March 2025 rows. Models use the original 77 features, categorical dtypes, pretrained TabPFN-3.5 Fast checkpoint, one estimator, seed 42, 200-query batches, native training-context preprocessing and no context subsampling. No parameter optimization or validation labels enter inference.

This is a **context-only truncation experiment**. The inherited 37 history features and temporal residual baselines are preserved; observations before September 2022 can still contribute to those precomputed inputs. This does not test rebuilding every feature and baseline using recent data exclusively. Predictions fit original training residuals and restore the original query baseline exactly once.

Pooled recent context includes all flat types. The companion per-type run routes each query to same-type recent context, continuing the previous flat-type experiment. All holdout rows are evaluated in both modes. The unlabelled competition test and existing submission artifacts remain unchanged.

## Overall result

| Model | RMSE (SGD) | MAE (SGD) | Bias (SGD) | Delta RMSE vs full pooled |
| --- | ---: | ---: | ---: | ---: |
| full_pooled | 455.814 | 329.748 | 0.768 | +0.000 |
| full_by_flat_type | 456.445 | 330.283 | -0.331 | +0.631 |
| recent_pooled | 456.733 | 331.837 | -12.469 | +0.920 |
| recent_by_flat_type | 457.039 | 332.482 | -12.429 | +1.225 |

## Per-flat-type RMSE

| Flat type | Recent context | Validation | Full pooled | Recent pooled | Full separate | Recent separate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-room | 17 | 10 | 393.299 | 466.290 | 438.918 | 354.923 |
| 2-room | 1410 | 788 | 434.327 | 424.146 | 424.938 | 427.071 |
| 3-room | 19850 | 9581 | 431.290 | 431.947 | 431.251 | 431.856 |
| 4-room | 22133 | 10544 | 457.198 | 457.684 | 457.355 | 458.557 |
| 5-room | 14257 | 6597 | 467.614 | 470.763 | 470.023 | 469.020 |
| executive | 3259 | 1475 | 549.437 | 550.046 | 553.482 | 556.275 |

## Runtime

| Recent-context mode | Fit seconds | Prediction seconds | Peak allocated VRAM (GiB) |
| --- | ---: | ---: | ---: |
| recent_pooled | 6.07 | 31.36 | 6.890 |
| recent_by_flat_type | 8.81 | 15.45 | 6.681 |

Per-type times sum the six runs; device memory reports the largest per-process PyTorch allocation. Process startup and data loading are excluded. These are development-holdout results without uncertainty intervals or significance claims. One-room recent context contains only 17 observations and its validation has only 10 observations.

Predictions for every validation row and all four protocols are saved to `predictions_combined_comparison.csv`. Individual recent-context files are `predictions_recent_pooled.csv` and `predictions_recent_type_<flat_type>.csv`.

Audit: [settings.json](settings.json), [scores.csv](scores.csv), [scores_by_flat_type.csv](scores_by_flat_type.csv), [monthly_scores.csv](monthly_scores.csv), [category_counts.csv](category_counts.csv), [verification.json](verification.json).

Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_recent_half.py --stage all`. Regenerate summary only with `--stage summarize`.
