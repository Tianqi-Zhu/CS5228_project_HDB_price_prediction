# TabPFN context-size comparison

Selected candidate: **CatBoost**.

The lowest observed TabPFN-based pooled RMSE is 540.13 (Forward blend, 50,000-row context), 0.84 SGD higher than CatBoost. The selection rule also requires consistent improvement across forecast origins.

Each completed run uses the same 79,817 transaction validation rows across three disjoint fixed-origin windows, the same TabPFN-3.5 checkpoint, 4 estimators and seed 42. Spatial inputs have 29 columns; rental histories add 12 strictly prior summaries. The target is monthly rent in SGD and the scoring metric is RMSE. Only the completed runs listed below are included.

| Model | Context rows | Pooled RMSE | 2023-01 | 2023-07 | 2024-04 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost | all history | 539.29 | 637.41 | 557.35 | 466.23 |
| Forward blend | 20000 | 540.44 | 637.41 | 561.10 | 465.81 |
| TabPFN + history | 20000 | 545.76 | 628.03 | 574.71 | 473.57 |
| TabPFN spatial | 20000 | 547.22 | 636.43 | 569.92 | 476.16 |
| Forward blend | 50000 | 540.13 | 637.41 | 560.40 | 465.65 |
| TabPFN + history | 50000 | 541.46 | 623.72 | 568.46 | 470.99 |
| TabPFN spatial | 50000 | 544.71 | 633.10 | 567.90 | 473.70 |

Promotion requires at least a 1% pooled RMSE improvement and improvement on every fold. The forward blend may tie its uncalibrated first fold, but must improve both later folds. These rules were fixed before TabPFN scoring. The blend weight for each later fold uses only earlier completed validation windows.

20,000-row context: 500-row prediction batches, checkpoint-default estimator batching. 50,000-row context: 200-row prediction batches, sequential estimator execution. Execution settings, timing and memory are recorded in each run; the model inputs, ensemble count and predictive-mean output remain the same. No COE or stock observations enter either model, and validation rents never update history features.

The validation windows have already been examined in this project. These are development comparisons, not an untouched final test. This experiment does not evaluate full-history TabPFN contexts, fine-tuning, log-target variants, or a new submission.

## Detailed results and verification

- [20,000-row context report](./report.md)
- [50,000-row context report](../tabpfn_50k/report.md)

`context_comparison.csv` includes RMSE, MAE and mean bias. `verification.json` records checks of unique source rows, labels, source/prediction hashes, disjoint windows and independently recomputed pooled RMSE. The CatBoost control predictions agree across runs.

Regenerate with `.venv/Scripts/python.exe analysis/summarize_tabpfn_contexts.py` after both runs complete.
