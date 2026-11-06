# MSE-only DiagScale full-data Kaggle submission

Ready-to-upload file: `submission_tabpfn_diagscale_mse_150k.csv` (50,000 rows, `Id,Predicted`).

A fresh DiagScale adapter is fitted on all 150,000 labelled rows for the previously selected one epoch. The original pretrained TabPFN backbone is frozen; 8,576 attention-scaling parameters use MSE only, learning rate0.01, weight decay0.01, seed42, 1,024-row tasks, 20% supervised queries and two training views. No rows are reserved for validation and no test targets or new tuning select the epoch. This is a full-data refit of the configuration measured at S$455.071 development RMSE; that score is not a measured score for the newly refitted model or the unlabelled competition test.

Test inference uses the saved new adapter, the existing77-feature full-data cache, all150,000 context rows, one deployment view, native preprocessing and200-row query batches without context subsampling. Targets are raw rent minus the original training temporal baseline; query baselines are added exactly once. The CatBoost residual corrections and additional amenity features are omitted.

The established test feature policy is preserved: March2025 test histories stop at February2025, later test histories stop at March2025. All labelled March rows remain available as model context. For 18,077 test rows beyond the ten-month validated temporal forecast range, the forecast level stays at month10 while actual horizon, age and calendar month remain accurate.

CSV predictions range from S$1,693.91 to S$6,342.99; schema,50,000 unique sample IDs, order, positivity, finiteness and CSV rounding are checked. Original development adapters, checkpoints, reports and the frozen-model submission remain unchanged. settings.json,training.json,inference_timing.json,verification.json andtest_month_summary.csv record the provenance.

Reproduce/resume: `.venv/Scripts/python.exe analysis/train_tabpfn_diagscale_mse_submission.py --stage all`. Training and inference use separate fresh processes; the adapter and predictions are cached under `.cache/tabpfn_diagscale_mse_final_submission/`. This script saves the CSV locally and does not upload it.
