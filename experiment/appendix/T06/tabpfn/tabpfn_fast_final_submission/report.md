# TabPFN-3.5-Fast Kaggle test submission

Ready-to-upload CSV: `reports/tabpfn_fast_final_submission/submission_tabpfn_fast_150k.csv` (50,000 rows; `Id,Predicted`).

The frozen checkpoint is fitted as a default regressor on residual rent targets using **all 150,000** labeled training rows as context. The feature set matches the 77-column categorical-month 80/20 setup (S$455.81 development validation RMSE). The failed embedding MLP is omitted. The supplied test set has no rent labels, so no test RMSE or other test loss is measurable.

March 2025 test features use rental histories through February 2025. April 2025-July 2026 test features use histories through March 2025. March labeled training rows are available to the full TabPFN context, but never feed into March test history features. Training context rows use the same pseudo-origin aligned rent histories and local temporal residual baseline as the validation setup.

The local temporal model was fitted and validated only for horizons 1-10. For later test months, its level and uncertainty are held at the horizon-10 forecast; the actual month, calendar category, forecast horizon, and history age remain accurate. This is an extrapolation that was not validated by the ten-month 80/20 experiment. See `test_month_summary.csv` and `feature_manifest.json`.

All 50,000 CSV rows were reloaded, checked finite and positive, and checked against the sample submission's exact column names and row order. Prediction range: S$1,613.18-S$5,945.49; peak GPU memory: 7.35 GiB.

Reproduce with `.venv/Scripts/python.exe analysis/train_tabpfn_fast_submission.py`. Model inference progress is resumable under ignored `.cache/tabpfn_fast_final_submission/`.
