# Two full-data TabPFN Kaggle submissions

Both CSVs contain50,000 rows with exactly `Id,Predicted`, in the supplied sample/test order. All rents are finite and positive.

| Candidate | Original development RMSE (SGD) | A/B/C/D/E weights | Upload CSV |
| --- | --- | --- | --- |
| ABCDE_equal | 454.703987 | [0.2, 0.2, 0.2, 0.2, 0.2] | [submission_tabpfn_ABCDE_equal_150k.csv](submission_tabpfn_ABCDE_equal_150k.csv) |
| ABDE | 454.787412 | [0.410715799104646, 0.13922914525091346, 0.0, 0.1996296313112111, 0.25042542433322945] | [submission_tabpfn_ABDE_weighted_150k.csv](submission_tabpfn_ABDE_weighted_150k.csv) |

These are the two lowest observed later-development RMSEs from the previous ensemble comparison, prepared at the user's explicit request. That comparison repeatedly reused development labels; its earlier-selected family was CDE, and its independent-promotion gate failed. The original gate remains unchanged. Neither listed score measures this full-data refit or the hidden Kaggle test, and the small difference between the two candidates is uncertain.

Every member uses all150,000 labelled training transactions. A reuses the verified fresh one-epoch full-data MSE-only DiagScale adapter and its one-view full-precision predictions. B/D/E reuse audited full-data frozen77/84/89-feature fits and two-view predictions. C is newly fitted on all150,000 rows using77 relative-history features, seed42 and two internal estimators; no context subsampling, neural-weight training or adapter is used for C. The19 historical-level fields are replaced by their value minus the original per-row temporal baseline, preserving missingness. All targets and residual baselines remain unchanged.

All weights stay fixed at their previous values. Each model predicts residual rent and adds the original query baseline once before averaging. A predictions are read from the full-precision cache rather than its rounded upload CSV. Existing source artifacts and submissions are checked by SHA256 and preserved.

The inherited temporal policy uses histories through February2025 for March2025 test records, and through March2025 for later test records. Added E spatial features retain their extra one-month lag. All provided labels remain in each TabPFN context. The50,000 test records span March2025-July2026;18,077 lie beyond the ten-month validated temporal range (maximum16months). The existing forecast level/uncertainty plateau at month10, while actual calendar, horizon and history age remain accurate. C only changes the rowwise representation and adds no new historical information.

Validation checks cover every test ID, output column, member alignment, positive/finite prediction, convex combination and CSV rounding. The test file has no rent labels, so local test RMSE cannot be calculated. These files are saved locally and have not been uploaded to Kaggle.

Reproduce/resume: `.venv/Scripts/python.exe analysis/train_tabpfn_top2_submission.py --stage all`. Source/weight hashes are in settings.json; the new C bundle is under.cache/tabpfn_top2_final_submission. Full-precision member predictions, monthly summaries, GPU timing and verification.json are retained here.
