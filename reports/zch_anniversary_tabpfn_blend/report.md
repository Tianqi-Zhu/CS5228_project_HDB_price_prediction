# Original zch anniversary ensemble + anniversary TabPFN

Seven zch round-02 tree members were reproduced from the branch source in an isolated worktree.
Their published earlier-selected weights are fixed. A single TabPFN weight was fitted on October 2023-March 2024 labels, then evaluated on June 2024-March 2025.
The original zch final 12-member grouped blend is a different model and is not reconstructed here.

| Model | TabPFN weight | June RMSE (SGD) | June MAE (SGD) |
|---|---:|---:|---:|
| zch_original | 0.0000 | 443.8631 | 322.9188 |
| tabpfn87 | 1.0000 | 441.9593 | 318.2158 |
| equal_blend | 0.5000 | 440.9520 | 318.8879 |
| earlier_locked_blend | 0.1893 | 442.3025 | 321.0325 |

The equal blend improves monthly RMSE versus zch in 9/10 months and versus TabPFN in 10/10 months.

Scores use the same 28,995 validation row IDs and actual rents. This period has been used repeatedly for development; it is not an untouched test.
The earlier-locked weight is the strictly earlier-selected option; the equal blend is a simple prespecified candidate that performed better on this development period.
The full-data candidate is `submission_zch_tabpfn87_equal_150k.csv` when generated. Kaggle test accuracy is unknown.
