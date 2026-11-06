# Matched TabPFN anniversary-history ablation

Frozen TabPFN 3.5 Fast; two estimator views; 121,005 fitting rows and 28,995 June 2024–March 2025 query rows.
The original 77 inputs, labels, residual target, temporal baseline, checkpoint, seed and inference budget are fixed.
The candidates add 12/24-month (87 columns) or 12/24/36-month (92 columns) same-block/flat-type cohorts.
Rent histories use fitting labels only and remain frozen at the May 2024 prediction origin.

| Model | Features | RMSE (SGD) | MAE (SGD) | Bias (SGD) |
|---|---:|---:|---:|---:|
| baseline77 | 77 | 455.3959 | 327.7937 | +10.7922 |
| anniversary87 | 87 | 441.9593 | 318.2158 | +4.1084 |
| anniversary92 | 92 | 442.0164 | 317.8146 | +1.3897 |

anniversary87 changes RMSE by -13.4367 SGD versus the control.
anniversary92 changes RMSE by -13.3795 SGD versus the control.

The holdout was previously examined in related experiments; these are development comparisons.
Confirm the gain on other chronological folds before choosing a final configuration.
Features and predictions are saved with source hashes for replay. No low-rent row filtering was applied.
