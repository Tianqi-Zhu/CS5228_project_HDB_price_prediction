# Direct TabM versus TabPFN comparison on the chronological 80/20 split

**Primary TabM seed-42 RMSE: S$463.196; difference from frozen TabPFN: +7.383 SGD.**

Retain TabPFN for this setup: the primary TabM run does not improve the same development holdout. Across the three fixed seeds, RMSE mean is **S$462.954**, with sample standard deviation **S$2.462**. Seeds are separate refits; no cross-seed prediction blend or seed selection is used.

## Matched data and target

All models are evaluated on the exact same 28,995 June 2024-March 2025 rows. TabM is trained from random initialization on all 121,005 January 2021-May 2024 rows. The 77-column categorical-month TabPFN source artifact is reused unchanged, including its pseudo-origin aligned training histories and fixed-May-2024 query histories. It is SHA-256 checked against the saved TabPFN feature manifest.

The target is the same residual rent: `y_train - train_baseline`; add `query_baseline` exactly once at prediction time. Source rent labels and all source rows are retained. No TabPFN embeddings, predictions or pretrained weights are inputs to TabM. This is a standalone model comparison, not the separate proposed area-adjustment/history-feature experiment. Hidden test data and the existing submission were not modified or scored.

## Selection before outer evaluation

All 102,829 rows before December 2023 fit each calibration candidate. The complete 18,176 December 2023-May 2024 rows select settings and stopping epoch. All 37 history inputs and nine temporal inputs for those queries are rebuilt using only labels through November 2023. This uses a full earlier calibration set, whereas some prior TabPFN adapter experiments used 600 sampled queries; their earlier calibration scores are not directly comparable.

Preprocessing fits only on each fitting partition: numerical medians, missingness indicators, means/scales and categorical vocabularies. Constant numerical representation columns are removed. The original nine categorical inputs use official TabM one-hot encoding with a reserved unknown/missing bucket. PLE bins are unsupervised fitting-data quantiles, 48 bins, dimension 16, version B. No outer inputs fit preprocessing, and no outer label chooses a hyperparameter, epoch, seed or checkpoint.

The official `tabm==0.0.3` implementation uses three 256-unit blocks, dropout 0.1 and 32 parameter-sharing internal members. Each member is trained with its own MSE before predictions are averaged. AdamW weight decay is 0.0003; tested learning rates are shown below. Targets are standardized on fitting rows only. Training uses BF16 autocast with FP32 loss/averaging, batch size 256, gradient clipping 1, maximum 100 epochs and patience 12. No broad hyperparameter search was performed.

| name | calibration_rmse_sgd | best_epoch | epochs_run | parameters | seconds | peak_vram_gib |
| --- | --- | --- | --- | --- | --- | --- |
| calibration/plain_lr002 | 520.443 | 1 | 13 | 3144160 | 50.854 | 0.757 |
| calibration/ple_lr002 | 521.994 | 2 | 14 | 3717600 | 72.258 | 1.236 |
| calibration/ple_lr001 | 521.188 | 2 | 14 | 3717600 | 72.429 | 1.236 |

Earlier-selected configuration: **`plain_lr002`**. All final seeds refit on the entire 80% portion with a fixed epoch count of **1**. The remaining 20% is never an early-stopping set. Primary seed 42 and additional seeds 43/44 were declared before scoring.

## Complete outer results

| model | n | rmse_sgd | mae_sgd | bias_sgd | delta_vs_frozen_sgd |
| --- | --- | --- | --- | --- | --- |
| tabpfn_frozen | 28995 | 455.814 | 329.748 | 0.768 | 0.000 |
| tabpfn_original_diagscale | 28995 | 455.212 | 330.625 | -2.798 | -0.602 |
| tabpfn_mse_only_development | 28995 | 455.071 | 330.790 | -8.428 | -0.743 |
| tabm_seed42 | 28995 | 463.196 | 330.909 | 35.186 | 7.383 |
| tabm_seed43 | 28995 | 465.285 | 332.303 | 62.726 | 9.472 |
| tabm_seed44 | 28995 | 460.379 | 334.558 | -0.913 | 4.566 |

Frozen TabPFN is the direct source-pipeline control. Original DiagScale and the numerically best MSE-only adapter are contextual development references; the latter's previously measured gain was tiny and not reliable across months. All saved reference losses are recomputed from aligned row predictions.

Primary TabM improves **0 of 10 months** against frozen TabPFN. Paired month-bootstrap median RMSE difference is **+7.360 SGD**, with a descriptive 95% interval **[+4.712, +10.168]** (20,000 resamples, seed 42). The seed standard deviation and the month-bootstrap interval measure different sources of variation.

## Size-related diagnostic groups

| model | group | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- | --- |
| tabpfn_frozen | floor_area_above_130sqm | 1603 | 566.600 | 409.022 | -3.766 |
| tabpfn_frozen | executive | 1475 | 549.437 | 396.370 | -2.307 |
| tabpfn_original_diagscale | floor_area_above_130sqm | 1603 | 566.033 | 414.835 | -34.801 |
| tabpfn_original_diagscale | executive | 1475 | 548.816 | 401.136 | -30.841 |
| tabpfn_mse_only_development | floor_area_above_130sqm | 1603 | 566.457 | 410.528 | -13.109 |
| tabpfn_mse_only_development | executive | 1475 | 549.078 | 396.669 | -4.949 |
| tabm_seed42 | floor_area_above_130sqm | 1603 | 580.895 | 424.858 | -46.668 |
| tabm_seed42 | executive | 1475 | 564.034 | 408.861 | -35.896 |
| tabm_seed43 | floor_area_above_130sqm | 1603 | 579.330 | 414.671 | 42.443 |
| tabm_seed43 | executive | 1475 | 563.980 | 402.404 | 51.855 |
| tabm_seed44 | floor_area_above_130sqm | 1603 | 575.129 | 419.903 | -36.825 |
| tabm_seed44 | executive | 1475 | 558.790 | 404.926 | -26.329 |

These groups are diagnostic only and did not choose the model. Higher absolute errors for larger properties can partly reflect higher rent levels; they do not by themselves establish a size-adjustment defect.

## Verification, artifacts and limits

Three targeted tests pass: fitting-only preprocessing and unknown categories, independent per-member loss, and official-model checkpoint replay with residual reconstruction. Full-schema GPU smoke tests passed for plain and PLE models. Every calibration checkpoint replays exactly after reload. All 28,995 outer IDs, months and actual labels align with the TabPFN references; predictions are finite and all losses are recomputed. `verification.json` records matching source, implementation, library, checkpoint and preprocessor hashes. Model weights and preprocessors are under `.cache/tabm_80_20/`; `scores.csv`, `monthly_scores.csv`, `size_group_scores.csv`, `calibration_scores.csv`, per-run learning curves, timing/memory and prediction files are under this report directory.

This repeatedly inspected ten-month development period is not an untouched test. Three final seeds do not replace additional chronological forecast origins. The supplied test extends to longer horizons. This is a bounded comparison of three declared configurations, not evidence about all possible TabM tuning choices. PyTorch peak allocation is not a measurement of resident physical VRAM.

## Reproduction

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-tabm.txt
.venv/Scripts/python.exe -m unittest tests.test_tabm_regression
.venv/Scripts/python.exe analysis/compare_tabm_80_20.py --smoke
.venv/Scripts/python.exe analysis/compare_tabm_80_20.py --stage all
.venv/Scripts/python.exe analysis/summarize_tabm_80_20.py
```

The runner supports `--stage prepare|calibration|final|summarize` and `--candidate`. Completed runs are reused with hash checks; unfinished training resumes optimizer, checkpoint, shuffle and dropout RNG states from the latest epoch. Main log: `reports/tabm_run.log`.

Method reference: [official TabM implementation and example](https://github.com/yandex-research/tabm).
