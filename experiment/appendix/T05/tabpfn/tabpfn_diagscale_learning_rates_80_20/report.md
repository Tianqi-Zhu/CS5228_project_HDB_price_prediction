# DiagScale learning-rate comparison on TabPFN-3.5-Fast

The corrected DiagScale setup was rerun at learning rates **0.003** and **0.005**, alongside the prior corrected **0.01** result. The **0.05** run was already on its last outer prediction batch when the request to stop it arrived, so it finished; no additional 0.05 trials were run. Each adapter used the same 77 input features, explicit indices for all nine categorical columns, 102,829 pre-December-2023 fitting rows, 600 December-2023–May-2024 calibration rows, TabPFN-3.5-Fast checkpoint, residual target, CRPS + MSE loss, AdamW weight decay 0.01, two training/checkpoint views, 1,024 rows per sampled training task, eight-epoch cap, patience three, and one-view deployment. Learning rate was the only changed setting.

All runs were evaluated on the same 28,995 June-2024–March-2025 validation rows, with the full 121,005 earlier rows used as inference context. These outer labels did not train the adapters or choose their epochs. This validation period has been examined in earlier project experiments, so the comparison is development evidence, not an untouched test estimate.

| Learning rate | Selected epoch (1-based) | Weighted earlier-month calibration RMSE (S$) | Outer RMSE (S$) | Outer MAE (S$) | Outer bias (S$) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | — | 521.864 | 455.814 | **329.748** | +0.768 |
| **0.003** | 6 | 520.607 | 456.133 | 331.187 | −1.823 |
| **0.005** | 6 | 521.892 | 457.057 | 331.798 | +2.186 |
| **0.01** | 2 | 520.848 | **455.212** | 330.625 | −2.798 |
| 0.05, completed before cancellation | 1 | **519.945** | 460.074 | 338.544 | −29.308 |

Among the adapted models, **0.01 has the lowest outer RMSE**. It beats frozen RMSE by S$0.602 (0.13%) but has higher MAE. Both newly requested lower rates have higher outer RMSE and MAE than frozen. The 0.003 and 0.05 calibration gains did not carry forward to the outer period; thus the earlier-month 600-row calibration ranking is not reliable enough to pick a learning rate by itself. The large negative bias at 0.05 is consistent with a too-aggressive update, although this experiment cannot isolate the mechanism.

Each saved `predictions.csv` contains 28,995 unique row IDs with no missing values; RMSE recomputed from those rows matches each `scores.csv`. The two requested runs exited successfully. Raw outputs are in `reports/tabpfn_diagscale_categorical_alignment_lr0003_80_20/` and `reports/tabpfn_diagscale_categorical_alignment_lr0005_80_20/`. The corrected 0.01 reference is in `reports/tabpfn_diagscale_categorical_alignment_80_20/`. Reproduce a rate with `python analysis/check_diagscale_categorical_alignment_80_20.py --learning-rate RATE`.
