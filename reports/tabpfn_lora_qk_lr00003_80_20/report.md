# TabPFN-3.5-Fast Q/K LoRA on HDB rent prediction

## Method

The experiment inserts rank-8, alpha-8 LoRA updates into the Q and K linear projections of all 18 attention modules in the installed TabPFN-3.5-Fast checkpoint. V, output projections, and all pretrained weights remain frozen. The LoRA B matrices start at zero, so insertion preserves the original prediction at initialization. The regression task has 299,008 trainable adapter parameters; adapters in the unused classification decoder do not train. BF16 autocast and activation checkpointing keep training within the RTX 4080's 16 GiB VRAM.

It uses the existing 77-feature calendar setup, including 9 explicit categorical columns, and predicts rent residuals relative to the same historical baseline as the DiagScale comparison. Training samples have 1,024 context-plus-query rows, two preprocessing views, CRPS+MSE loss, weight decay 0.01, at most eight epochs, patience three, and seed 42. The inner fit uses 102,829 rows through November 2023. Checkpoints use 600 sampled calibration rows from December 2023 through May 2024. The later June 2024 through March 2025 labels are used only for final evaluation.

## Results

| Learning rate | Selected checkpoint | One-view calibration weighted RMSE (S$) | Result |
| --- | ---: | ---: | --- |
| Frozen | — | 521.864 | Reference |
| 0.003 | Initial, zero adapter | 521.896 | All trained checkpoints worsened internal validation; no useful LoRA update selected. |
| 0.0003 | Epoch 1 | 522.011 | Internal two-view checkpoint loss improved, but one-view calibration did not. |

The matched two-view calibration comparison for learning rate 0.0003 was **520.472 frozen versus 516.704 LoRA** weighted RMSE. This is why the trained checkpoint was evaluated on the outer holdout with two views. The zero-adapter 0.003 run was not scored on the outer holdout because early stopping restored the frozen model.

| Outer 80/20 chronological holdout, two views | Rows | RMSE (S$) | MAE (S$) |
| --- | ---: | ---: | ---: |
| Frozen TabPFN | 28,995 | 455.396 | 327.794 |
| Q/K LoRA, learning rate 0.0003 | 28,995 | 457.043 | 333.378 |

The Q/K LoRA result is **S$1.647 RMSE worse** than its matched frozen baseline. It is worse in 9 of 10 validation months; November 2024 is effectively tied. The earlier one-view frozen setup scored about S$455.21 RMSE, so this LoRA configuration does not improve the best TabPFN result. The inner two-view improvement did not generalize to the later period. Possible causes include temporal drift and fitting the small, sampled calibration period, but the experiment does not isolate either cause.

Training peaked at 11.84 GiB allocated GPU memory. Outer two-view inference peaked at 9.12 GiB. The GPU was sufficient for this rank-8 experiment.

## Files and reproduction

- Adapter implementation: `analysis/experiments/lora_qk_tabpfn.py`
- Experiment runner: `analysis/compare_tabpfn_lora_qk_80_20.py`
- Identity/gradient test: `tests/test_lora_qk_tabpfn.py`
- Experiment summaries: `calibration.json`, `calibration_2views.json`, `scores_2views.csv`, `monthly_scores_2views.csv`, `timing_2views.json`
- Row predictions and adapter weights are kept in ignored report/cache paths. The report JSON records input and checkpoint SHA-256 hashes.

Reproduce the selected trial with:

```powershell
.venv/Scripts/python.exe analysis/compare_tabpfn_lora_qk_80_20.py --learning-rate 0.0003 --calibration-only
.venv/Scripts/python.exe analysis/compare_tabpfn_lora_qk_80_20.py --learning-rate 0.0003 --calibration-two-views
.venv/Scripts/python.exe analysis/compare_tabpfn_lora_qk_80_20.py --learning-rate 0.0003 --outer-only --outer-views 2
```

The 0.003 control is reproduced with the first command using `--learning-rate 0.003`. These results are one seed and one chronological holdout; they do not establish that all LoRA ranks or learning rates would fail.
