# TabPFN component adaptation experiments

All scores are RMSE in SGD on the same 28,995 June 2024-March 2025 rows. Inference uses every one of the 121,005 January 2021-May 2024 context rows. The existing 77-column categorical-month feature artifact, category positions, missingness, residual target and temporal baseline are unchanged. No test submission was changed.

## Audit of previously completed experiments

The all-attention rank-8 Q/K LoRA two-view experiment was already complete when this task began: frozen 455.396, LoRA 457.043. It is reused, not rerun. The earlier exported-embedding MLP (460.903) and four-training/four-deployment-view DiagScale (456.545) also already existed. They are distinct from the new native-head and fixed-checkpoint inference-view experiments.

## Prespecified procedure

See `plan.json` for feature, checkpoint and source hashes. All new adapters fit only the 102,829 rows before December 2023. The existing 600 December 2023-May 2024 calibration queries and population weights select epochs and configurations. Training keeps the upstream random episodes within this fitting pool, two training views, 1,024 task rows, 20% supervised queries, seed 42, AdamW weight decay 0.01, eight-epoch cap and patience three. This architectural ablation preserves the existing training sampler; it is not the separate historical-only month-coherent sampler experiment.

The new checkpoint rule evaluates population-weighted one-view deployment RMSE with the exact typed frames and an independently reloaded component. The unmodified checkpoint is eligible. This differs from the older two-view checkpoint rule; a matched CRPS+MSE DiagScale control is therefore retrained alongside MSE-only DiagScale. Components start from the frozen base checkpoint, not the previously winning adapter. All frozen parameter tensors are hashed before and after training.

ICL variants adapt Q/K projections only in all eight ICL blocks (rank 2 or 4), or only the final two (rank 4), using learning rate 0.0003. Native-head tuning trains the pretrained `heads.mlp_regression` and `heads.output_projection`, learning rate 0.00003. Both DiagScale losses use learning rate 0.01. The head is the original predictive-distribution decoder, not a new MLP over exported embeddings.

The fixed-view experiment reuses the *same original winning DiagScale adapter* for one, two and four inference views. Earlier calibration alone selects the view count. All fixed view conditions are reported on the full outer period without choosing a new view from those outer results.

## Earlier calibration and component selection

| name | weighted_rmse_sgd | initial_weighted_rmse_sgd | selected_epoch | epochs_run | trainable_parameters |
| --- | --- | --- | --- | --- | --- |
| diag_crps_mse | 519.598 | 521.864 | 0 | 4 | 8576 |
| diag_mse_only | 519.983 | 521.864 | 0 | 4 | 8576 |
| icl_last2_rank4 | 521.548 | 521.864 | 1 | 5 | 32768 |
| icl_rank2 | 521.205 | 521.864 | 0 | 4 | 65536 |
| icl_rank4 | 521.758 | 521.864 | 0 | 4 | 131072 |
| regression_head | 521.730 | 521.864 | 1 | 5 | 9320328 |

Epoch -1 is the frozen, zero-update checkpoint; epoch 0 is after the first training epoch. Selected components: `{"head": "regression_head", "icl": "icl_rank2", "loss_control": "diag_crps_mse", "loss_mse": "diag_mse_only"}`.



| views | weighted_rmse_sgd |
| --- | --- |
| 1 | 520.848 |
| 2 | 518.146 |
| 4 | 517.155 |

Earlier-selected inference views: **4**.

## Complete outer development results

| model | n | rmse_sgd | mae_sgd | bias_sgd | delta_vs_winning_diag_sgd |
| --- | --- | --- | --- | --- | --- |
| frozen_one_view | 28995 | 455.814 | 329.748 | 0.768 | 0.602 |
| winning_diag_one_view | 28995 | 455.212 | 330.625 | -2.798 | 0.000 |
| existing_all_qk_lora_two_views | 28995 | 457.043 | 333.378 | -14.375 | 1.832 |
| existing_frozen_two_views | 28995 | 455.396 | 327.794 | 10.792 | 0.184 |
| diag_crps_mse | 28995 | 455.122 | 330.872 | -9.000 | -0.090 |
| diag_mse_only | 28995 | 455.071 | 330.790 | -8.428 | -0.141 |
| icl_rank2 | 28995 | 455.472 | 330.472 | -5.549 | 0.260 |
| original_diag_2view | 28995 | 456.032 | 328.944 | 14.978 | 0.820 |
| original_diag_4view | 28995 | 455.376 | 327.833 | 14.912 | 0.165 |
| regression_head | 28995 | 455.990 | 331.042 | -6.986 | 0.778 |

Results of the prespecified choices:

| model | rmse_sgd | delta_vs_winning_diag_sgd |
| --- | --- | --- |
| diag_crps_mse | 455.122 | -0.090 |
| diag_mse_only | 455.071 | -0.141 |
| icl_rank2 | 455.472 | 0.260 |
| original_diag_4view | 455.376 | 0.165 |
| regression_head | 455.990 | 0.778 |

The direct MSE-only minus matched CRPS+MSE control difference is **-0.051 SGD RMSE**; paired month-bootstrap 95% interval **[-0.380, +0.298]**. This is the loss ablation; a difference from the older adapter also includes the changed checkpoint-selection rule and explicit PyTorch initialization seed.

| model | months_improved_vs_winning_diag | bootstrap_median_delta_sgd | bootstrap_2.5pct_sgd | bootstrap_97.5pct_sgd |
| --- | --- | --- | --- | --- |
| diag_crps_mse | 5 | -0.083 | -0.580 | 0.375 |
| diag_mse_only | 6 | -0.138 | -0.784 | 0.491 |
| icl_rank2 | 5 | 0.268 | -0.433 | 0.950 |
| original_diag_2view | 3 | 0.821 | 0.188 | 1.498 |
| original_diag_4view | 4 | 0.168 | -0.465 | 0.795 |
| regression_head | 1 | 0.782 | 0.026 | 1.499 |

The intervals use 20,000 paired resamples of ten complete months with seed 42. They describe this development period, which has been used repeatedly for prior model design; they are not an untouched estimate of future-month or Kaggle performance. Only one training seed and a small, declared hyperparameter set were tested. The test extends beyond this ten-month horizon.

## Recommendation

The numerically lowest RMSE among the declared family/view choices is `diag_mse_only` at **S$455.071**, a difference of **-0.141** from the previous winner. Interpret its magnitude together with the month table and intervals above. A small gain on this repeatedly examined development holdout does not establish a dependable deployment improvement. Retain the existing submission while confirming promising changes at other forecast origins or training seeds. No outer score selects a new adapter hyperparameter, epoch or view count.

## Verification and reproduction

Five component/LoRA/DiagScale unit tests pass. Full-model smoke runs verify saved-checkpoint replay and gradients for each component type. Every saved new outer prediction has 28,995 unique, correctly aligned IDs and finite predictions; all reported losses are recomputed from those predictions. `verification.json` records source/adapter hashes, frozen-weight checks and prediction hashes. `epoch_trace.json`, `calibration_scores.csv`, `monthly_scores.csv`, per-run timing files and the chart preserve the full comparison. Peak memory is PyTorch allocated memory, not resident physical VRAM.

```powershell
.venv/Scripts/python.exe -m unittest tests.test_tabpfn_component_adaptation tests.test_lora_qk_tabpfn tests.test_diagscale_tabpfn
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --smoke
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --stage all
.venv/Scripts/python.exe analysis/summarize_tabpfn_components.py
```

For lower cumulative GPU memory use, the actual full scoring run uses fresh processes for each model after calibration:

```powershell
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --stage calibration
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --stage views
.venv/Scripts/python.exe analysis/score_tabpfn_components_individually.py
.venv/Scripts/python.exe analysis/summarize_tabpfn_components.py
```

The seeded training suite and isolated outer-scoring logs are `reports/tabpfn_components_run.log`, `reports/tabpfn_components_outer_run.log`, and `outer_*.log` in this report directory. The interrupted unseeded pilot, preliminary smoke checks and their old plans are explicitly named separately and excluded from all result tables. One outer run resumed saved batches; its `seconds_this_attempt` is not an end-to-end benchmark.

Completed training runs and inference batches resume without overwriting completed results. To run one calibration candidate, use `--stage calibration --candidate icl_rank2` (or another candidate from `--help`). Existing feature code, adapters, reports and submission CSVs are preserved.
