# Why the DiagScale adaptation initially underperformed

## Confirmed categorical preprocessing mismatch

The earlier fine-tuning calls passed pandas categorical columns, but TabPFN's `FinetunedTabPFNRegressor.fit` validates them into NumPy arrays before constructing its training and checkpoint tasks. The previous DiagScale scripts did not set `categorical_features_indices` explicitly. Consequently, the training and built-in checkpoint path lost the nine categorical dtype markers, while our regular frozen and outer prediction paths saw pandas categoricals. High-cardinality `SUBZONE`, `BLOCK_ID`, and `STREET_ID` were especially affected. This is a real inconsistency in our experiment setup.

On the same 600-row earlier-month calibration set, one-view frozen RMSE was **S$521.864** with the intended pandas category handling, but **S$517.908** when the inputs were converted to NumPy without category indices. Passing the nine explicit category positions to the NumPy path restored **S$521.864**. These are different feature interpretations, not a model improvement. The former built-in checkpoint metrics therefore cannot be compared directly with the frozen pandas prediction metrics.

The controlled correction uses the same TabPFN-3.5-Fast checkpoint, residual target and 77 features, chronological periods, one-view deployment, and tuned DiagScale hyperparameters. It supplies the nine categorical indices during both adapter training and outer inference. The script is `analysis/check_diagscale_categorical_alignment_80_20.py`.

## Controlled results

The adapter fits on 102,829 rows before December 2023, selects an epoch with 600 sampled December 2023–May 2024 rows, and predicts 28,995 June 2024–March 2025 rows with all 121,005 preceding rows in its inference context. Outer labels were not used to train or select the adapter. The outer holdout has been examined in earlier project work and is a development comparison.

| Setup | Earlier-month weighted calibration RMSE (S$) | Outer RMSE (S$) | Outer MAE (S$) | Outer bias (S$) |
| --- | ---: | ---: | ---: | ---: |
| Frozen one-view TabPFN-3.5-Fast | 521.864 | 455.814 | 329.748 | +0.768 |
| Previous tuned DiagScale, missing categorical indices during training | 521.136 | 456.297 | 331.742 | −9.592 |
| Corrected DiagScale, categorical indices aligned | **520.848** | **455.212** | 330.625 | −2.798 |

The corrected adapter lowers outer RMSE by **S$0.602 (0.13%)** versus frozen and **S$1.085** versus the previous tuned adapter. It raises MAE by **S$0.876** versus frozen. It improves RMSE in seven of ten outer months. This correction explains why the prior implementation could look worse, but the resulting gain is small and depends on the metric. `predictions.csv` has all 28,995 unique row IDs and no missing values; its recomputed metrics match `scores.csv`.

The corrected training used CRPS + MSE, learning rate 0.01, weight decay 0.01, two training and checkpoint views, 1,024 rows per sampled fine-tuning task, BF16, and an eight-epoch cap with patience three. The selected checkpoint was epoch 2 in one-based counting. It trained 8,576 DiagScale parameters in 12 modules; the backbone remained frozen. Fine-tuning peaked at 11.83 GiB allocated VRAM and outer inference at 8.29 GiB, so the RTX 4080 completed this run.

## Remaining likely limits

- The frozen engineered rental-history baseline is already strong. The adapter has little remaining signal to recover, and its tiny RMSE gain is smaller than changes that might arise from a different forecast period. We have not measured uncertainty across multiple chronological periods.
- Fine-tuning samples random 1,024-row context/query tasks within the earlier period, while deployment conditions on a 121,005-row context and forecasts later months. The training objective therefore does not reproduce the long-context, forward-time task. This is a plausible generalization limit, not isolated by the categorical correction.
- The earlier full-80%-data MSE-only refit worsened RMSE to S$458.272 with a +S$32.165 bias. The corrected experiment did not repeat that full-data refit, so its S$455.212 result applies to the earlier-month adapter and full inference context only.
- This is a port of the method to TabPFN-3.5-Fast with one deployment view. The [paper](https://arxiv.org/pdf/2610.12005) and [released code](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models) evaluated a TabPFN-3 recipe with a larger default task and more deployment views. Published average improvements do not guarantee improvement for this rental dataset.

The corrected result supports keeping frozen TabPFN as a simple benchmark. If RMSE is the target metric, the aligned DiagScale adapter is a marginal improvement on this development split; it needs another chronological holdout before treating that gain as reliable.
