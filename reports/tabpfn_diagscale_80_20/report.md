# DiagScale adaptation of TabPFN-3.5-Fast

The exact categorical-month residual setup from the 80/20 experiment is used: 77 inputs, one estimator, 121,005 full outer context rows and 28,995 outer queries. All rental histories and temporal baselines have causal origins. The earlier December 2023 origin supplies a training-side checkpoint set; outer labels do not train or select the adapter.

Frozen RMSE: **S$455.814**. Adapted RMSE: **S$457.799** (change +1.986). See `scores.csv` and `monthly_scores.csv`.

The optimizer targeted **8,576** zero-initialized diagonal parameters in 12 attention scaling modules; all original weights were frozen. Selected adapter norm: 16.1029. The earlier calibration RMSE was S$523.808. Peak fine-tuning GPU allocation: 11.80 GiB. BF16 autocast was needed because the installed default FP16 path generated NaN adapter gradients on this checkpoint.

This ports the paper's query-key scaling mechanism to TabPFN-3.5-Fast. The paper's published results use TabPFN-3 on TabArena, so they are not a predicted gain for this checkpoint or rental task. The installed fine-tuning loop samples context/query tasks randomly from the pre-December training period; the checkpoint and final outer evaluation are strictly chronological. The outer holdout has already been examined in prior project experiments and remains a development comparison.

Reference: [Ning et al., *Test-Time Compute for Tabular Foundation Models*](https://arxiv.org/pdf/2610.12005) and [released implementation](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models).

## Full 80/20 training refit

The first run above reserved December 2023–May 2024 for checkpoint selection, so its adapter learned from 102,829 earlier rows. To use the full 80% training side for gradient updates, a second run refitted from the same frozen checkpoint on all **121,005** pre-June-2024 rows. It used the two training epochs selected by the earlier chronological calibration set, then scored the same **28,995** June 2024–March 2025 validation rows. Outer labels did not affect the selected epoch or adapter weights.

| Model | Adapter training rows | Outer RMSE (S$) | Outer MAE (S$) | Bias (S$) | RMSE change vs frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen TabPFN-3.5-Fast | 0 | 455.814 | 329.748 | +0.768 | — |
| DiagScale, earlier-month adapter | 102,829 | 457.799 | 332.564 | −5.819 | +1.986 |
| DiagScale, full 80% refit | 121,005 | 458.272 | 327.929 | +32.165 | +2.458 |

The full refit improves MAE by S$1.820 but worsens RMSE by S$2.458 (0.54%). Its RMSE is lower than the frozen model in only June and October 2024; January 2025 rises from S$457.668 to S$469.052. The positive bias is largest in January (+S$100.653). The earlier-month adapter loses in nine of ten months. The local improvement during December–May did not carry into the later holdout, consistent with temporal shift or overfitting to the selected period; this experiment alone cannot distinguish those causes.

The full refit used BF16 autocast on the RTX 4080, **two** selected epochs, 4,096-row training tasks, one estimator, MSE-weighted fine-tuning, and 8,576 trainable diagonal weights across 12 scaling modules. Original backbone weights stayed frozen. Peak allocated GPU memory was 3.18 GiB during refit and 5.04 GiB during outer prediction; the earlier checkpoint-selection run peaked at 11.80 GiB. Thus this run was feasible on the available GPU. The built-in fine-tuning task sampler draws context/query partitions within the training period rather than respecting forecast origin; this is a remaining mismatch with the chronological outer task.

Files: `refit_scores.csv`, `refit_monthly_scores.csv`, `refit_training_audit.json`, `refit_settings.json`, and the row-level `refit_predictions.csv` (ignored by Git). Reproduce the first run with `python analysis/compare_tabpfn_diagscale_80_20.py`, then the full refit with `python analysis/refit_tabpfn_diagscale_80_20.py`. The saved feature bundle and licensed 3.5-Fast checkpoint are required in `.cache/`.

## Parameter audit against the DiagScale release

This is an exploratory DiagScale adaptation on **TabPFN-3.5-Fast**, not a reproduction of the paper's **TabPFN-3** recipe. The zero-initialized per-head/per-dimension multiplier, frozen backbone, AdamW, learning rate 0.03, weight decay 0.01, and BF16 computation follow the released mechanism and TabPFN-3 settings. The 3.5-Fast learning rate was borrowed without a task-specific or backbone-specific sweep.

| Setting | This experiment | Paper / released TabPFN-3 recipe |
| --- | --- | --- |
| Regression training loss | MSE only: CRPS 0, MSE 1 | Released runner leaves the regressor loss at its defaults; the installed 9.2.0 defaults are CRPS 1, MSE 1 |
| Views | 1 training, 1 checkpoint, 1 deployment | 2 training, 2 checkpoint, 8 deployment |
| Epoch cap / early-stopping patience | 5 / 2 | 30 / 8 |
| Context + query rows per training task | 4,096 | Released runner uses the library default; the installed 9.2.0 default is 50,000 |
| Backbone | TabPFN-3.5-Fast | TabPFN-3 |

The one-view deployment preserves a matched comparison with this project's one-view frozen control; it does not test the paper's eight-view predictor. The chronological six-month checkpoint period was deliberately chosen for rent forecasting instead of the released runner's default random 10% training-side validation. Training tasks still sample context and query rows randomly within their fitting period. The full-data refit also used a new two-epoch cosine learning-rate schedule: the selected checkpoint came from a five-epoch schedule, so refitting for two epochs is not an exact continuation of its optimization trajectory. Thus the observed RMSE increases apply to **these settings** and cannot rule out a closer-to-paper recipe or a better tuned 3.5-Fast adapter.

Sources: [paper, Appendix B.1–B.2](https://arxiv.org/pdf/2610.12005), [released DiagScale configuration](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/diagscale.yaml), and [reference runner](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/src/ttc/models.py).
