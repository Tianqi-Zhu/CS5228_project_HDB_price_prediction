# DiagScale parameter tuning on the chronological 80/20 rent split

## Selection and evaluation

The TabPFN-3.5-Fast categorical-month residual model uses the same 77 aligned temporal/spatial/categorical features as the earlier run. All candidates fit adapters on **102,829** rows through November 2023. Checkpoints use **600** label-independent sampled transactions from December 2023–May 2024 (100 per month) with histories and temporal baselines rebuilt at the November 2023 origin. The winner is selected by one-view rent RMSE weighted to the full month sizes. Only then is it evaluated on the fixed **28,995** June 2024–March 2025 outer holdout, using all **121,005** earlier rows as inference context. Outer labels never train or select an adapter. The outer holdout has been examined in prior project experiments, so this is a development comparison.

All new candidates use zero-initialized DiagScale on the TabPFN-3.5-Fast checkpoint, a frozen backbone, AdamW weight decay 0.01, BF16 autocast, CRPS weight 1 plus mean-MSE weight 1, eight-epoch cap, and early-stopping patience three. The original paper uses TabPFN-3 with up to 30 epochs, patience eight, and two training/checkpoint views; this is a resource-bounded parameter search on our different checkpoint. The outer comparison uses one view to match the project's frozen baseline. [Paper](https://arxiv.org/pdf/2610.12005); [released configuration](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/main/configs/tabpfn3/diagscale.yaml).

## Earlier-month calibration

| CRPS + MSE candidate | LR | Training / checkpoint views | Task rows | Best epoch | Weighted RMSE (S$) | Fit time | Peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One view | 0.03 | 1 / 1 | 2,048 | 2 | 525.332 | 316 s | 11.80 GiB |
| One view | 0.01 | 1 / 1 | 2,048 | 2 | 523.188 | 336 s | 11.80 GiB |
| **Two views** | **0.01** | **2 / 2** | **1,024** | **1** | **521.136** | 551 s | 11.83 GiB |

For context, the earlier MSE-only one-view adapter at LR 0.03 scored S$523.808 on the same weighted calibration rows, but used 4,096-row tasks and a five-epoch schedule. The original CRPS + MSE trial at 4,096 rows ran out of GPU memory during backward; 2,048 rows worked with one view, and the two-view trial used 1,024 rows to manage memory. Thus the view comparison also changes task size. We did not run the two-view LR 0.03 combination after the lower LR was better in the one-view comparison. See `candidate_scores.csv` and each candidate's `calibration.json` for the checkpoint trajectories and exact settings.

## Outer holdout

| Model | RMSE (S$) | MAE (S$) | Bias (S$) |
| --- | ---: | ---: | ---: |
| Frozen one-view TabPFN-3.5-Fast | **455.814** | **329.748** | +0.768 |
| Earlier MSE-only DiagScale | 457.799 | 332.564 | −5.819 |
| **Selected CRPS + MSE DiagScale** | **456.297** | **331.742** | −9.592 |

Tuning reduced outer RMSE by **S$1.503** versus the earlier DiagScale adapter, but the frozen model remains better by **S$0.483 RMSE** and **S$1.994 MAE**. The tuned adapter wins RMSE in four of ten months; these wins do not offset losses in the other six. Better earlier-month calibration therefore did not produce an outer holdout gain. This does not rule out other settings or a full TabPFN-3 implementation. See `scores.csv`, `monthly_scores.csv`, and the ignored row-level `predictions.csv`.

Run the fixed candidates with `python analysis/tune_tabpfn_diagscale_80_20.py --candidate NAME`, then use `--score-best`. The saved `.cache/tabpfn_diagscale_80_20/inner_features.joblib` and licensed 3.5-Fast checkpoint are required. Candidate adapter tensors live in the ignored `.cache/tabpfn_diagscale_tuning_80_20/` directory. `selection.json` records the completed candidate scores and the untested combination.
