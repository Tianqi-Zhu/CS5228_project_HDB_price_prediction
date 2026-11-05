# Four-view DiagScale experiment at learning rate 0.01

The corrected TabPFN-3.5-Fast DiagScale experiment was rerun with **four training and checkpoint views** instead of two. Learning rate 0.01, CRPS + MSE loss, weight decay 0.01, 1,024-row sampled training tasks, eight-epoch cap, patience three, nine explicit categorical indices, the 77-feature residual setup, and all data periods were held fixed. The four-view run completed at the original task size on the RTX 4080. Checkpoint selection chose epoch **4** (one-based), versus epoch **2** for the two-view run. The backbone remained frozen and 8,576 DiagScale parameters were trainable.

The adapter fit used 102,829 rows before December 2023. Checkpoints used 600 sampled December 2023–May 2024 rows. The outer validation period is June 2024–March 2025: 28,995 queries with all 121,005 prior rows as inference context. Outer labels were used only for evaluation, although this period has been examined in earlier project experiments.

| Training / checkpoint views | Deployment views | Earlier-month weighted calibration RMSE (S$) | Outer RMSE (S$) | Outer MAE (S$) | Outer bias (S$) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen TabPFN baseline | 1 | 521.864 | 455.814 | 329.748 | +0.768 |
| DiagScale 2 / 2 | 1 | 520.848 | **455.212** | 330.625 | −2.798 |
| DiagScale 4 / 4 | 1 | 524.523 | 457.416 | 333.258 | −10.189 |
| DiagScale 4 / 4 | 4 | **516.157** | 456.545 | **329.530** | +7.961 |

The **controlled training-view comparison** is the two middle rows: with one deployment view in both, changing training/checkpoint views from two to four worsened outer RMSE by **S$2.205** and MAE by **S$2.633**. Four-view deployment of the same four-view-trained adapter then reduced its outer RMSE by **S$0.872** and MAE by **S$3.727**, but outer RMSE remained **S$1.333** above the best two-view-trained, one-view-deployed setup and **S$0.731** above the frozen one-view baseline. Its MAE was S$0.218 below the frozen baseline. The calibration improvement from four-view deployment did not translate into the lowest outer RMSE.

The checkpoint MSE values for two and four views average different preprocessing ensembles, so their absolute values are not a direct comparison of rent accuracy. We did not run a frozen four-view baseline or a two-view-trained adapter with four-view deployment; those would be needed to isolate how much of the four-view deployment gain is generic ensembling. The four-view deployment pass completed with the full context, but its reported peak PyTorch allocation was **17.92 GiB** under Windows GPU memory management, so this configuration has limited memory headroom.

The saved four-view `predictions.csv` has all 28,995 unique row IDs and no missing values. RMSE and MAE recomputed from it match `scores.csv`. Training and one-view deployment outputs are in `reports/tabpfn_diagscale_categorical_alignment_4views_80_20/`; four-view deployment outputs are in `reports/tabpfn_diagscale_categorical_alignment_4views_inference_80_20/`. Reproduce training and the controlled one-view comparison with `python analysis/check_diagscale_categorical_alignment_80_20.py --learning-rate 0.01 --views 4`; score the saved adapter with four deployment views using `python analysis/score_diagscale_four_view_inference_80_20.py calibration` and then `outer`.
