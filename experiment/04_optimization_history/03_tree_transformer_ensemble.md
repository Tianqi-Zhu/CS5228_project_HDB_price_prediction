[Chinese version](03_tree_transformer_ensemble.zh-CN.md)

# Round 3:controlled feature and model routes

Use fixed `data/train.csv`/`data/auxiliary/`; raw files unchanged. No external rent targets, test labels or realized future macro inputs.

Select parameters/weights on 2023-10–2024-03, then refit at each origin with fixed iterations. Retain every validation row.

| Temporal window | Rows | Previous blend RMSE | Current frozen blend RMSE | Difference | Months improved |
|---|---:|---:|---:|---:|---:|
| 2024-04—2025-03 | 35,158 | 452.9386 | 451.3097 | -1.6289 | 10/12 |
| 2024-06—2025-03 | 28,995 | 443.8630 | 443.0052 | -0.8578 | 7/10 |

Cross-type shared history is a separate extension with the same earlier weight selection; it does not replace the original blend above:

| Temporal window | Original round 3 blend | Cross-type extended blend | MAE | Bias |
|---|---:|---:|---:|---:|
| 2024-04—2025-03 | 451.3097 | 451.4509 | 331.0261 | -16.3999 |
| 2024-06—2025-03 | 443.0052 | 443.0735 | 324.7667 | -28.2698 |

A temporal-adaptive blend uses only prediction errors from 2023-10–2024-03 and 2024-04–2024-05, before the 2024-06 origin, with prespecified 50/50 losses and 0.002 shrinkage. Recent RMSE is 443.1081, below the performance of the original wave-three blend; retain it as an unsuccessful attempt.

Lower RMSE does not improve every metric: round 3 annual MAE331.05 versus 328.44, recent 324.82 versus 322.92. Choose according to the actual objective.

Fixed depth8 GPU CatBoost, changing features only:

| Window | Original62-input RMSE | New77-input RMSE |
|---|---:|---:|
| screen | 526.0796 | 525.2100 |
| outer | 454.7682 | 458.7133 |
| june | 445.5774 | 449.5340 |

Retain every window; early standalone improvement does not imply later gain. Full matched62-input model controls:`controlled_model_comparison.csv`.

## Feature engineering

- Add Gaussian-weighted statistics12/18/24/30/36 months earlier over±three months, strictly before each row's month.
- Add historical local-market normalization, low/high shares, min/max ratios, effective support, and rents shrunk toward long-run block means.
- Long horizons query only origin-observed months; no invented missing months. Block/type statistics do not identify tenants or renewals.

## Models and training

- Compare XGBoost depths3/5/7/9, MLP, ResNet and FT on strong shared features; direct versus historical-baseline residual targets.
- New features: LightGBM15/31 leaves, GPU CatBoost6/8; also low-rent classifier/two conditional experts.
- Shorter supervised history, recent weights or two-year-supported samples retain every earlier record in history pools.
- Weak standalone models may still help; weights use earlier windows only, without later retuning.

## Checks and limitations

- Completed 76 earlier configurations; recomputed102 prediction files, reloaded22 selected models; source SHA-256 unchanged.
- Fifteen tests cover causal windows, unobserved months, cross-type normalization, weights, supervised selection and existing boundaries.
- Both later windows are repeatedly inspected, overlapping development data; month-block intervals are diagnostics only.
- Target 400–430 unmet. `verified_metrics.csv` retains all trials/failures, not just minima.

Reproduce:`python -m analysis.experiments.third_wave_training screen`, `third_wave_models screen`, `third_wave_features screen`, `third_wave_mixture screen`,then `third_wave_ensemble select/outer/june` and `third_wave_report`. Preserve locked files/completed results.

## Lowest development ablation


Separately add this round's feature, model, training-strategy or expert group to the previous blend, returning other new weights to it. Reuse fixed weights without refitting.

| Retained new group | Annual RMSE | Recent RMSE |
|---|---:|---:|
| training | 452.7714 | 443.6751 |
| models | 451.1810 | 442.9025 |
| features | 451.6424 | 443.0659 |
| mixture | 452.6346 | 443.6439 |

Lowest development result keeps new model group only: previous blend76.26%, FT15.99%, residual XGBoost5.52%, depth7 XGBoost2.23%. Annual451.1810, recent 442.9025; recent MAE323.7295 versus 322.9187.

**Group selection follows later-window inspection. Earlier-locked full round 3 remains451.3097 annual/443.0052 recent; the ablation minimum is not independent-test performance.**

All eight ablations retain complete validation rows; source labels/months and recomputed/group metrics match. Files:`output/models/round_03_tree_transformer_ensemble/ablations/`.
