[Chinese version](10_plr_transformer_final.zh-CN.md)

# Round 10:actual architecture training

Nine early neural configurations cover numerical tokens, explicit crosses, conditional experts and linear-token control; also two asymmetric CatBoost variants. Earlier neural route/epoch selection fixes later durations; trees use original 833 iterations. Neural62 candidates remove five36-month fields and predict24-month-baseline residuals; tree62 window inputs predict rents directly. Equal column counts do not mean equal content.

## Structures/provenance

- FT uses [author RTDL backbone](https://arxiv.org/abs/2106.11959),linear/PLE/PLR tokens from [author embeddings](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md),following [numerical-embedding paper](https://arxiv.org/abs/2203.05556). Capacity96×two versus 128×three. Project adaptation, not bitwise paper reproduction.
- Deep & Cross implements [DCN-V2](https://arxiv.org/abs/2008.13535) full-matrix crosses plus parallel two-layer deep branch. Compare linear+two crosses and PLR+three; project implementation, not unchanged author code.
- Conditional experts: shared PLR/category embeddings, width128 trunk, four/eight regressors, input-only soft gates; MSE+0.001 balance penalty, no true-rent routing.

- Asymmetric [CatBoost policies](https://catboost.ai/docs/en/references/training-parameters/common#grow_policy):Depthwise6 versus Lossguide8/max64 leaves, min leaf20. Match original CPU symmetric inputs, target, 833 iterations, lr0.045, L2=15, seed 42; document other capacity limits, not one-parameter attribution.

## Full-window standalone results

| Representative model | Annual RMSE | Recent RMSE |
|---|---:|---:|
| cat_symmetric_saved | 456.4077 | 446.3793 |
| cat_depthwise6 | 460.1803 | 448.0859 |
| ft_ple_128_3 | 458.2931 | 449.8158 |
| ft_plr_128_3 | 462.0714 | 451.0982 |
| dcn_plr_3 | 461.5278 | 452.1061 |
| ft_linear_96_2 | 461.9852 | 458.9254 |
| moe_plr_8 | 458.6529 | 478.9720 |

Separate standalone/blend metrics. Limited budgets/one seed do not establish universal architecture superiority.

## Increment under fixed blending rules

Fix round 9 support groups, month-centered errors, global/group regularization0.003; change only added predictions. Refit weights per origin without tuning blend hyperparameters. Baseline441.9591/450.8650 matches; rule changes are not architecture gains.

| Added route | Annual RMSE | Recent RMSE |
|---|---:|---:|
| only_ft_plr | 451.0295 | 441.8345 |
| baseline | 450.8650 | 441.9591 |
| only_tree | 450.8034 | 441.9752 |
| only_ft_ple | 450.7508 | 441.9845 |
| only_dcn | 450.8395 | 442.1377 |
| only_ft_linear | 450.7687 | 442.2048 |
| all_new | 450.8987 | 442.5100 |
| all_new_plus_tree | 450.8121 | 442.5193 |
| only_moe | 450.8200 | 442.8225 |

Lowest `only_ft_plr`:recent **441.8345**,annual **451.0295**. Later development selection, not earlier-locked independent-test best.
Versus round 9, recent−0.1245 with seven/ten months improved, but annual+0.1645. PLR is not best standalone, yet complements fixed-rule blending. [Monthly variation](../../output/experiment_runs/round_10_plr_transformer_final/monthly_comparison.json) cannot establish Kaggle improvement.
Refit required models and export [50,000-row CSV](../../output/submissions/candidates/kaggle_structure_blend.csv). User submitted; screenshot **415.37529**. Select among history 415.86912, macro 415.78232 and this candidate.

## Verification scope

Total23 base fits(19 neural, four trees), plus nine inclusion routes×three windows. Base fitting uses fitting labels only, no later query stopping. Blend cutoffs: annualMarch2024, JuneMay2024, fullMarch2025. Earlier windows already selected models/epochs/features; all are reused development, with overlapping later periods.
Earlier neural selection:`all_new`; extra tree controls recorded separately. Reload23 models/27 weights, match baseline and pass nine architecture/cutoff tests. Standalone tables contain different input groups, not fair cross-group architecture rankings.

[All standalone runs](../../output/experiment_runs/round_10_plr_transformer_final/model_metrics.csv), [all blends](../../output/experiment_runs/round_10_plr_transformer_final/ensemble_metrics.csv), [selection](../../output/experiment_runs/round_10_plr_transformer_final/selection.json), [replay](../../output/experiment_runs/round_10_plr_transformer_final/verification.json), [summary](../../output/experiment_runs/round_10_plr_transformer_final/summary.json).

Final report selection:[FINAL_MODEL.md](../01_final/README.md).
