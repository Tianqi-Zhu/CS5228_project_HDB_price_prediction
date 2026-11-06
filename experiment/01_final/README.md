[Chinese version](README.zh-CN.md)

# Final model selection and report conclusions (2026-11-05)

**Among the three submitted candidates in the user screenshot, recommend `kaggle_structure_blend.csv`,displayed Kaggle 415.37529.** Identical [final/submission.csv](../../output/submissions/final/submission.csv) passes hash/50,000-row checks.

## 1. Consistent comparison of three submissions

| Policy | Local annual RMSE | Local recent RMSE | Displayed Kaggle score | Report role |
|---|---:|---:|---:|---|
| Main history-feature blend | 451.1810 | 442.9025 | 415.86912 | Main reference |
| Supplied-macro augmentation | 451.2081 | 442.8892 | 415.78232 | Macro ablation |
| Grouped blending + PLR-FT | 451.0295 | 441.8345 | 415.37529 | Recommended final |

Final Kaggle score improves history by 0.49383 and macro by 0.40703. It also beats both on corresponding annual/recent validation, supporting consistent selection among three. Differences are small; no statistically significant or guaranteed unseen-private-board superiority claim.
User submission screenshots supply scores without public/private identification; call them displayed Kaggle scores. Local annual is35,158 April2024–March2025 rows, recent 28,995 June2024–March2025. They overlap and underwent repeated selection, so are not independent tests.

## 2. Precise final-method description

Final method is a **history-driven grouped blend of trees and Transformers**,not one Transformer. Twelve channels include LightGBM, CatBoost, XGBoost, original FT, pruned36-month FT four-seed mean, and PLR-FT. The seed mean is one channel; 15 underlying predictors are not15 architectures.
Inputs come from train.csv/supplied auxiliary data: past rents at block/street/area/type levels, market-relative prices, and 12/24/36-month cohort counts/distributions. New neural models remove five36-month fields and predict residuals from an adjusted historical baseline. PLR-FT uses width128, three attention blocks and learned periodic numerical embeddings.
Blend at most four24-/36-month support groups. Center historical errors by month; nonnegative weights sum1 with channel cap0.6. Global weights shrink to original, groups to global, regularization0.003. Groups below500/unseen groups use global weights. Inference does not read query rents.
Submission base models fit all 150,000 supplied labels. Blend fitting uses 53,135 historical out-of-time rows from 2023-10 through 2024-03 and 2024-04 through 2025-03, equally weighting total loss by origin group, with labels through 2025-03. Predict all 50,000 test rows without hidden rents, external answer matching or manual filling.

## 3. Report narrative and controls

First compare classical/neural methods on shared inputs, then explain history/auxiliary increments with a fixed-model ladder, and finally compare these three complete submissions. Retain history as strong reference, macro as auxiliary ablation, structure as final. Differing-input standalone scores are not fair architecture rankings.
Retain bounded FT linear/PLE/PLR, DCN and expert controls to show complexity does not universally win. PLR is not the best standalone network, but complements the fixed-objective/regularization/grouping blend. Supplementary tables retain other structures/negative results without every search configuration entering the body.
The three Kaggle submissions compare pipelines. History pruning, seed averaging, grouped blending and PLR all changed; not all 0.49383 gain belongs to PLR. Matched PLR addition changes round 9 recent/annual 441.9591/450.8650 to 441.8345/451.0295:recent−0.1245, annual+0.1645. This supports complementarity, not stable standalone replacement. Round 9 is absent from the screenshot's three submissions; retain its better annual result.

## 4. Suggested results wording

> We select a history-driven grouped tree/Transformer ensemble. Its annual/recent RMSEs451.0295/441.8345 outperform the history and supplied-macro pipelines on matched windows; its displayed Kaggle 415.37529 is lowest among three submissions, 0.49383 below history. Anniversary histories supply the main predictive foundation; conditional fusion and periodic embeddings add complementarity. Reused development windows and submission-informed selection make these development/submission comparisons, not independent generalization tests. Controlled studies separately address macro and architectural contributions.

## 5. Reproduction and evidence

- [Final selection](submission_selection.json) and [three-submission comparison](submission_comparison.csv).
- [Final model/prediction inventory](../../output/submissions/components/round_10_final_structure/summary.json), [full-data blend weights](../../output/submissions/components/round_10_final_structure/blend_model.json).
- [Round 9 blending controls](../04_optimization_history/09_grouped_blending.md), [round 10 architecture controls](../04_optimization_history/10_plr_transformer_final.md).
- [Architecture/blend reload checks](../../output/experiment_runs/round_10_plr_transformer_final/verification.json):23 base models and 27 weight sets replay; nine tests pass. New final networks also complete full-data refitting/reloading.
