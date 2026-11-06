[Chinese version](09_grouped_blending.zh-CN.md)

# Round 9:change only blending

Lowest among 31 configurations: recent **441.9591**,annual **450.8650**. Previous442.7032/451.4310; original history 442.9025/451.1810. Base models/features unchanged.

## Policies/information boundaries

Eleven channels: ten old leaves+new four-seed FT. Compare nonnegative sum1 weights, regularized shrinkage, 24-/36-month/type/disagreement groups, limited negatives, frozen versus latest-observed updating. Nonnegative channel cap0.6.
Useful change: subtract monthly mean errors before learning weights to reduce market-level bias effects. Shrink groups to global. Inference combines outputs without query rents or validation-based calibration.
The lowest candidate, `followup_support_pair_centered_0.003`, uses up to four groups based on availability of 24-/36-month anniversary history. Global regularization is 0.003; within-group weights shrink toward global weights with regularization 0.003. Groups with fewer than 500 fitting records, and unseen groups, fall back to global weights.

## Selection scope

Fit on 2023-10–2024-01 and select on 2024-02–2024-03: the early choice is `support24_mse`, with later recent RMSE 442.7381. The reported minimum was selected after inspecting later results and is not an early-locked choice or independent-test winner. Full results retain the initial 22 configurations and nine subsequent group-shrinkage controls.
Annual weights use pre-April2024; June adds April–May predictions from pre-April models. Origin groups each contribute half total loss. Frozen controls use early six months only. Annual/recent overlap and are repeatedly examined.

## Main results

| Policy | Annual RMSE | Recent RMSE |
|---|---:|---:|
| previous_history | 451.1810 | 442.9025 |
| previous_ft_seedmean | 451.4310 | 442.7032 |
| global_mse_0.003 | 450.8960 | 442.8107 |
| global_month_centered_0.003 | 450.9816 | 442.2495 |
| support 24_month_centered | 450.8048 | 442.1071 |
| followup_support24_centered_0.003 | 450.6221 | 442.0146 |
| followup_support_pair_centered_0.003 | 450.8650 | 441.9591 |
| support 24_mse | 450.7252 | 442.7381 |

All31 controls:[comparison.csv](../../output/experiment_runs/round_09_grouped_blending/comparison.csv). Retain positive/negative results and original submissions.

## Stability/checks

- outer: new−previous−0.5660; ten/twelve months improve; descriptive block interval[-0.8809,-0.3242].
- june: new−previous−0.7441; ten/ten improve; descriptive block interval[-1.0774,-0.4845].

Post-search development resampling describes variability, not independently confirmed confidence.31×three=93 global/group fitting workflows include repeated frozen/updated annual controls, not93 independent models. Reload93 JSONs, reproduce metrics/constraints; five tests cover cutoffs, weights, centering, fallback and label-free prediction.

## Full-data submission

Reuse150,000-label base models/50,000 predictions; refit blend weights only on historical out-of-time October2023–March2024 and April2024–March2025, origins equally weighted, labels throughMarch2025. No test labels. Full/June weights differ by cutoff and are saved separately.
- [New blend CSV](../../output/submissions/candidates/kaggle_blend_support_centered.csv).
- [Submission inventory](../../output/submissions/components/round_09_grouped_blend/summary.json).
- [June weights](../../output/experiment_runs/round_09_grouped_blending/weights_june.csv), [full-data weights](../../output/experiment_runs/round_09_grouped_blending/weights_full.csv), [replay](../../output/experiment_runs/round_09_grouped_blending/verification.json).

Not submitted; Kaggle unknown. User reports old history/macro near415 and normal best near411; local gains do not directly convert to leaderboard gains.
