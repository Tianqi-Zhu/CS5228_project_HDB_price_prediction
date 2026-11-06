[Chinese version](08_feature_pruning_seeds.zh-CN.md)

# Round 8:actual training for small improvements

Best recent candidate **442.7032** removes five36-month FT fields, refits seeds 42/43/44/45 for three epochs and equally averages, replacing original 15.9872% FT member only. Improves history 442.9025 by **0.1993**.

Annual451.4310 worsens451.1810 by 0.2500. Five/ten recent months improve; descriptive block interval[-0.3017,0.0065] crosses0. Worth submission comparison, not stable-gain proof. See submission record for CSV/full fits.

Explore ensemble stability/history availability without a hard430 target. Repeated local development scores are not independent tests or Kaggle results.

| Policy | Annual RMSE(35,158 rows) | Recent RMSE(28,995 rows) |
|---|---:|---:|
| ft_seedmean_replacement | 451.4310 | 442.7032 |
| ft_seedmean_given_macro | 451.5252 | 442.7710 |
| bag_ft | 451.4458 | 442.8867 |
| ft_replacement | 451.4458 | 442.8867 |
| bag_selected | 451.1810 | 442.9025 |
| previous_history | 451.1810 | 442.9025 |
| ft_replacement_given_macro | 451.4886 | 442.9054 |
| bag_half_exploratory | 451.1839 | 442.9363 |
| bag_full_exploratory | 451.2189 | 442.9967 |
| ridge_predictions_100.0 | 451.0689 | 443.5067 |
| convex_intercept_0.0 | 450.5756 | 443.6517 |
| ridge_predictions_10.0 | 450.6389 | 443.7871 |
| convex_intercept_0.001 | 450.8080 | 444.0298 |
| ridge_predictions_1000.0 | 451.3687 | 444.0614 |
| ridge_support_1000.0 | 451.4145 | 444.1057 |
| ridge_typed_1000.0 | 451.3237 | 444.1115 |
| ridge_support_10.0 | 451.9185 | 444.3781 |
| bag_ft_tabm | 450.8420 | 444.4590 |
| tabm_blend | 450.8295 | 444.6968 |
| ridge_support_100.0 | 451.7346 | 444.7317 |
| ridge_typed_100.0 | 451.4510 | 444.9987 |
| ridge_typed_10.0 | 451.9090 | 445.1720 |
| lgb_support_100 | 451.4666 | 446.1129 |
| cat_seed44 | 455.8232 | 446.3410 |
| cat_seed43 | 457.4130 | 446.5902 |
| lgb_support_300 | 452.0788 | 447.5127 |
| ft_no36_seed44 | 458.3687 | 448.1642 |
| cat_seed45 | 455.5519 | 448.1751 |
| ft_no36_seed43 | 456.7655 | 451.0116 |
| ft_no36_seed45 | 465.6682 | 451.3991 |
| ft_no36 | 465.7112 | 454.1830 |
| tabm_no36_seed43 | 458.9428 | 456.5361 |
| tabm_no36_seed42 | 456.3055 | 460.3527 |
| tabm_no36_seedmean | 456.4700 | 461.2430 |
| tabm_no36_seed44 | 456.5778 | 469.1322 |

## Training and selection

1. Second-level residual model: 37 base predictions (37 actually registered), 13 configurations. Fit on 2023-10–2024-01 and select on 2024-02–2024-03, choosing `ridge_predictions_10.0`. Annual meta-fitting uses only predictions and labels before 2024-04; the June model may add historical April–May records. Normalize origin weights to mean 1 to preserve regularization strength. There are 26 later fits.
2. CatBoost averaging: fixed CPU recipe, 62 inputs, 833 iterations; new seeds 43/44/45 plus original 42, nine fits. Earlier replacement ratios0/0.5/1 select0.0; original leaf weight0.3155610738 retained.
3. Neural pruning: remove five36-month fields. They were constant/all missing and already dropped in early encoders, so early inputs/predictions are identical. Later refit FT (three epochs) and three TabM seeds(18/15/15):eight fits, all rows retained.
4. Fixed combinations: replace FT leaf at 0.1598721528. Blend TabM mean at earlier least-squares weight capped0.3, selected0.300000. Two prespecified combinations also reported without later-label weighting.

## Checks and interpretation

After inspecting single-seed pruning, add seeds 43/44/45 at three fixed epochs across three windows: nine fits, equally averaged with 42 without seed selection. Total39 meta+nine Cat+eight pruning+nine FT=65 fits (excludes submission refits). Reload52 saved models for complete-window replay. Additional tests cover cutoffs, weight scales, categories and pruning.
Initial inconsistent meta-weight scales are archived in `archive_weight_scale_debug`; corrected runs fully repeated. Debug scores do not enter this report/selection.
`bag_half_exploratory`/`bag_full_exploratory` are unselected earlier fixed-ratio controls, not earlier winners. `given_macro` uses supplied prediction-month COE/stocks at existing weights; explain separately from history-only.
Later minima are exploratory development choices. Earlier-locked weights do not erase repeated architecture/window selection; Kaggle testing remains needed. Preserve original submissions.

Full evidence:[metrics](../../output/experiment_runs/round_08_feature_pruning_seeds/metrics.csv), [replay](../../output/experiment_runs/round_08_feature_pruning_seeds/verification.json), [monthly stability](../../output/experiment_runs/round_08_feature_pruning_seeds/monthly_comparison.json), [meta plan](../../output/experiment_runs/round_08_feature_pruning_seeds/plan.json), [seed selection](../../output/experiment_runs/round_08_feature_pruning_seeds/bag_selection.json), [neural controls](../../output/experiment_runs/round_08_feature_pruning_seeds/deep_plan.json).
## Kaggle feedback/new submissions

User reports history/macro near415, normal-method best near411; exact decimals/partition unknown, so no fixed conversion. User-described cheating scores near170 are excluded.

Refit four FT seeds on 150,000 rows, three epochs; reload models and predict50,000 queries. IDs, finite values, saved consistency/source hashes pass. Preserve originals.

- First choice:[history, local442.7032](../../output/submissions/candidates/kaggle_history_ft_no36_mean.csv).
- Additional:[macro, local442.7710](../../output/submissions/candidates/kaggle_macro_ft_no36_mean.csv).
- [Full training/export record](../../output/submissions/components/round_08_ft_seed_average/summary.json).

New files not uploaded; Kaggle scores unknown.
