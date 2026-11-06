[Chinese version](01_history_target_search.zh-CN.md)

# New rental-prediction exploration round

Screen **44 new configurations**; refit seven annually using earlier-window selection.
Previous local blend RMSE **465.6623**; new earlier-locked blend **463.7589**.
New−old difference **-1.9034 SGD**. Target 400–430:**not reached**.

## Validation protocol

- Screening: 2023-10 through 2024-03,17,977 rows, information through 2023-09; select settings, epochs, features and weights.
- Annual: 2024-04 through 2025-03,35,158 rows, information through 2024-03; retain every row.
- Outer fits use inner-selected fixed durations without annual early stopping; weights lock before this round's annual scoring.
- Historical labels precede query cutoffs; simulate0/5/11-month delays.
- Distribution models need six-month warm-up; controlled27-input models retain all fitting rows. Cross-group gains are not purely architectural/feature effects.
- Annual validation was repeatedly inspected before; development evidence, not independent testing or Kaggle scores.

## Models and features explored

| Route | Content |
| --- | --- |
| Distributional histories | Block/street/planning-area/town/type means, standard deviations, same-month type-relative means, low/high proportions, last means, 6/12-month statistics |
| Market-target decomposition | Direct rent, residuals or ratios to historical market baseline; ratio models weight by baseline squared to retain SGD squared loss |
| Horizon simulation | Generate0/5/11-month delays per fitting row to match fixed-origin multi-month forecasting |
| Spatial comparables | Same-type neighbors within24 months; 16/48/128-neighbor statistics weighted by location, area, lease year and time decay |
| Stratified prediction | Predict low/normal/high probabilities, then combine conditional means; query rent is unavailable at inference |
| Stability | 6/12/24-month half-lives; relative prices only; remove high-cardinality address categories; 15/31/63 leaves |
| Controlled feature additions | Fix27 inputs; add block distributions, multi-level distributions, relative histories or comparable properties only |

## Annual results

| Configuration | Features | RMSE | MAE |
| --- | ---: | ---: | ---: |
| distribution_cat6 | 122 | 466.3714 | 341.2348 |
| compact_plus_relative_history_15 | 47 | 466.6302 | 341.6711 |
| compact_normal | 118 | 467.0800 | 346.0002 |
| compact_plus_relative_history_63 | 47 | 467.5661 | 342.5580 |
| distribution_residual | 122 | 468.8795 | 346.0106 |
| distribution_ratio | 122 | 469.8058 | 347.1550 |
| mixture_three | 122 | 472.1591 | 348.9446 |
| New locked blend | Multiple models | 463.7589 | 338.5783 |
| Previous local blend | Multiple models | 465.6623 | 336.9480 |

## Blending and interpretation

Weights use earlier17,977-row predictions only; nonnegative, sum1, cap0.6, no intercept. Candidates/weights:`ensemble_selection.json`.

| Member | Weight |
| --- | ---: |
| old_catboost | 36.1128% |
| old_ft_transformer_default | 5.0438% |
| old_hist_gradient_boosting | 4.4362% |
| old_tabnet | 5.3526% |
| old_xgboost | 6.4580% |
| compact_normal | 24.6252% |
| compact_plus_relative_history_15 | 4.7052% |
| compact_plus_relative_history_63 | 0.3056% |
| distribution_cat6 | 4.7445% |
| distribution_ratio | 3.4128% |
| distribution_residual | 0.1884% |
| mixture_three | 4.6149% |

With5000 three-month block resamples of fixed predictions, the descriptive95% new−old RMSE interval is[-2.629,-1.600],reflecting these twelve months only.

No evidence reaches400–430. Low-rent stratification, distribution histories or simulated horizons do not guarantee gains; small development changes need not transfer.
Original street-capitalization audit shows no clear stratification signal; formatting is not a new main route.

## Files and reproduction

- `metrics.csv`:all screened/annual configurations;`summary.json`:main findings.
- `verification.json`:independent recomputation, labels/order/source hashes, model reload and query-label perturbation checks.
- `.cache/round_01_history_target_search/`:rebuildable features;`output/models/round_01_history_target_search/`:local models/row predictions.
- Raw CSVs unchanged; no new feature reads public target-month rents; no competition submission generated/uploaded this round.

```powershell
python -m unittest tests.test_frontier_search -v
python -m analysis.experiments.frontier_search screen
python -m analysis.experiments.frontier_refinement screen
python -m analysis.experiments.frontier_mixture screen
python -m analysis.experiments.frontier_compact screen
python -m analysis.experiments.frontier_ensemble select
python -m analysis.experiments.frontier_ensemble evaluate
python -m analysis.experiments.frontier_report
```

Reuse existing cs5228 Conda environment; no new dependencies.

## Old-model-only blend control

Same earlier-window optimizer/nonnegative/cap rules with old models only yield annual **465.1374**. Adding new models gives **463.7589**,another **1.3785 SGD** gain.
This controls expansion beyond the original three old members, but not one new feature's effect. Added after annual scoring, with weights still fitted solely on earlier labels.

## Same-origin control against earlier best TabPFN

Refit all 12 members at 2024-06; blend RMSE **456.8220** on identical28,995 rows, versus prior TabPFN **455.0710**. No new record.
Two blends reweighted solely with pre-June labels also fail to improve. Results, members, commands:[June controls](../../output/experiment_runs/round_01_history_target_search/matched_june/README.md).
Annual improvement is not an overall record; report both windows separately.
