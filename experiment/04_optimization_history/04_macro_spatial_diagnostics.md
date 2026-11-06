[Chinese version](04_macro_spatial_diagnostics.zh-CN.md)

# Round 4:exploration toward local RMSE 430

Reference442.9025 on identical28,995 June2024–March2025 rows; target 430. Train/supplied aux only, no public rent answers or difficult-row deletion.

Reducing RMSE from 442.9025 to 430 requires a further 2.91% RMSE decrease, or about 5.74% in MSE. No saved same-submission local/Kaggle pairs validate a fixed 20-point conversion. The local target of 430 does not guarantee a Kaggle score of 411.

## New methods/results

| Route | Earlier-window configurations | Approach |
|---|---:|---|
| distribution | 8 | Predict rent-band probabilities and integrate fitting-band means; compare absolute, market-normalized, local-baseline targets/band widths |
| experts | 8 | Fit type-specific LightGBM/CatBoost; rare types fall back to frozen predictions |
| flows | 8 | Use unlabelled query-batch counts and historical cohort counts only; requires batch inference, not target rents |
| horizons | 18 | Simulate0/3/6/9/12-month delays; freeze histories/cohorts; raw/residual/ratio targets with unchanged total weight per source row |
| known_macro | 12 | Separate scenario: supplied query-month COE/stock observations, with rent labels still frozen at origin |
| spatial | 12 | Other-block comparables from 12/24/36 months earlier±one month; match location, area, lease year and exclude own block |

Select parameters, early-stopping lengths and blend weights only on 2023-10–2024-03 predictions; then refit for fixed rounds before the 2024-04 or 2024-06 origins. Prespecify pure-RMSE and fixed 0.001-shrinkage objectives, both with nonnegative weights summing to 1 and no intercept.

| Window | Policy | Previous best RMSE | Current RMSE | Difference | MAE | Months improved |
|---|---|---:|---:|---:|---:|---:|
| outer | conservative | 451.1810 | 451.5005 | +0.3195 | 330.9300 | 5/12 |
| outer | rmse | 451.1810 | 451.5241 | +0.3431 | 331.2677 | 5/12 |
| june | conservative | 442.9025 | 443.4141 | +0.5116 | 324.7150 | 3/10 |
| june | rmse | 442.9025 | 443.3215 | +0.4190 | 324.7018 | 3/10 |

### Separate supplied-macro control

COE/stock files cover test months. Query-month observations are within supplied data but differ from genuinely unknown future deployment conditions. Report separately; query rent labels remain unavailable.

| Window | Blend objective | Previous best RMSE | Known-macro RMSE | Difference |
|---|---|---:|---:|---:|
| outer | conservative | 451.1810 | 451.2081 | +0.0270 |
| outer | rmse | 451.1810 | 451.2310 | +0.0499 |
| june | conservative | 442.9025 | 442.8892 | -0.0133 |
| june | rmse | 442.9025 | 442.9041 | +0.0016 |

Without future macro observations, new best recent blend **443.3215**; target 430 unmet. Retain earlier442.9025 rather than calling a failed trial a new optimum.

## Errors and feasibility

In the recent best model, 868 rents≤1800 are about 3% of rows but22.5% of SSE. Actual-rent groups are diagnostics, not inference routing. Executive flats contribute about 7.4% of error; fixing that group alone cannot easily close the gap.

Inputs omit unit, renovation, tenant and contract status; same-month/block/type rents differ. Target 430 merits testing but is neither guaranteed nor disproved by duplicate-group standard deviations. Distinguish learnable missing information from random/unobserved variation.

Recent/annual windows are repeatedly used development sets; continued selection biases minima. Kaggle Public is not final Private. Repository records411.75928 on 2026-10-24, but current rankings were not independently accessible by browsing; no live-rank verification claim.

## Verification and artifacts

Checked78 prediction files'IDs, months, labels, counts/metrics; reloaded12 selected models, reconstructed eight blends; raw hashes match and 23 tests pass.

- `verified_metrics.csv`:recomputed successful configurations, including deterioration.
- `baseline_error_groups.csv`:grouped error diagnostics.
- `ensemble_selection.json`:earlier-locked members/weights.
- `ensemble_comparison.csv`:matched-window442.90 controls.
- `summary.json` / `verification.json`:target status/check records.

## Full-refit submission files


Refit the best history blend on 150,000 labels with fixed durations; predict50,000 queries. Save ten base and one known-macro member; main reloaded predictions match. Test histories exclude same-month labels row by row and freeze after fitting.

| File | Corresponding local development RMSE | Information setting |
|---|---:|---|
| `output/submissions/components/history_macro_ensemble/submission.csv` | 442.9025 | Main historical policy |
| `output/submissions/components/history_macro_ensemble/submission_given_macro.csv` | 442.8892 | Supplied prediction-period macro augmentation |

Both files contain 50,000 rows with `Id,Predicted` columns and original test ID order. Neither was uploaded; actual Kaggle scores are unknown. Table scores describe local temporal-validation methods, not measured errors on these 50,000 unlabeled queries.

First establish a local/Kaggle pair for the main policy, then choose horizon/error targets for further work. File/input SHA-256:`submission_export.json`.

Reproduce:`python -m analysis.experiments.best_ensemble_submission`; reuse completed components by input hashes/fixed durations.
