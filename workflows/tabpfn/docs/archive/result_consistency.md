[Chinese version](result_consistency.zh-CN.md)

# Experiment-result consistency checks

These checks distinguish actual conflicts from numerical differences caused by windows, features or model workflows. Repeat machine-readable checks with:

```powershell
conda activate cs5228
python -m analysis.main verify
```

## Confirmed consistency

- CatBoost C is reused exactly in `baseline_mean_catboost` and `baseline_classical_models`, both with RMSE **487.2160049568078**.
- LightGBM C is reused exactly in `baseline_classical_models`, `features_auxiliary_ablation` and the deep-learning summary reference, all with RMSE **485.09281617236115**.
- `baseline_tabm` and `baseline_spatiotemporal` share three historical-tree folds; TabM does not mistake differently split trees for its baseline.
- Current training, test and submission-template SHA-256 values match common input hashes in saved results.
- Required EDA, study and experiment-result files exist.

## Apparent conflicts with different definitions

| Values | Assessment | Reason |
| --- | --- | --- |
| Auxiliary E: 476.9744; old spatial baseline: 477.2036 | No conflict | Both use 2024-04 to 2025-03, but E has 29 unified property/time/building/facility features; old spatial experiments have nine old baseline plus 17 spatial features, with different time, age and furnishing definitions. |
| Deep blend: 466.4699; ReLU MLP: 463.9719 | Not directly comparable | Deep blending evaluates all 12 months; nested-selected ReLU evaluates only 2025-01 to 2025-03. Lower RMSE does not establish superiority on the same window. |
| Latest TabM blend fold: 468.8345; three-fold pooled: 542.7844 | No conflict | 468.8345 covers only the latest 12 months; 542.7844 pools three origins including harder earlier periods. The same explains historical-tree latest 469.6138 versus pooled 543.0956. |
| CatBoost C: 487.2160; old spatial CatBoost base: 487.0542 | No conflict | Same window, different baselines: current C uses seven standard features; old base uses nine including furnishing, cyclical sin/cos and another lease-age definition. |
| Recent EDA grouped-median baseline about 490.889; ReLU grouped-mean baseline 491.267 | No conflict | Both cover Q1 2025 but differ in grouping, cleaning and aggregation implementations. |

## Input hashes

Old ReLU CSV byte-level SHA-256 differs from current Windows files, but corresponding Git blob IDs match. Logical versions are identical; cross-platform checkout line endings explain the bytes, not replacement by another experiment. The unified verifier checks SHA-256 recorded in current-workspace results.

## Current-environment rerun differences

On 2026-10-04, reruns used independent output directories in `cs5228` (Python 3.11.15, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0):

- CatBoost, LightGBM, auxiliary ablation, grouped panels and four spatial-control tables exactly match saved metrics.
- Restoring source-branch lease-missing indicators and anomalous-lease semantics again selects two ReLU epochs, RMSE 463.7769 versus historical 463.9719, a difference of -0.1951 SGD.
- Economic signals and ablations use unpinned scikit-learn `HistGradientBoostingRegressor` minor versions, with maximum four-fold drift of 3.3847 and 5.2615 SGD respectively.
- Spatiotemporal history also depends on `HistGradientBoostingRegressor`, with maximum fold-RMSE drift 4.0719 SGD; splits, row counts and input Git blobs are unchanged.
- Core temporal-correlation tables match exactly; intermediate monthly series differ by at most 0.001045 without changing four-decimal results.

These are software-version reproduction differences, distinct from validation-window differences. See the [historical verification summary](#historical-verification-summary) for the full run list.

## Conclusion

No mismatched data/splits or overwritten result files were found. Differences follow windows, feature definitions, aggregation or recorded software drift. Prefer same-split, same-environment comparisons rather than ranking the lowest RMSE from every report.

## Historical verification summary

This combines original integration checks and 2026-10-04 reruns, all from early stages. Historical 9/30-test records and current 119 tests are distinct, not replacements; early scores are not final-blend results.

### Numerical and workflow preservation during integration

Main/six-auxiliary EDA summaries match. Integrated EDA's 11 figures and 18 tables were checked by statistical group with float tolerance 1e-10.
pandas 3 string-type displays and median-tie order may differ from earlier versions without changing statistics.

| Model | Pre-migration RMSE | Post-migration RMSE |
| --- | ---: | ---: |
| Mean prediction | 846.7549 | 846.7549 |
| CatBoost A | 761.4192 | 761.4192 |
| CatBoost B | 529.9588 | 529.9588 |
| CatBoost C | 487.2160 | 487.2160 |
| Ridge | 624.8788 | 624.8788 |
| Random Forest | 493.7719 | 493.7719 |
| LightGBM | 485.0928 | 485.0928 |

Unified LightGBM validation is about RMSE 485.09/MAE361.41. That stage completed full fitting on 150,000 rows, 50,000-row Id/Predicted export, identical row predictions after reload, and raw-CSV hashes.
The 15-page EDA PDF was rendered and checked. Nine regression checks passed then, validating migration/workflow rather than new generalization or a Kaggle submission.

### 2026-10-04 independent-environment rerun

Environment: Python 3.11.15, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, CatBoost 1.2.10, LightGBM 4.7.0, PyTorch 2.11.0+cu128 and CUDA12.8. RTX4060 Laptop GPU available; pip check passed.
Outputs used a separate temporary directory without overwriting official metrics.

| Workflow | Result |
| --- | --- |
| `compileall` | Pass |
| `python -m unittest discover -s tests -v` | 30/30 pass |
| `python -m analysis.main verify` | All structure, hash and cross-report reuse checks pass |
| Full EDA | 150,000 training rows, 50,000 test rows; 11 figures, 18 tables; 100% building matching |
| CatBoost A/B/C baselines | Exact match to saved `model_metrics.csv` |
| Ridge/RF/CatBoost/LightGBM comparison | Exact match to saved `family_metrics.csv` |
| LightGBM temporal validation | RMSE 485.09; monthly, region and flat-type errors generated |
| `submit` → save model → `predict` | 50,000 rows, `Id,Predicted` columns; maximum reload prediction difference 0 |
| Deep-model CUDA smoke test | One auxiliary-feature MLP epoch succeeds |
| Deep-result summary | Historical blend RMSE 466.469947 matches saved summary |
| TabM CUDA smoke test | One temporal fold, one epoch, two members succeed |
| Full nested ReLU validation | Two epochs selected; RMSE 463.776866 |
| Auxiliary-feature ablation | All eight stages complete; official metrics match exactly |
| Economic signals and feature ablation | All four forward folds complete; explained sklearn-version drift |
| Grouped temporal panel | Complete; official pooled metrics match exactly |
| Spatial overview, MRT fixed effects, matching and facility fixed effects | All complete; four sampled core tables match exactly |
| Spatiotemporal history model | Three forward folds and final 50,000-row prediction complete; explained sklearn-version drift |
| Temporal-correlation study | Complete; core correlation tables match exactly |
| PDF report | Successfully built from temporary EDA outputs |

Tests use standard-library unittest, without pytest. joblib's fallback warning for old physical-CPU detection is not a model error.
ReLU, economic and spatiotemporal software drift is described under current-environment rerun differences. Historical results remain intact rather than being overwritten by rerun values.
