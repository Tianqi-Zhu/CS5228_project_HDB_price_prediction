# Controlled month encoding and school-calendar proxy experiments

**Retain the aligned integer-month residual control.** Categorical month scores S$455.81 RMSE versus S$455.86 for the control, a S$0.05 difference (0.011%) with improvement in only four of ten months. The earlier-selected calendar candidate scores S$457.03 and fails outer confirmation. This controlled comparison does not support replacing the current model.

The aligned residual baseline, its 37 rental-history inputs, elapsed-month index, rent labels, and all 121,005 context rows are held fixed. The complete 28,995-row June 2024-March 2025 development holdout is predicted for each candidate with the same Fast checkpoint, one estimator, seed 42, and 200-row batches. No new fitted property reference or economic model is introduced.

Integer month is the saved S$455.86 control; its first 400 predictions are replayed with the full context before reuse. Categorical month changes only its dtype. Cyclical month replaces only month-of-year with sine/cosine and retains the continuous elapsed-month index.

All new features are label-free and complete from the first context month, including warmup rows. Existing history warmup missingness is deliberately unchanged across candidates, rather than imputed with future data. Availability audits record zero missing entries for every added column.

Two earlier ten-month folds (September 2022-June 2023 and July 2023-April 2024) retain every pre-origin context row. Query evaluation samples 200 rows per month independently of labels, with month population/sample weights. Fold query histories and baselines are rebuilt at the fold's fixed origin. Sampling is for earlier query evaluation only; neither context nor final 80/20 validation is sampled.

Calendar variants use `cyclical`, selected only on earlier folds before new outer scoring. The calendar block has six recurring monthly proxies: January annual primary-school opening, January/August university intake, and one-/two-month lead windows. These are hypotheses and feature representations, not exact historical term/holiday calendars. University campus distance is not available. The interaction block adds primary-window x primary-school access/count features plus a source-location-missing flag; flat type remains an existing categorical input.

## Earlier selection

| model | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- |
| integer | 607.049 | 471.328 | 108.223 |
| categorical | 602.587 | 470.251 | 77.467 |
| cyclical | 600.915 | 471.572 | 60.186 |
| calendar | 599.762 | 468.585 | 77.596 |
| school_interactions | 614.706 | 475.569 | 135.911 |

Earlier selected candidate: `calendar`. Selection requires at least 0.25% pooled improvement with neither earlier fold more than 0.5% worse. Simpler configurations win within 0.1% of best.

## Full 80/20 results

| model | n | rmse_sgd | mae_sgd | bias_sgd |
| --- | --- | --- | --- | --- |
| integer | 28995 | 455.863 | 329.984 | -2.428 |
| categorical | 28995 | 455.814 | 329.748 | 0.768 |
| cyclical | 28995 | 459.070 | 339.263 | -42.462 |
| calendar | 28995 | 457.034 | 330.222 | 5.274 |
| school_interactions | 28995 | 457.421 | 327.049 | 31.030 |

Recommendation: `integer`. A candidate chosen on earlier folds is retained only if its full outer RMSE improves; a different candidate is not selected by inspecting outer scores. The outer holdout has been examined previously, so this remains a development result. Ten-month performance does not establish accuracy across the supplied seventeen-month test.

Each model-ready bundle is saved under `.cache/tabpfn_v3_features/calendar_80_20_<variant>.joblib`. Use raw `y_train - train_baseline` to fit and add `query_baseline` exactly once to predictions. See feature manifests, availability audits, timings, frozen settings and selections.

Calendar references: [NUS](https://nus.edu.sg/registrar/calendar), [MOE primary-school opening example](https://www.sengkangpri.moe.edu.sg/files/Parent%20Resources/2024/2024_term_1_notification.pdf).

Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_calendar_80_20.py`.

## Verification and interpretation

All 35 relevant tests passed. Saved feature bundles were independently checked against the source: all 76 non-month columns, source row IDs, raw labels and additive residual baselines are unchanged. The first 400 full-context control predictions reproduce the saved control to within S$0.000001. All 28,995 final row IDs are unique, every prediction is finite, and all saved pooled metrics were independently recomputed. Earlier-fold query horizons, label alignment and strictly pre-origin contexts were also checked. No inference job subsampled context or required an out-of-memory retry.

Every newly added feature has zero missing values in both context and query, including the 3,129 January 2021 context rows. Categorical month has all twelve consistent levels and no missing entries. This avoids the comprehensive model's new reference-feature warmup gap. Existing common history warmup is deliberately preserved in every candidate: for example, TEMP_FORECAST_RMSE remains missing in approximately 38.2% of context rows and 0% of query rows. This experiment controls that existing gap; it does not eliminate all missing historical information or fill it using future outcomes. See verification.json and availability_summary.csv.

| model | rmse_sgd | delta_rmse_sgd | mae_sgd | bias_sgd | months_improved | absolute_error_p95 | absolute_error_p99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| integer | 455.863 | 0.000 | 329.984 | -2.428 | 0 | 940.199 | 1527.348 |
| categorical | 455.814 | -0.049 | 329.748 | 0.768 | 4 | 938.775 | 1531.588 |
| cyclical | 459.070 | 3.207 | 339.263 | -42.462 | 2 | 930.499 | 1495.639 |
| calendar | 457.034 | 1.171 | 330.222 | 5.274 | 1 | 942.395 | 1542.302 |
| school_interactions | 457.421 | 1.558 | 327.049 | 31.030 | 2 | 950.788 | 1550.698 |

Cyclical encoding has a negative mean error of S$42.46, compared with S$2.43 for the integer control. The school-interaction variant improves MAE from S$329.98 to S$327.05 but increases RMSE and both the 95th and 99th percentiles of absolute error. Its mean error is positive S$31.03. This is a real metric tradeoff; selection continues to use the existing RMSE objective.

The calendar candidate improved earlier pooled screening RMSE from S$607.05 to S$599.76 (1.20%), but worsened the final ten-month holdout by S$1.17. Earlier query scores use fixed monthly samples, so they are screening estimates rather than complete-fold scores. The result suggests limited stability of these seasonal representations across market periods. It does not establish that school calendars have no rental effect. The recurring month windows add representational hints, rather than new observations, and approval month may differ from signing or move-in month. Primary-school proximity is the existing static dataset snapshot; historical school-opening dates are not modeled.

## Full-context runtime

| model | fit_seconds | predict_seconds | peak_vram_gib | context_rows | subsampling | keep_cache_on_device | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| calendar | 14.935 | 682.316 | 7.153 | 121005 | False | True | TabPFNV3p5 |
| categorical | 15.099 | 711.228 | 7.219 | 121005 | False | True | TabPFNV3p5 |
| cyclical | 13.946 | 682.896 | 7.210 | 121005 | False | True | TabPFNV3p5 |
| school_interactions | 14.961 | 778.461 | 7.126 | 121005 | False | True | TabPFNV3p5 |

GPU memory is peak PyTorch allocation. Metrics and runtime for the integer control are reused after the matched replay, rather than counting another full control run. All new outer runs used 121,005 context rows and 200-row query batches.

Additional files: comparison.csv, monthly_comparison.csv, timings.csv, verification.json, availability_summary.csv, and the prespecified plan at ../../EXPERIMENT_PLAN_TABPFN_CALENDAR.md.
