# Spatial-temporal CatBoost correction of TabPFN residuals

Earlier chronological validation selects **E2**. On the existing 80/20 development holdout, this frozen choice scores **S$456.490 RMSE**, a change of **S$+1.419** from the unchanged MSE-only DiagScale baseline.

<!-- result-summary-start -->
## Observed conclusions

| model | validation_rmse_sgd | strength | development_holdout_rmse_sgd | development_holdout_mae_sgd | delta_rmse_sgd |
| --- | --- | --- | --- | --- | --- |
| B0 | 525.984 | 0.000 | 455.071 | 330.790 | 0.000 |
| E1 | 525.342 | 0.500 | 456.438 | 335.926 | 1.367 |
| E2 | 525.249 | 0.500 | 456.490 | 336.038 | 1.419 |

1. Residual predictability is weak: E2 unshrunk validation zero-baseline R2 is -0.004157; lambda=0.5 produces skill 0.002795 (about 0.280% reduction in squared error). Both early-stopping probes select only four trees, rather than using the800-tree cap. This is the selected conservative result, not an execution failure.

2. Basic correction does not improve the later development period: E1 increases RMSE by S$1.367 and MAE by S$5.136.

3. Spatial-temporal features improve validation RMSE over E1 by only S$0.093; E2 is S$0.052 worse than E1 on the later holdout. Their final-model importance sum is 0.000%, with 0 selected splits on these three features (selected_tree_splits.csv). The small validation difference therefore provides no direct evidence that the model learned useful historical-market effects; adding candidate columns can alter tree fitting even when those columns receive no final splits.

4. The earlier-selected E2 does not confirm a gain: RMSE changes by S$+1.419. E2 month-cluster descriptive95% interval is [+0.852, +2.010] SGD. Its underprediction bias increases from S$-8.428 to S$-35.291; the learned downward adjustment transfers poorly to the later period.

5. Retain B0. The tested correction adds complexity without a supported performance benefit. Generalization to the unlabelled competition test cannot be assessed; neither models nor existing submission are replaced.

Validation: 15 tests passed. Independent saved-model replay reproduces every E1/E2 holdout residual and corrected prediction bit for bit, and lambda0 reproduces B0 exactly. Fold adapter/feature/prediction hashes and all protected source hashes are verified. See summary_verification.json.

To regenerate this concise conclusion after the main runner, execute `.venv/Scripts/python.exe analysis/summarize_tabpfn_spatial_temporal_residual.py`.
<!-- result-summary-end -->

## Scope and baseline ambiguity

The numerical-best existing declared TabPFN result is MSE-only DiagScale (S$455.071). Its adapter was trained before December 2023 and its epoch selected on 600 weighted December 2023-May 2024 queries. The repository also retains older frozen/integer controls in its README; no single global earlier-only rule selected a winner across every later architectural family. This study explicitly targets the existing numerical-best adapted model, fixed before fitting residual corrections. It does not infer a new universal winner from its outer scores.

The primary 77-feature bundle, TabPFN-3.5 Fast pretrained checkpoint, original selected DiagScale adapter, one deployment view, seed42, native preprocessing and residual reconstruction are preserved. All 121,005 January 2021-May 2024 context rows and 28,995 June 2024-March 2025 development queries are identical to the reference. Baseline predictions are reused and independently rescored; no parameter or existing prediction file is modified.

**There is no labelled competition test set.** The 50,000 competition rows remain unpredicted by this correction experiment, and their RMSE/MAE cannot be computed. The original 80/20 holdout has been repeatedly inspected in previous work. Its role here is a frozen development-period confirmation, not an untouched final-test claim.

## Leakage-safe residual generation

| origin | end_exclusive | context_rows | query_rows | context_max_month |
| --- | --- | --- | --- | --- |
| 2022-09 | 2023-01 | 60079 | 10104 | 2022-08 |
| 2023-01 | 2023-05 | 70183 | 11741 | 2022-12 |
| 2023-05 | 2023-09 | 81924 | 12194 | 2023-04 |
| 2023-09 | 2023-12 | 94118 | 8711 | 2023-08 |
| 2023-12 | 2024-06 | 102829 | 18176 | 2023-11 |

Each fold starts from the original pretrained backbone and fits a new adapter using only pre-origin rows. No existing adapter trained on later rental labels is used for historical residuals. The previously selected MSE-only recipe is fixed to one epoch (epoch0 in the original trace), two training views, lr0.01, weight decay0.01, 1024-row tasks and 20% task queries. OOF query labels never enter gradient fitting, early stopping, or checkpoint selection. All remaining backbone parameters are hash-checked against the original frozen tensor digest. Original historical context inputs are reused; every target-derived query history and temporal forecast is rebuilt at that fold origin using existing utilities. Query windows are complete and disjoint; no sampling or duplicate residual observations is used.

Residuals are actual SGD rent minus out-of-sample TabPFN SGD rent. Target reconstruction adds the query temporal baseline exactly once. The fixed historical recipe was developed in prior project work and is not itself a newly prospective hyperparameter choice. Fold-specific one-epoch recipes preserve the algorithm but are not byte-identical copies of the final adapter selected by its original calibration. The final baseline retains its original pre-December-2023 learned adapter and all pre-June-2024 inference context.

## Residual diagnostics before final evaluation

Diagnostics use only out-of-sample residuals before June 2024. residual_diagnostics_month.csv and residual_diagnostics_town.csv contain counts, mean residuals, spread, quantiles and prediction errors. residual_distribution.json reports pooled distribution and autocorrelation of monthly mean residuals (not arbitrarily ordered transaction rows). See residual_diagnostics.png. Spatial and temporal correlations are descriptive, not causal.

## Residual features and CatBoost selection

E1 (10 columns): TABPFN_RENT, FLAT_TYPE_CLEAN, FLOOR_AREA_SQM, FLAT_AGE_YEARS, LATITUDE, LONGITUDE, TOWN_CLEAN, APPROVAL_MONTH_INDEX, APPROVAL_MONTH_OF_YEAR, TEMP_FORECAST_HORIZON.

E2 (13 columns): E1 plus LOCAL_HIST_MEDIAN, LOCAL_GROWTH_3M, LOCAL_COUNT_6M.

E1 includes forecast horizon because simultaneous future batches have stale fixed-origin information. Calendar year is represented by the existing continuous month index; it is not duplicated as a second equivalent year field. Town and flat type are native CatBoost categories. Numeric missing values retain native NaN; no label-informed imputation or scaling is fitted. No new amenity datasets or proximity features are introduced.

E2 median uses all strictly pre-origin town x flat-type rents with at least20 observations, otherwise flat-type then global historical median. Growth compares medians from the latest three observed months and the preceding three months, with at least10 observations in each window at the same grouping level; flat-type/global fallbacks have the same paired-window requirement. Insufficient growth is missing. The six-month transaction count always refers to the exact town x flat-type group, even when the median falls back. All future queries in one batch share the same origin; no intermediate query labels update histories. These are recent historical growth signals, not realised future market growth.

CatBoost uses CPU, depth5, lr0.04, L2=5, seed42 and an800-tree cap. September2022-August2023 residuals fit the early-stopping probe; September-November2023 selects its tree count (patience75). Each variant then refits on all September2022-November2023 OOF residuals. December2023-May2024 selects lambda from {0,.25,.5,.75,1} and the preferred variant, with ties favouring simpler models. After choices are frozen, final CatBoost models refit on all pre-June2024 OOF residuals, retaining those tree counts and strengths. No outer targets enter an eval_set or selection. [CatBoost fit semantics](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit).

| model | strength | rmse_sgd | mae_sgd | bias_sgd | r2 | residual_zero_baseline_r2 | residual_ordinary_r2 | selected_correction_zero_baseline_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.000 | 525.984 | 396.799 | 24.738 | 0.412 | 0.000 | nan | 0.000 |
| E1 | 0.500 | 525.342 | 402.077 | -11.502 | 0.413 | -0.005 | -0.007 | 0.002 |
| E2 | 0.500 | 525.249 | 402.010 | -11.822 | 0.413 | -0.004 | -0.006 | 0.003 |

Residual zero-baseline R2 = 1-SSE(residual prediction)/sum(actual residual squared), with the unshrunk residual prediction. Ordinary residual R2 uses residual deviations from their validation mean. The selected-correction skill applies the selected lambda and can be zero even if the raw residual model has negative skill. Full lambda curves are in validation_strength_curves.csv; per-variant tree counts, model hashes, strengths and row counts are in E1_selection.json/E2_selection.json.

## Frozen 80/20 development confirmation

| model | n | strength | rmse_sgd | mae_sgd | bias_sgd | r2 | delta_rmse_sgd | rmse_improvement_pct | mae_improvement_pct | validation_rmse_sgd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 28995 | 0.000 | 455.071 | 330.790 | -8.428 | 0.517 | 0.000 | 0.000 | 0.000 | 525.984 |
| E1 | 28995 | 0.500 | 456.438 | 335.926 | -34.773 | 0.514 | 1.367 | -0.300 | -1.553 | 525.342 |
| E2 | 28995 | 0.500 | 456.490 | 336.038 | -35.291 | 0.514 | 1.419 | -0.312 | -1.587 | 525.249 |

The table uses development-holdout RMSE/MAE rather than inventing unavailable competition-test scores. Improvements are relative to B0. Both E1/E2 are scored with their earlier-selected strength; the primary model remains the variant selected before outer scoring. Zero strength returns an exact copy of B0.

Paired uncertainty intervals below are candidate-minus-B0 RMSE, with5000 month-cluster and5000 building-cluster resamples, seed42. They describe this development period and do not adjust for repeated project-wide holdout reuse.

| model | cluster | lower_95 | upper_95 | clusters |
| --- | --- | --- | --- | --- |
| E1 | month | 0.816 | 1.967 | 10 |
| E1 | building | 1.047 | 1.693 | 8210 |
| E2 | month | 0.852 | 2.010 | 10 |
| E2 | building | 1.091 | 1.755 | 8210 |

## Interpretation and limitations

Positive validation skill is evidence of predictability at that earlier origin; it does not guarantee the same residual bias or conditional error pattern in the next market regime. E1 versus E2 tests the incremental utility of three historical market features under fixed conservative CatBoost settings. Comparing them is predictive, not causal attribution. The expanding residual windows have three/four-month horizons, residual validation has six, and the outer period has ten; long-horizon extrapolation and changing context size remain limitations. Town-level histories can already be encoded indirectly by TabPFN, leaving little incremental signal. Sparse groups and changing rental-market bias can make an apparently learnable correction unreliable.

No existing best model or submission is replaced. A positive outer development result would still require another forecast origin or genuinely future labelled data before claiming test generalization. Verification covers checkpoint/source preservation, OOF chronology, absence of outer training IDs, feature schema, SGD units, prediction alignment, complete coverage and exact lambda-zero replay.

## Artifacts and reproduction

settings.json and feature_schema.json define the experiment; fold_boundaries.csv and oof_provenance_*.json document contexts/adapters. Binary fold adapters, query feature bundles and E1/E2 CatBoost models are under .cache/tabpfn_spatial_temporal_residual_80_20/. Row predictions, diagnostics, curves, importances, figures, metrics and verification.json are in this report directory.

Run `.venv/Scripts/python.exe -m unittest tests.test_spatial_temporal_residual tests.test_aligned_temporal_features`, then `.venv/Scripts/python.exe analysis/compare_tabpfn_spatial_temporal_residual.py --stage all`. Completed fold predictions and CatBoost models are hash-checked and reused. The baseline checkpoint, adapter, sources, original report and submission hashes must remain unchanged.
