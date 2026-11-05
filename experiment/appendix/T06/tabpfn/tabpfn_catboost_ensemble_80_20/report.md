# TabPFN and CatBoost validation ensembles

**Primary ensemble validation RMSE: S$456.01; MSE: 207,942.32 SGD squared.** The blend selected on earlier forecasts is slightly worse than the retained TabPFN alone (S$455.86 RMSE), so retain TabPFN.

The chronological 80/20 split retains 121,005 context rows (January 2021-May 2024) and 28,995 validation rows (June 2024-March 2025). TabPFN predictions are hash-checked and reused from the best retained aligned residual setup and its effectively tied categorical-month variant.

CatBoost uses 600 iterations, depth 7, learning rate 0.05, L2 regularization 5, seed 42, and four CPU threads. cb68 is the existing raw-rent control with 68 features and preceding-month context histories. cb77raw uses the 77 aligned inputs to predict raw rent. cb77residual uses those same inputs to predict the same residual target as TabPFN and adds the baseline back exactly once. No query labels enter fitting, an eval_set, or early stopping.

Blend weights use earlier ten-month forecasts (September 2022-June 2023 and July 2023-April 2024). Each uses every earlier context row and the same fixed 200-query-per-month samples as the calendar experiment, weighted by month population/sample count. Six model pairs each have a fixed 50/50 blend and a constrained weighted-least-squares blend. The latter has no intercept and a TabPFN weight in [0,1]. Outer labels never fit a weight or select a pair.

Earlier-selected ensemble: `integer_cb68_fitted`. Primary reported ensemble: `integer_cb68_fitted`. Its prediction is **0.663028 x TabPFN + 0.336972 x CatBoost**.

## Primary validation loss

| model | n | rmse_sgd | mae_sgd | bias_sgd | mse_sgd2 |
| --- | --- | --- | --- | --- | --- |
| integer | 28995 | 455.863 | 329.984 | -2.428 | 207810.981 |
| cb68 | 28995 | 460.782 | 338.909 | -28.262 | 212320.450 |
| integer_cb68_fitted | 28995 | 456.007 | 331.604 | -11.133 | 207942.325 |
| integer_cb68_equal | 28995 | 456.632 | 332.911 | -15.345 | 208512.354 |

Outer confirmation passed: `False`. RMSE is in SGD; MSE is in SGD squared. The primary ensemble and weights were frozen before new outer scoring. If it fails confirmation, retain the original TabPFN rather than select another pair from outer scores.

## All validation comparisons

| model | n | rmse_sgd | mae_sgd | bias_sgd | mse_sgd2 |
| --- | --- | --- | --- | --- | --- |
| integer | 28995 | 455.863 | 329.984 | -2.428 | 207810.981 |
| categorical | 28995 | 455.814 | 329.748 | 0.768 | 207766.011 |
| cb68 | 28995 | 460.782 | 338.909 | -28.262 | 212320.450 |
| cb77raw | 28995 | 459.773 | 333.602 | 4.954 | 211391.531 |
| cb77residual | 28995 | 460.897 | 333.572 | 10.792 | 212425.861 |
| integer_cb68_equal | 28995 | 456.632 | 332.911 | -15.345 | 208512.354 |
| integer_cb77raw_equal | 28995 | 456.428 | 330.316 | 1.263 | 208326.433 |
| integer_cb77residual_equal | 28995 | 456.651 | 329.977 | 4.182 | 208530.574 |
| categorical_cb68_equal | 28995 | 456.526 | 332.798 | -13.747 | 208416.340 |
| categorical_cb77raw_equal | 28995 | 456.370 | 330.224 | 2.861 | 208273.396 |
| categorical_cb77residual_equal | 28995 | 456.805 | 330.051 | 5.780 | 208670.705 |
| integer_cb68_fitted | 28995 | 456.007 | 331.604 | -11.133 | 207942.325 |
| integer_cb77raw_fitted | 28995 | 456.116 | 330.038 | 0.589 | 208042.001 |
| integer_cb77residual_fitted | 28995 | 455.876 | 329.515 | 1.276 | 207823.007 |
| categorical_cb68_fitted | 28995 | 455.770 | 331.039 | -7.307 | 207726.229 |
| categorical_cb77raw_fitted | 28995 | 455.850 | 329.726 | 2.128 | 207798.873 |
| categorical_cb77residual_fitted | 28995 | 455.762 | 329.533 | 1.810 | 207719.406 |

## Earlier calibration scores

| model | kind | rmse_sgd | mae_sgd | bias_sgd | mse_sgd2 | consistent | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| integer | standalone | 607.049 | 471.328 | 108.223 | 368508.520 |  |  |
| categorical | standalone | 602.587 | 470.251 | 77.467 | 363110.779 |  |  |
| cb68 | standalone | 637.597 | 504.181 | -140.088 | 406529.424 |  |  |
| cb77raw | standalone | 615.147 | 485.159 | -31.914 | 378406.324 |  |  |
| cb77residual | standalone | 610.994 | 475.107 | 99.821 | 373313.655 |  |  |
| integer_cb68_equal | ensemble | 598.638 | 471.844 | -15.933 | 358366.907 | True | True |
| integer_cb77raw_equal | ensemble | 599.923 | 469.819 | 38.155 | 359908.110 | True | True |
| integer_cb77residual_equal | ensemble | 606.777 | 471.179 | 104.022 | 368178.855 | False | False |
| categorical_cb68_equal | ensemble | 600.296 | 474.207 | -31.311 | 360355.734 | False | False |
| categorical_cb77raw_equal | ensemble | 599.860 | 470.974 | 22.777 | 359831.956 | True | True |
| categorical_cb77residual_equal | ensemble | 604.145 | 470.440 | 88.644 | 364991.293 | True | True |
| integer_cb68_fitted | ensemble | 596.043 | 467.611 | 24.549 | 355267.664 | True | True |
| integer_cb77raw_fitted | ensemble | 599.547 | 468.903 | 50.951 | 359456.212 | True | True |
| integer_cb77residual_fitted | ensemble | 606.342 | 470.714 | 105.869 | 367650.686 | True | False |
| categorical_cb68_fitted | ensemble | 596.271 | 468.773 | 16.953 | 355539.600 | True | True |
| categorical_cb77raw_fitted | ensemble | 598.743 | 469.216 | 41.916 | 358493.743 | True | True |
| categorical_cb77residual_fitted | ensemble | 602.471 | 469.910 | 79.793 | 362971.323 | True | True |

## Error diversity

| tabpfn | catboost | error_correlation | prediction_disagreement_rmse |
| --- | --- | --- | --- |
| integer | cb68 | 0.987 | 78.825 |
| integer | cb77raw | 0.988 | 71.409 |
| integer | cb77residual | 0.985 | 79.696 |
| categorical | cb68 | 0.987 | 80.669 |
| categorical | cb77raw | 0.988 | 72.260 |
| categorical | cb77residual | 0.987 | 75.504 |

Earlier query evaluation is sampled; final validation and all context sets are complete. The outer period has been examined before and is a development holdout, not an untouched test. Its ten-month horizon does not establish seventeen-month test performance.

Weights are in weights.csv and weight_selection.json; row predictions, monthly losses, CatBoost timings and model artifacts are also saved. Binary models remain under ignored .cache/.

Reproduce/resume: `.venv/Scripts/python.exe analysis/compare_tabpfn_catboost_ensemble.py`.

## Interpretation and verification

The earlier-selected blend improves 3 of ten validation months, but its pooled RMSE is S$0.14 worse than integer-month TabPFN. The fixed 50/50 blend is S$0.77 worse. The selected weight is 66.3028% TabPFN and 33.6972% original-feature CatBoost; it was estimated before new outer scoring. CatBoost underpredicts by S$28.26 on this outer period, compared with S$2.43 for TabPFN. The blend underpredicts by S$11.13. Earlier-period calibration therefore does not guarantee a gain in the final period.

The lowest observed outer blend is `categorical_cb77residual_fitted`, with RMSE S$455.76244 and MSE 207719.40601. Its weights are 0.895961 categorical-month TabPFN and 0.104039 aligned-residual CatBoost. It improves on its TabPFN component (S$455.81357) by approximately S$0.05, and is an exploratory comparison rather than a replacement selected before outer scoring. Errors are strongly correlated across all pairs (0.985-0.988), limiting diversification.

Seven new tests passed, including a regression test that checks fresh and resumed residual predictions are bit-identical after CSV persistence and that the baseline is added once. A cached rerun of the full experiment succeeded without any CatBoost refit. All 28,995 prediction IDs and labels were checked, all losses independently recomputed, all fitted weights recomputed solely from earlier queries, and every ensemble checked against its convex formula and component prediction range. Eight saved CatBoost model hashes were also verified. See verification.json and resume_precision_fix.json.
