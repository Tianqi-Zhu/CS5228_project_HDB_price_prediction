# Monthly flat-type rent forecasts with COE category A

A pooled Ridge model maps the rent and COE histories available at an origin to a future monthly mean rent for each flat type. It predicts a change from the origin's recent three-month mean. Short history uses three months; long history uses twelve. Horizon and calendar seasonality are explicit. Regularized type interactions allow different responses while sharing statistical strength. The intercept and all adjustment coefficients are penalized, so strong regularization approaches the recent-mean baseline instead of retaining unconditional historical growth.

COE features are the twelve-month mean log premium, three-month minus twelve-month mean log premium, three-month log change, and a damped horizon interaction. No future COE observations or validation rent labels are inputs. Category-A auction premiums are aggregated by monthly median.

| validation_start | model | group_months | weighted_mean_rmse_sgd | mean_rmse_sgd | weighted_bias_sgd |
| --- | --- | --- | --- | --- | --- |
| 2024-06 | last_month | 57 | 56.133 | 148.510 | -40.198 |
| 2024-06 | recent3 | 57 | 54.645 | 153.424 | -44.600 |
| 2024-06 | damped_trend | 57 | 43.732 | 183.717 | -20.567 |
| 2024-06 | ridge_history | 57 | 42.987 | 136.358 | 21.076 |
| 2024-06 | ridge_history_coe | 57 | 80.842 | 127.235 | 61.593 |
| 2023-11 | last_month | 96 | 73.597 | 193.425 | -55.150 |
| 2023-11 | recent3 | 96 | 68.281 | 193.619 | -53.428 |
| 2023-11 | damped_trend | 96 | 79.630 | 318.089 | -58.087 |
| 2023-11 | ridge_history | 96 | 227.206 | 345.201 | 216.030 |
| 2023-11 | ridge_history_coe | 96 | 402.729 | 442.151 | 363.077 |

Weighted mean RMSE weights monthly flat-type errors by transaction counts. These scores measure group-mean forecasting, not individual transaction accuracy, and cannot be compared directly with TabPFN's transaction RMSE. The two outer holdouts overlap, so they are reported separately.

The target is the raw monthly arithmetic mean; changing property composition can move it. Rare flat types receive little weight and have uncertain estimates. There are only 41 training months in the 80/20 split and 34 in the 17-month split. Economic associations do not establish causality. This prototype has not yet been added to TabPFN context features.

These are development backtests. Initial results with an unpenalized intercept are retained in `initial_unpenalized_intercept/`; they motivated penalizing the entire adjustment toward a recent-mean baseline. Adding COE does not improve the revised model in either holdout, and neither learned specification beats a recent-mean baseline on the 17-month holdout. The COE forecast feature is not promoted to the TabPFN experiment.

Reproduce: `.venv/Scripts/python.exe analysis/compare_flat_type_economic_forecast.py`.
