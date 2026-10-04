# E baseline versus engineered economic features

Outcome: individual monthly HDB rent (SGD). Lower RMSE is better; a negative delta improves on E in the same fold.

| Validation start | Variant | RMSE (SGD) | Delta vs E (SGD) |
| --- | --- | ---: | ---: |
| 2022-07 | E_amenities | 658.14 | +0.00 |
| 2022-07 | E_plus_coe | 658.64 | +0.50 |
| 2022-07 | E_plus_sg_stocks | 660.93 | +2.79 |
| 2022-07 | E_plus_both | 661.98 | +3.84 |
| 2023-01 | E_amenities | 642.74 | +0.00 |
| 2023-01 | E_plus_coe | 642.94 | +0.20 |
| 2023-01 | E_plus_sg_stocks | 645.32 | +2.59 |
| 2023-01 | E_plus_both | 645.39 | +2.66 |
| 2023-07 | E_amenities | 567.69 | +0.00 |
| 2023-07 | E_plus_coe | 567.45 | -0.24 |
| 2023-07 | E_plus_sg_stocks | 567.51 | -0.18 |
| 2023-07 | E_plus_both | 567.99 | +0.31 |
| 2024-04 | E_amenities | 478.82 | +0.00 |
| 2024-04 | E_plus_coe | 479.63 | +0.81 |
| 2024-04 | E_plus_sg_stocks | 479.54 | +0.72 |
| 2024-04 | E_plus_both | 479.92 | +1.10 |

## Design

E uses the original 29 property, time, block and amenity features. COE adds prior-month A/C premiums, their month-on-month growth, and bids divided by quota. Stocks use only .SI symbols: the median one-month adjusted-close return, compounded three-month return, and rolling three-month volatility of that basket return. E + both uses all nine signals.

The same HistGradientBoosting model, feature preparation, rental rows and four expanding-window splits are used for every variant. The market observations for approval month t end at t-1. This is a monthly-updated scenario, not a fixed-origin forecast for the entire future test period.

## Interpretation

Improved folds (out of four): E_plus_both 0/4, E_plus_coe 1/4, E_plus_sg_stocks 1/4.
E + both has lower monthly RMSE in 4 of 33 validation months; it is worse in all four complete forward windows.
The E scores reproduce the earlier forward E baseline within 0.00004 SGD RMSE over 4 matching folds.
This is a fixed-model comparison, not proof that economic signals lack predictive value. There are few independent market months, the last holdout has been examined previously, and the supplied stock-basket membership is a snapshot. Reported row-level errors do not establish causal effects.

Run python analysis/compare_economic_features.py in the 5228_hw environment. See forward_scores.csv for RMSE, MAE and bias, monthly_errors.csv for temporal drift, feature_coverage.csv for missingness, and results.json for exact feature sets and source hashes.
