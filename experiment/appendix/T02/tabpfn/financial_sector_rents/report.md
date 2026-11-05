# Financial-sector stocks and Singapore mean HDB rents

Primary result: the equal-weight DBS/OCBC/UOB basket's prior-month return correlates -0.250 with next-month national arithmetic mean-rent growth (n=49, paired month-block 95% interval -0.532 to 0.066). This is an association, not a causal effect. See the forecasting checks below before using financial-stock features.

Decision: retain the current feature set. No full-control development association survives the dependent-FDR screen. Earlier forecasts selected S68.SI_lag3; later growth RMSE is 1.099, versus 0.951 for persistence. The original bank lag-1 hypothesis also fails to beat the best rent-only control. These later months have already been examined in related experiments.

## Baskets and weighting

Banks: DBS (D05.SI), OCBC (O39.SI), UOB (U11.SI). Broader available financial-services basket: those three banks plus Singapore Exchange (S68.SI) and iFAST (AIY.SI). Nonbank sensitivity basket: SGX/iFAST. Five individual stocks are also reported, without selecting companies on full-period rent correlation. This is the available-file universe, not a comprehensive financial-sector index; insurers and other missing firms are not imputed.

Equal-weight baskets rebalance monthly. The bank cap-seeded sensitivity uses only December 2020 historical STI-fund fair-value weights, initialized at January 2021 month-end prices and allowed to drift; it is not a point-in-time free-float index. That snapshot was published 25 February 2021, before all 2022-onward regression and forecast fits. January-February 2021 index paths are a descriptive backcast. Main returns use adjusted closes; CLOSE-based calculations are a sensitivity check for dividend/adjustment dependence, not an independently reconstructed corporate-action series.

The nonfinancial market control contains 37 January-2021 Singapore-listed inception stocks, explicitly excluding all five financial names. This prevents a bank-heavy broad-market index from being the only sector comparator. Later listings are not added retrospectively.

## Correlations of changes, not just levels

Rent growth is 100*log(arithmetic_mean_rent(t)/arithmetic_mean_rent(t-1)); stock returns use log changes in basket levels. Lag 1 means the preceding calendar month's return; lag 6 means the single-month return six months earlier, not a six-month cumulative return. Lag 0 is descriptive and cannot predict a month's rent before that month's stocks are observed. We also scan three- and twelve-month cumulative changes as descriptive sensitivity checks; overlapping windows have few independent cases.

| signal | stock_lag_months | months | pearson_r | spearman_rho | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| banks_equal | 0 | 50 | 0.014 | 0.042 | -0.203 | 0.267 |
| banks_equal | 1 | 49 | -0.250 | -0.188 | -0.532 | 0.066 |
| banks_equal | 3 | 47 | -0.065 | -0.033 | -0.314 | 0.226 |
| banks_equal | 6 | 44 | -0.161 | -0.161 | -0.508 | 0.239 |
| finance_equal | 0 | 50 | 0.033 | 0.015 | -0.194 | 0.305 |
| finance_equal | 1 | 49 | -0.317 | -0.274 | -0.618 | 0.036 |
| finance_equal | 3 | 47 | -0.133 | -0.107 | -0.369 | 0.142 |
| finance_equal | 6 | 44 | -0.328 | -0.254 | -0.553 | -0.003 |
| nonbank_finance_equal | 0 | 50 | 0.035 | -0.003 | -0.168 | 0.285 |
| nonbank_finance_equal | 1 | 49 | -0.262 | -0.217 | -0.532 | 0.050 |
| nonbank_finance_equal | 3 | 47 | -0.140 | -0.175 | -0.393 | 0.119 |
| nonbank_finance_equal | 6 | 44 | -0.364 | -0.295 | -0.597 | -0.076 |
| banks_cap_seeded | 0 | 50 | 0.002 | 0.026 | -0.216 | 0.256 |
| banks_cap_seeded | 1 | 49 | -0.246 | -0.198 | -0.529 | 0.069 |
| banks_cap_seeded | 3 | 47 | -0.058 | -0.020 | -0.315 | 0.243 |
| banks_cap_seeded | 6 | 44 | -0.176 | -0.172 | -0.526 | 0.232 |

Intervals resample paired contiguous monthly blocks (length max(3, growth window)), not individual rentals. A long-window interval is omitted if fewer than two blocks are available. Full-sample correlations are descriptive; growth_correlations.csv separately reports January 2022-May 2024 development and the already-examined June 2024-March 2025 period. Trending index/mean-rent level correlations are saved separately as a spurious-correlation diagnostic, never used for model selection.

Raw mean rents are paired with an adjusted arithmetic mean using a property model fitted on 2021 only (building, flat type/model, floor area and lease year), preserving the national reference level. This tests whether changing rental composition drives a correlation. Both rent targets and both price fields are fully disclosed.

## Does finance add an association beyond rent history and other markets?

Development-only monthly regressions use Newey-West covariance with three monthly lags. Unadjusted slopes, rent-history/trend/seasonality controls, and full controls are reported. Full controls add nonfinancial Singapore returns, US S&P 500 returns and USD/SGD changes at the same lagged signal month. No same/future-month stock observation enters a predictive regression. No interest-rate, wage, employment or migration causal identification is claimed.

| signal | months | effect_per_10pct_stock | effect_ci_low | effect_ci_high | p_value | q_lag1_finance_tests |
| --- | --- | --- | --- | --- | --- | --- |
| banks_equal | 29 | -0.344 | -1.388 | 0.700 | 0.498 | 1.000 |
| finance_equal | 29 | -0.695 | -2.501 | 1.110 | 0.430 | 1.000 |
| nonbank_finance_equal | 29 | -0.305 | -1.119 | 0.508 | 0.442 | 1.000 |
| banks_cap_seeded | 29 | -0.293 | -1.325 | 0.738 | 0.559 | 1.000 |
| D05.SI | 29 | -0.056 | -0.936 | 0.823 | 0.895 | 1.000 |
| O39.SI | 29 | -0.271 | -1.194 | 0.653 | 0.547 | 1.000 |
| U11.SI | 29 | -0.510 | -1.498 | 0.479 | 0.294 | 1.000 |
| S68.SI | 29 | 0.414 | -0.902 | 1.731 | 0.518 | 1.000 |
| AIY.SI | 29 | -0.275 | -0.704 | 0.155 | 0.196 | 1.000 |

Effects above are rent-growth log points associated with a +10% stock/basket move, not SGD rent increases. Benjamini-Yekutieli q-values account for dependent tests: q_lag1_finance_tests covers the nine lag-1 signals times two rent targets; q_all_lag_company_tests covers all 1/3/6-month lags for each control specification. These are small-sample approximations; the 29 development months provide limited information for a many-control regression.

Full-control development scans surviving a 5% dependent-FDR threshold: 0 out of 54.

## Incremental forecast experiment

One-month-ahead expanding Ridge (alpha 10) starts after twelve mature labels from January 2022. Baselines are zero growth/carry-forward, trailing-three-month momentum, rent-history regression and rent-history plus nonfinancial/US/FX markets. Each financial signal is tested at lag 1/3/6; the three main baskets also have return/momentum/volatility bundles. Earlier January 2023-May 2024 forecasts select a financial candidate and the best control. June 2024-March 2025 is reused confirmation, not an untouched final test. All candidates use the same monthly cases and training-only normalization.

| target | candidate | baseline | earlier_selected_finance | earlier_control_rmse | earlier_candidate_rmse | later_control_rmse | later_candidate_rmse | delta_rmse | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean_rent | banks_equal_lag1 | persistence | False | 1.027 | 1.592 | 0.951 | 1.094 | 0.143 | -0.011 | 0.425 |
| mean_rent | banks_equal_lag1 | rent_market_control | False | 1.569 | 1.592 | 1.107 | 1.094 | -0.012 | -0.057 | 0.011 |
| mean_rent | S68.SI_lag3 | persistence | True | 1.027 | 1.531 | 0.951 | 1.099 | 0.148 | 0.032 | 0.401 |
| mean_rent | S68.SI_lag3 | rent_market_control | True | 1.569 | 1.531 | 1.107 | 1.099 | -0.007 | -0.169 | 0.103 |
| mean_rent | finance_equal_lag1 | persistence | False | 1.027 | 1.571 | 0.951 | 1.139 | 0.188 | 0.011 | 0.453 |
| mean_rent | finance_equal_lag1 | rent_market_control | False | 1.569 | 1.571 | 1.107 | 1.139 | 0.032 | -0.080 | 0.076 |
| mean_rent | banks_cap_seeded_lag1 | persistence | False | 1.027 | 1.597 | 0.951 | 1.096 | 0.145 | -0.009 | 0.428 |
| mean_rent | banks_cap_seeded_lag1 | rent_market_control | False | 1.569 | 1.597 | 1.107 | 1.096 | -0.011 | -0.050 | 0.011 |
| adjusted_mean_rent | banks_equal_lag1 | persistence | False | 0.876 | 1.370 | 0.865 | 1.031 | 0.166 | -0.003 | 0.456 |
| adjusted_mean_rent | banks_equal_lag1 | rent_market_control | False | 1.351 | 1.370 | 1.027 | 1.031 | 0.004 | -0.005 | 0.020 |
| adjusted_mean_rent | S68.SI_lag3 | persistence | True | 0.876 | 1.287 | 0.865 | 1.037 | 0.172 | 0.014 | 0.430 |
| adjusted_mean_rent | S68.SI_lag3 | rent_market_control | True | 1.351 | 1.287 | 1.027 | 1.037 | 0.010 | -0.131 | 0.124 |
| adjusted_mean_rent | finance_equal_lag1 | persistence | False | 0.876 | 1.341 | 0.865 | 1.066 | 0.202 | 0.042 | 0.454 |
| adjusted_mean_rent | finance_equal_lag1 | rent_market_control | False | 1.351 | 1.341 | 1.027 | 1.066 | 0.040 | -0.046 | 0.080 |
| adjusted_mean_rent | banks_cap_seeded_lag1 | persistence | False | 0.876 | 1.376 | 0.865 | 1.030 | 0.165 | -0.009 | 0.459 |
| adjusted_mean_rent | banks_cap_seeded_lag1 | rent_market_control | False | 1.351 | 1.376 | 1.027 | 1.030 | 0.003 | -0.007 | 0.019 |

Forecast errors above are mean-rent growth log points, not individual-rental SGD RMSE. forward_scores.csv additionally reports SGD errors of reconstructed monthly means. Block intervals are conditional on fitted models and omit model/lag selection uncertainty. All candidate errors, including unfavorable ones, are retained.

## Interpretation

A stable finance-return association could proxy expectations about credit, economic conditions, employment or bonuses; a negative or positive empirical sign alone does not identify any of those mechanisms. Bank prices also reflect overseas business, interest margins and discount rates. Stock and rent levels can trend together while monthly changes are unrelated. Adding financial-sector features requires earlier/later forecast gains beyond rent history and nonfinancial-market controls, followed by a matched row-level model experiment and new rental data. No submission or TabPFN model is modified here.

The rental data cover only 51 months spanning reopening and a changing rental trend. Historical company constituents, free floats and data-release vintages are incomplete. The later ten months were already examined in related project work. Statistical intervals and q-values do not create fresh validation after repeated exploration.

## Sources and reproduction

- [banks](https://www.abs.org.sg/about-us/memberbanks)
- [sgx ifast](https://links.sgx.com/1.0.0/corporate-announcements/1JOGK1M7YBA5D0A0/20170609_SGX_welcomes_iFAST_Financial_as_Securities_Trading_Member_and_Clearing_Member.pdf)
- [ifast historical business](https://www.ifastcorp.com/ifastcorp/articleFiles/webarticles/14645/SG/iFAST%20Corp_Annual%20Report%202021_20220401.pdf)

Existing FRED S&P 500 and SGD/USD controls retain their worksite-follow-up source manifest. Historical bank weights retain the December 2020 STI-fund source used by the overall-index experiment.

Run `.venv/Scripts/python.exe analysis/compare_financial_sector_rents.py`. Settings are in analysis/configs/financial_sector_rents.json. Outputs include financial indices/returns, bank weight audits, national raw/adjusted means, correlations, controlled regressions, all forward predictions/scores and charts. results.json records source/code hashes, versions and strict forecast cutoff checks.
