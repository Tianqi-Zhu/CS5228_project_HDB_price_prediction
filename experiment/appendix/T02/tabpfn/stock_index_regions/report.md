# Overall stock index, mean HDB rents and regional differences

This analysis constructs a historical-cap-seeded Singapore stock-return index, checks national mean-rent growth, and tests whether regional stock/rent associations differ. It extends the sector and worksite experiments; existing submissions are unchanged.

Result: same-month stock-return/mean-rent-growth correlation is 0.005. Prior-month correlation is -0.253 (paired month-block 95% interval -0.517 to 0.067). The interval includes zero. Regional slope-equality tests do not establish differing stock responses; neither do they prove equal effects. Earlier stock forecasts beat the best rent-only control in 0 of 12 region/target cases. Retain the current feature set on this evidence.

## Index construction and S&P comparison

An S&P-style float-cap index uses I(t) = sum_i[price_i(t) x shares_i(t) x investable_float_i(t)] / divisor(t), with adjustments for corporate actions and constituent changes. [S&P's methodology](https://www.spglobal.com/spdji/en/methodology/article/index-mathematics-methodology/) describes the calculation. The supplied OHLC/adjusted-price/volume file has neither issuer shares nor free-float history; trading volume is not a replacement for shares outstanding.

The custom **cap_seeded_22** index uses December 2020 [STI-tracking ETF holdings](https://links.sgx.com/1.0.0/corporate-announcements/453E6FXS2LCL8VUR/649560_Nikko_AM-Semi_Annual_Report_31122020.pdf) as a proxy for historical capitalisation weights. The source total fair value sums exactly to S$478,159,013. The 22 matching stocks account for **81.56%** of that portfolio; the eight unmatched stocks are disclosed in index_constituents.csv, rather than replaced with current winners. Initial bank weight is **48.72%**.

Let w(i,0) be each matched holding's fair value divided by the matched total. With P_adj(i,0) the January 2021 month-end adjusted close, calculate fixed notional units q(i)=w(i,0)/P_adj(i,0), then I(t)=100 sum_i q(i)P_adj(i,t). Portfolio value weights drift naturally; returns equal prior-month value weights times constituent returns. Dividends/splits are represented only as the adjusted-close source handles them. This is a fixed-universe, historical-cap-seeded return proxy, **not an exact free-float index, official STI, or S&P 500 replica**. It omits later float/share issuance and index reconstitution/divisor changes.

Controls are monthly-rebalanced **equal_weight_matched** (the same 22 names) and **equal_weight_inception** (42 names priced in January 2021). The later 9CI listing is excluded from the inception basket; membership is not chosen by full-period returns. All indices start at 100 in January 2021. Source holdings were published **25 February 2021**, so the first two months are a retrospective descriptive backcast; all regression/forecast fitting starts in 2022, after publication. No current market caps enter any historical prediction.

## Do stock returns correlate with mean-rent changes?

The main comparison is monthly stock log return through t-1 versus 100*log(mean_rent(t)/mean_rent(t-1)). This uses actual arithmetic mean rents, not mean log rent. Same-month correlations are descriptive only; they are unavailable when forecasting that month's rent. Lag 3/6, three-month and twelve-month cumulative growth are sensitivity scans, never selected by full-period correlation. Overlapping long-window observations have fewer effective independent cases.

| region | target | months | pearson_r | spearman_rho | correlation_ci_low | correlation_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| Singapore | mean_rent | 49 | -0.253 | -0.204 | -0.517 | 0.067 |
| Singapore | adjusted_mean_rent | 49 | -0.256 | -0.177 | -0.501 | 0.062 |
| central region | mean_rent | 49 | -0.197 | -0.196 | -0.471 | 0.123 |
| central region | adjusted_mean_rent | 49 | -0.187 | -0.139 | -0.455 | 0.145 |
| east region | mean_rent | 49 | -0.038 | -0.111 | -0.320 | 0.281 |
| east region | adjusted_mean_rent | 49 | -0.144 | -0.147 | -0.399 | 0.184 |
| north region | mean_rent | 49 | -0.197 | -0.169 | -0.457 | 0.148 |
| north region | adjusted_mean_rent | 49 | -0.136 | -0.134 | -0.392 | 0.203 |
| north-east region | mean_rent | 49 | -0.130 | -0.136 | -0.375 | 0.147 |
| north-east region | adjusted_mean_rent | 49 | -0.137 | -0.128 | -0.380 | 0.139 |
| west region | mean_rent | 49 | -0.312 | -0.290 | -0.584 | -0.012 |
| west region | adjusted_mean_rent | 49 | -0.322 | -0.275 | -0.592 | -0.019 |

This table uses all available labeled months and is descriptive; block intervals resample three adjacent months together. Growth_correlations.csv includes separate 2022-May 2024 development and June 2024-March 2025 descriptive confirmation segments for every index. Level_correlations.csv records the correlations of trending levels solely as a warning: a high level correlation does not establish that stock returns explain rent growth.

## Are regional relationships uniform?

Rental addresses are joined to the supplied HDB block REGION field, giving Central, East, North, North-East and West. Raw means are paired with an adjusted arithmetic mean: a fixed 2021-only model removes building/flat-type/model/area/lease-year composition effects and restores each region's own 2021 reference property level. Both versions are reported because changing rental mix can create apparent regional differences.

Actual mean-rent growth is broadly similar in direction across regions, while levels and monthly fluctuations differ. This descriptive endpoint comparison is distinct from testing whether stock-return coefficients differ:

| region | start_month | end_month | start_mean_sgd | end_mean_sgd | cumulative_change_pct |
| --- | --- | --- | --- | --- | --- |
| Singapore | 2021-01 | 2025-03 | 2048.901 | 3162.708 | 54.361 |
| central region | 2021-01 | 2025-03 | 2193.393 | 3333.864 | 51.996 |
| east region | 2021-01 | 2025-03 | 2056.234 | 3180.697 | 54.686 |
| north region | 2021-01 | 2025-03 | 1929.612 | 2992.112 | 55.063 |
| north-east region | 2021-01 | 2025-03 | 2000.518 | 3090.856 | 54.503 |
| west region | 2021-01 | 2025-03 | 2016.422 | 3142.082 | 55.825 |

Each region's regression relates its monthly rent growth to prior stock return, controlling its own lagged rent growth, trailing-three-month mean growth, a linear time trend and calendar sine/cosine. Coefficients are associations for a +10% stock return, not effects of an intervention. Development regressions use January 2022-May 2024 (29 monthly observations); conditional intervals use Newey-West covariance with three lags. Benjamini-Yekutieli q-values correct dependent national/regional tests.

| region | target | effect_per_10pct_stock | effect_ci_low | effect_ci_high | p_value | q_primary_associations |
| --- | --- | --- | --- | --- | --- | --- |
| Singapore | mean_rent | -0.914 | -2.800 | 0.971 | 0.325 | 1.000 |
| central region | mean_rent | -0.821 | -3.552 | 1.911 | 0.540 | 1.000 |
| east region | mean_rent | 0.492 | -2.504 | 3.487 | 0.737 | 1.000 |
| north region | mean_rent | -0.598 | -2.978 | 1.782 | 0.608 | 1.000 |
| north-east region | mean_rent | -0.460 | -1.926 | 1.006 | 0.522 | 1.000 |
| west region | mean_rent | -1.566 | -3.860 | 0.729 | 0.171 | 1.000 |
| Singapore | adjusted_mean_rent | -1.038 | -2.803 | 0.726 | 0.235 | 1.000 |
| central region | adjusted_mean_rent | -0.887 | -3.016 | 1.242 | 0.397 | 1.000 |
| east region | adjusted_mean_rent | -0.404 | -2.833 | 2.026 | 0.734 | 1.000 |
| north region | adjusted_mean_rent | -0.130 | -2.231 | 1.970 | 0.899 | 1.000 |
| north-east region | adjusted_mean_rent | -0.291 | -1.615 | 1.033 | 0.652 | 1.000 |
| west region | adjusted_mean_rent | -1.679 | -4.157 | 0.800 | 0.174 | 1.000 |

The formal regional equality test estimates all five slopes, then resamples the SAME contiguous month blocks for every region, preserving common shocks and cross-region dependence. It tests the largest standardized pairwise slope contrast against the centered bootstrap null. Pairwise_region_contrasts.csv gives simultaneous intervals across the ten region pairs. We do not count five regional rows as five independent stock observations.

| target | phase | months | equality_p_value | q_all_equality_tests |
| --- | --- | --- | --- | --- |
| mean_rent | development | 29 | 0.477 | 1.000 |
| mean_rent | all_after_reference | 39 | 0.755 | 1.000 |
| adjusted_mean_rent | development | 29 | 0.624 | 1.000 |
| adjusted_mean_rent | all_after_reference | 39 | 0.864 | 1.000 |

A large equality-test p-value means insufficient evidence of different slopes; it does not prove identical effects. The all_after_reference analysis includes the confirmation period and is descriptive sensitivity evidence. Regional_equality_tests.csv also includes the equal-weight alternatives and corrects across all twelve equality checks.

## Forward prediction check

Expanding one-month-ahead Ridge forecasts fit only already observed rent-growth labels, with training-only normalization. Baselines are zero rent growth (carry forward the last mean), trailing-three-month momentum, and rent-history/trend/seasonality regression. Each stock index adds prior return, prior three-month momentum and prior three-month volatility. Fitting starts in January 2022 with at least twelve mature months; earlier January 2023-May 2024 forecasts choose the control and stock index, then June 2024-March 2025 confirms them. The stock candidate remains forced for reporting even when earlier errors favor no stocks.

| region | target | selected_control | selected_stock | development_control_rmse | development_stock_rmse | outer_control_rmse | outer_stock_rmse | delta_rmse | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Singapore | mean_rent | persistence | equal_weight_inception | 1.027 | 1.495 | 0.951 | 1.138 | 0.187 | -0.034 | 0.543 |
| Singapore | adjusted_mean_rent | persistence | equal_weight_inception | 0.876 | 1.284 | 0.865 | 1.073 | 0.208 | -0.027 | 0.600 |
| central region | mean_rent | persistence | equal_weight_inception | 1.384 | 1.912 | 1.867 | 2.216 | 0.349 | -0.069 | 0.839 |
| central region | adjusted_mean_rent | persistence | equal_weight_inception | 1.005 | 1.531 | 1.574 | 1.943 | 0.368 | -0.013 | 0.908 |
| east region | mean_rent | persistence | equal_weight_inception | 2.099 | 2.295 | 1.201 | 1.351 | 0.150 | -0.250 | 0.301 |
| east region | adjusted_mean_rent | persistence | equal_weight_inception | 1.620 | 1.892 | 0.930 | 1.186 | 0.256 | -0.026 | 0.376 |
| north region | mean_rent | persistence | equal_weight_inception | 1.597 | 2.350 | 1.790 | 1.587 | -0.203 | -0.574 | 0.075 |
| north region | adjusted_mean_rent | persistence | cap_seeded_22 | 1.620 | 2.330 | 1.404 | 1.399 | -0.005 | -0.193 | 0.108 |
| north-east region | mean_rent | persistence | equal_weight_inception | 0.842 | 1.205 | 0.947 | 1.174 | 0.228 | 0.040 | 0.485 |
| north-east region | adjusted_mean_rent | momentum3 | equal_weight_inception | 0.762 | 1.131 | 0.933 | 0.942 | 0.009 | -0.149 | 0.254 |
| west region | mean_rent | persistence | cap_seeded_22 | 1.534 | 1.878 | 1.761 | 1.797 | 0.036 | -0.161 | 0.592 |
| west region | adjusted_mean_rent | persistence | cap_seeded_22 | 1.419 | 1.706 | 1.728 | 1.811 | 0.082 | -0.140 | 0.550 |

RMSE above is rent-growth log points, not individual-rental SGD RMSE. Forward_scores.csv additionally reports errors in predicted monthly mean SGD rent. Conditional paired block-bootstrap intervals omit selection/fitting uncertainty and are not adjusted for all model comparisons. This monthly-updated exercise assumes preceding-month rents are published; it is not a fixed-origin forecast of the entire Kaggle test period.

## Practical limits and feature decision

Use a stock-index feature only if it improves earlier and later forecasts against rent-history controls, with a stable interpretation across index definitions and regions. A correlation of levels, one favorable lag, or one favorable region is insufficient. These observational relationships cannot establish stock-market causation; employment, migration, interest rates, housing supply and reopening can drive both markets. New rent data and a matched TabPFN ablation are needed before changing the final feature ensemble.

There are only 51 labeled months with serial dependence; the confirmation period has been examined elsewhere in this repository. Historical-cap coverage is incomplete, initial weights can become stale, companies can merge, and the price file is a later adjusted-history snapshot. A fixed 2021 mix model misses changing property premiums. Neither commercial employment exposure nor regional causal treatment is measured. Exact reproduction of a float-cap index requires point-in-time issuer/free-float shares, constituent changes and corporate-action data beyond this file.

## Reproduction

Run `.venv/Scripts/python.exe analysis/compare_stock_index_regions.py`. Inputs are read-only. The seed table is analysis/configs/sti_etf_weights_2020_12.csv; its source, publication date and limitations are in stock_index_methodology.json. Outputs include index levels/returns, monthly constituent weights, all regional rent panels, correlations, slope/equality tests, forward predictions, earlier selections and charts. results.json stores source/code hashes, versions, settings and cutoff checks.
