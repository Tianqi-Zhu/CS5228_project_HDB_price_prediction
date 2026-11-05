# Singapore HDB rents and industry stock signals

This is a predictive experiment, not an estimate of a causal stock-market effect. Source data and submission files were preserved.

Data: 150,000 rental rows but only 51 independent calendar months (January 2021-March 2025); 43 supplied Singapore-listed stocks. All stock observations after March 2025 are excluded.

## Decision and selected candidates

Earlier data selected rent-only **persistence**, best stock candidate **market_plus_developers**, and best industry candidate **market_plus_developers**. These choices were frozen before evaluating June 2024-March 2025. Selection averages persistence-normalized development MSE across horizons 1, 3 and 6 months; Ridge alpha is fixed at 10.

| scenario | variant | rmse_sgd | delta_vs_selected_rent_baseline | delta_vs_matched_rent_ar | improved_months | delta_rmse_ci_low | delta_rmse_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| monthly_updated_h1 | persistence | 26.02 | 0.00 | -4.34 | 0 | 0.00 | 0.00 |
| monthly_updated_h1 | market_plus_developers | 34.24 | 8.22 | 3.88 | 4 | -0.34 | 24.24 |
| monthly_updated_h1 | broad_sg_lag1 | 30.34 | 4.32 | -0.02 | 5 | -0.94 | 15.58 |
| monthly_updated_h1 | broad_median_lag1 | 30.52 | 4.50 | 0.16 | 5 | 0.00 | 15.48 |
| rolling_origin_h3 | persistence | 31.01 | 0.00 | -16.01 | 0 | 0.00 | 0.00 |
| rolling_origin_h3 | market_plus_developers | 50.60 | 19.59 | 3.58 | 3 | -7.62 | 51.64 |
| rolling_origin_h3 | broad_sg_lag1 | 47.37 | 16.36 | 0.35 | 5 | -11.25 | 50.13 |
| rolling_origin_h3 | broad_median_lag1 | 50.14 | 19.13 | 3.12 | 5 | -10.40 | 54.86 |
| rolling_origin_h6 | persistence | 48.92 | 0.00 | -87.04 | 0 | 0.00 | 0.00 |
| rolling_origin_h6 | market_plus_developers | 105.64 | 56.73 | -30.32 | 2 | 31.09 | 91.95 |
| rolling_origin_h6 | broad_sg_lag1 | 145.24 | 96.32 | 9.28 | 1 | 68.36 | 121.63 |
| rolling_origin_h6 | broad_median_lag1 | 134.39 | 85.47 | -1.58 | 0 | 57.60 | 120.76 |
| fixed_origin_may2024 | persistence | 49.95 | 0.00 | -99.75 | 0 | 0.00 | 0.00 |
| fixed_origin_may2024 | market_plus_developers | 139.81 | 89.85 | -9.90 | 3 | 10.66 | 134.48 |
| fixed_origin_may2024 | broad_sg_lag1 | 140.06 | 90.11 | -9.65 | 3 | 5.65 | 133.71 |
| fixed_origin_may2024 | broad_median_lag1 | 150.57 | 100.61 | 0.86 | 3 | 11.01 | 147.01 |

Negative deltas improve rent-only forecasts. These are RMSEs of the aggregate property-adjusted monthly index, not individual property RMSEs. Intervals use paired three-month moving-block bootstrap; ten outer months and one fixed-origin episode cannot establish a robust effect.

## Industry hypotheses

| basket | stocks |
| --- | --- |
| banks | D05.SI, O39.SI, U11.SI |
| reits_industrial | A17U.SI, M44U.SI, ME8U.SI |
| reits_commercial | C38U.SI, K71U.SI, N2IU.SI, T82U.SI |
| reits_all | A17U.SI, M44U.SI, ME8U.SI, C38U.SI, K71U.SI, N2IU.SI, T82U.SI, BUOU.SI, HMN.SI |
| developers | C09.SI, U06.SI, U14.SI, Z25.SI |
| travel_hospitality | C6L.SI, S58.SI, S59.SI, G13.SI, H15.SI, HMN.SI |
| engineering | S63.SI, V03.SI, 5E2.SI |
| telecom_infrastructure | Z74.SI, CJLU.SI |

These are manually specified business-exposure baskets, not an exhaustive official sector taxonomy. Banks proxy credit and business conditions; REITs proxy property/financing conditions; developers proxy development sentiment; travel/hospitality proxy reopening and visitor activity; engineering proxies industrial demand; telecom infrastructure is a comparison. These mechanisms are hypotheses, not findings. Baskets overlap, notably REITs and hospitality. Industrial/commercial REIT subsets are excluded from the joint bundle to reduce duplication. Singapore listing does not imply Singapore-only revenue or HDB residential exposure. The broad basket is the supplied snapshot, not STI.

Banks and REIT groupings were cross-checked against primary association sources. Current classifications do not reconstruct historical constituent changes. See sources below and sector_membership.csv for names supplied by the dataset.

## Forecast design

1. Fit a property adjustment on 2021 rentals only (town, flat type/model, area, lease commencement year); aggregate rent minus fitted property effect plus the 2021 mean. The model has no approval-date input. All scored forecast origins are after the reference period. This is a fixed-mix approximation, not an official HDB rental index.
2. Baselines are persistence, extrapolated three-month momentum, and regularized rent autoregression. Rent regression inputs are latest change, trailing three/six-month mean changes, and known target-month sine/cosine. Models directly predict cumulative rent change at each horizon.
3. Stock features are equal-weight monthly adjusted-close returns, compounded three-month momentum, and three-month return volatility. One basket's signals are tested at delays 0, 2, and 5 months: lag 1, 3, and 6 relative to next-month rent. Broad median-return controls match the earlier basket approach. Relative candidates add broad signals and sector-minus-broad signals; subtracting volatility is a contrast, not excess-return volatility.
4. Basket returns require 80% member coverage. Prices never backfill or carry across missing months. Training-only median imputation, standardization and Ridge coefficients are refitted at each origin. Minimum training size is eight mature monthly targets; early long-horizon models are therefore fragile, and development windows differ across horizons.
5. A training example at origin j and horizon h is eligible only if j+h is no later than the current origin. Market and rent information stop at that origin. Development scoring starts July 2022 and ends May 2024. The outer target months are June 2024-March 2025. Rolling h=3/6 forecasts do not know intervening rent or stock observations. The fixed-origin path uses May 2024 information for all ten horizons.

## Earlier ranking (exploratory)

| variant | normalized_mse | horizons |
| --- | --- | --- |
| persistence | 1.00 | 3 |
| momentum_3m | 1.27 | 3 |
| market_plus_developers | 4.50 | 3 |
| all_sectors | 4.82 | 3 |
| developers_lag3 | 4.82 | 3 |
| developers_lag1 | 5.05 | 3 |
| market_plus_travel_hospitality | 6.02 | 3 |
| telecom_infrastructure_lag3 | 6.12 | 3 |
| reits_commercial_lag3 | 6.21 | 3 |
| banks_lag3 | 6.31 | 3 |
| broad_median_lag3 | 6.33 | 3 |
| telecom_infrastructure_lag1 | 6.37 | 3 |

The rank explores many baskets/lags with very few months. It is not a significance test. growth_correlations.csv reports development-only raw Pearson/Spearman associations and early/late sign stability; it does not select stocks by full-sample correlation. Every candidate's forward scores are saved, including unsuccessful ones.

## Individual-rental ablation

A controlled HistGradientBoosting model uses the existing 29 property/time/block/amenity features, all training rows, and the existing fixed 200-iteration settings. It adds each sector's three immediate stock features, the broad basket, or the joint bundle. Sector selection is independent of monthly-index selection: two disjoint earlier windows (September 2022-February 2023 and July-December 2023) choose the industry using row-count-weighted MSE; June 2024-March 2025 then confirms it. The monthly-updated and fixed-origin scenarios select independently on their own earlier scores.

Monthly-updated inputs use stocks through the prior month. Fixed-origin validation carries the last known basket features from before the fold across the forecast period; it never substitutes realized future stocks. No stock forecasting is attempted. The rental training model is frozen for both scenarios.

| scenario | variant | development_rmse | outer_rmse | delta_rmse | improved_months | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| monthly_updated | no_stocks | 631.83 | 471.63 | 0.00 | 0 | 0.00 | 0.00 |
| monthly_updated | broad_sg | 632.20 | 472.23 | 0.60 | 3 | 0.16 | 1.10 |
| monthly_updated | engineering | 631.67 | 472.17 | 0.54 | 2 | 0.30 | 0.93 |
| fixed_origin | no_stocks | 631.83 | 471.63 | 0.00 | 0 | 0.00 | 0.00 |
| fixed_origin | broad_sg | 632.45 | 472.29 | 0.66 | 1 | 0.32 | 0.94 |
| fixed_origin | engineering | 631.22 | 472.20 | 0.57 | 1 | 0.30 | 0.81 |

Paired row-model intervals resample contiguous month blocks, retain each month's actual rental count, and recompute pooled RMSE. They are conditional on the fitted models and selected candidate, and omit selection/model-fitting uncertainty.

Recommendation: retain the current submission features for now. The selected industry does not show a clear row-model improvement in both availability scenarios. A favorable aggregate correlation or retrospectively best outer sector is insufficient reason to add it.

This tree control does not measure incremental value for the optimized TabPFN ensemble. Before promoting any candidate, use identical TabPFN contexts/seeds and rent-history features, select on earlier origins only, and confirm on newly available rental months. The current project's outer period has already been examined by other experiments.

## Limits and next evidence

Only 51 months cover the pandemic/reopening and a changing rent trend. Stock values were supplied as a later snapshot: survival, corporate actions, stale/delisted prices and revised adjusted prices cannot be independently reconstructed here. The broad basket includes a listing that starts September 2021; its weights change with available coverage. A frozen 2021 property model can miss changing property premiums. Rent approvals are assumed available at month end; real publication delays would require an extra lag. The ten-month holdout has been used elsewhere in this repository and is not an untouched final test. No test-set labels are available; test CSV dates extend to July 2026, much longer than the validated horizon.

For stronger evidence, collect earlier stock/rent history, point-in-time constituents, and economic series with actual release dates (employment, wages, migration, interest rates and housing supply). Test a small prespecified sector bundle against these controls. Predictive improvement does not identify a causal channel; this observational design cannot rule out common drivers or reopening confounding.

## Reproduction and outputs

Run `.venv/Scripts/python.exe analysis/compare_stock_sectors.py`. Use `--monthly-only` for the aggregate experiment alone, or `--output <directory>` to keep another run. `results.json` saves settings, selections and source/code hashes. Monthly forecast records include origin, horizon, target month and latest training-label date. The program checks that all input hashes remain unchanged.

- [bank membership](https://www.abs.org.sg/about-us/memberbanks)
- [reit sectors](https://www.reitas.sg/singapore-reits/s-reit-sectors/)
- [historical reit list](https://www.reitas.sg/wp-content/uploads/2024/09/the-pdf-doc_26-Aug-2024_1.pdf)
- [netlink business](https://www.imda.gov.sg/regulations-and-licensing-listing/nationwide-broadband-network)
