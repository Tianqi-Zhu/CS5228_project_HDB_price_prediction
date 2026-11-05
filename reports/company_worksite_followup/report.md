# Company stock returns near engineering worksites: follow-up

This follows the exploratory K&S 1 km/six-month-lag finding. Company/site/radius/lag hypotheses were frozen from the earlier report. The June 2024-March 2025 rental period has already been examined: these are robustness and development checks, not fresh confirmation. No submission file is modified.

## Is the K&S signal independent of global semiconductor and currency moves?

The original near-minus-3-6 km comparison is repeated with US S&P 500, semiconductor-sector and USD/SGD returns measured at the same six-month-lag signal date. USD-listed company returns are also converted to SGD log returns. Existing recent local rent changes, national adjusted rent growth and broad/peer Singapore stock returns remain controls. All available observations precede approval; no current/future rental label enters its prediction fit.

| model | development_control_rmse | development_stock_rmse | development_ratio | baseline_rmse | stock_rmse | delta_rmse_pct | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_controls | 4.556 | 4.424 | 0.971 | 3.631 | 3.348 | -7.817 | -0.531 | -0.048 |
| sector_fx_controls | 4.487 | 4.466 | 0.995 | 3.937 | 3.799 | -3.505 | -0.355 | -0.010 |

Errors here concern monthly property-adjusted local relative growth (log points), not individual rents. Paired three-month block intervals are conditional on fitted models and omit the earlier radius/lag search. Exact t-6 returns are used; a six-month lag is not a six-month cumulative return.

## Distance decay and spatial specificity

Disjoint 0-1, 1-2 and 2-3 km bands use the same 3-6 km reference ring. A stronger association close to the site would be more consistent with a local channel, but small counts, shared homes and a site's particular geography prevent a causal interpretation. Nominal partial-slope intervals below are not a new multiple-testing discovery claim.

| band | model | months | effect_per_10pct_stock | effect_ci_low | effect_ci_high | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| band_0_1km | original_controls | 29 | 1.104 | 0.223 | 1.986 | 0.016 |
| band_0_1km | sector_fx_controls | 29 | 0.796 | -0.365 | 1.958 | 0.168 |
| band_1_2km | original_controls | 29 | -0.190 | -1.216 | 0.835 | 0.704 |
| band_1_2km | sector_fx_controls | 29 | -1.273 | -2.650 | 0.105 | 0.068 |
| band_2_3km | original_controls | 29 | -0.026 | -0.719 | 0.666 | 0.938 |
| band_2_3km | sector_fx_controls | 29 | -0.305 | -1.110 | 0.501 | 0.439 |

Matched spatial placebos use 32 HDB-building centers at least 6 km from the K&S site and at least 1.2 km from each other. Match covariates use **2021 only**: rental density, mean rent, floor area, lease year and flat-type mix. The true site's reference density is 51.3 rentals/month, versus 2077 SGD mean rent. There is no selection on later rent growth or model errors. Some placebos can fail monthly coverage and are reported as unavailable rather than counted as successful controls.

| model | valid_placebos | true_delta_pct | placebo_median_delta_pct | placebos_improving_more | descriptive_rank_fraction |
| --- | --- | --- | --- | --- | --- |
| original_controls | 31 | -7.817 | 0.819 | 3 | 0.125 |
| sector_fx_controls | 31 | -3.505 | 0.633 | 6 | 0.219 |

The rank fraction describes this matched set; it is not a randomized-treatment p-value. Placebo neighborhoods remain correlated and may contain unrecorded employers. Sites are points, not employment-weighted factory boundaries. Matching equalizes observable reference-year housing characteristics only.

## Same location, other companies' stocks

Keep the K&S geography, six-month lag and sector/FX controls fixed, but replace only the company-return signal. If another company's stock works as well, the effect is less clearly company-specific. These are diagnostic controls, not new candidate selection.

| assigned_company_stock | baseline_rmse | stock_rmse | delta_rmse | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- |
| KLIC | 3.937 | 3.799 | -0.138 | -0.355 | -0.010 |
| V03.SI | 3.937 | 3.887 | -0.050 | -0.141 | 0.018 |
| S63.SI | 3.937 | 3.988 | 0.051 | -0.054 | 0.144 |
| FLEX | 3.937 | 4.285 | 0.348 | 0.145 | 0.671 |
| 5E2.SI | 3.937 | 3.997 | 0.060 | -0.094 | 0.129 |
| S59.SI | 3.937 | 3.936 | -0.001 | -0.082 | 0.043 |

## Does it add value for individual rentals with rent-history features?

A matched CatBoost residual model uses the existing 77 aligned TabPFN input features, including high-cardinality block/street IDs, strictly prior rent histories and the identical temporal baseline offsets. The backbone here is CatBoost, not a newly trained TabPFN. Every ablation retains the same context/query rows and fits y minus the existing temporal baseline, then adds that baseline once. The 2022-09 and 2023-07 origins have 2,000 monthly-stratified query samples each, pooled with their population weights; 2024-06 uses all 28,995 query rentals. Earlier/later scores are not interchangeable with the monthly-index RMSE above.

Nuisance inputs include site distance/exposure, standalone lagged company and US-market/sector/FX signals and availability flags. Sector-local models add distance-weighted SP500/SOX/FX terms. Only the company-local model adds exposure times the company's return, so a geometry improvement or a nearby semiconductor-cycle effect cannot alone count as company-specific value. The primary K&S kernel is Gaussian with 1 km bandwidth, truncated at 3 km. The frozen peer-site hypotheses use their earlier radii/lags. All-firm versions test the five frozen hypotheses jointly.

Both scenarios freeze rental histories at the fold origin. Monthly-updated stocks use each row's historical t-L observations. Fixed-origin stocks use t-L only if it is already observable at the fold cutoff; otherwise zero plus an unavailable indicator is used. Unknown future stocks are never filled with their realized values. This missingness policy is a diagnostic deployment assumption, not a forecast of future stock prices.

| scenario | subset | earlier_control_rmse | earlier_candidate_rmse | earlier_delta_rmse | later_control_rmse | later_candidate_rmse | later_delta_rmse | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monthly_updated | all_rentals | 615.664 | 611.614 | -4.051 | 460.344 | 460.272 | -0.072 | -1.774 | 1.011 |
| monthly_updated | klic_within_1km | 589.952 | 572.317 | -17.635 | 404.482 | 402.287 | -2.195 | -5.673 | 0.690 |
| fixed_origin | all_rentals | 614.873 | 610.602 | -4.271 | 460.276 | 460.242 | -0.034 | -1.679 | 1.011 |
| fixed_origin | klic_within_1km | 592.332 | 577.164 | -15.168 | 404.654 | 402.069 | -2.585 | -6.706 | 0.691 |

See row_incremental_comparisons.csv for nuisance-only, 77-feature rent-only and all-firm controls, and row_fold_scores.csv for separate folds and distance bands. Intervals resample paired month blocks and retain rental/sample weights; they are conditional and do not account for model selection. Full reproducible row predictions are cached under .cache/company_worksite_followup/row_predictions.csv.gz.

## Interpretation and promotion

Treat any gain as a hypothesis about predictive information, not proof that stock prices cause nearby rents. A surviving company-local signal would require better earlier pooled performance against rent-history and local-sector controls, later near-site improvement with limited whole-sample harm, robust geographic specificity, and ultimately new rental months plus a matched TabPFN evaluation. No radius or lag is reselected on the reused later period. These tests do not measure employee residence, hiring, bonuses or workplace headcounts; common drivers and commuting remain plausible explanations.

The original 63-scan search found no association surviving dependent multiple-testing correction. This follow-up does not reset that search history. There are only 51 rental months. Market controls and adjusted company prices are later-vintage histories; complete point-in-time release archives are unavailable. Current address pins locate historically documented sites, but exact historical plant occupancy and boundaries are unverified. Currency conversions use monthly endpoints; global market closes and official release timing can require additional availability lags in live use.

## Sources and reproduction

Sources: [S&P 500](https://fred.stlouisfed.org/series/SP500), [Nasdaq Composite](https://fred.stlouisfed.org/series/NASDAQCOM), [semiconductor sector](https://fred.stlouisfed.org/series/NASDAQSOX), [SGD per USD](https://fred.stlouisfed.org/series/DEXSIUS). Corporate addresses retain the independently documented sources from the original worksite report. The design is recorded in analysis/configs/company_worksite_followup.json.

Run `.venv/Scripts/python.exe analysis/acquire_worksite_market_controls.py`, then `.venv/Scripts/python.exe analysis/compare_company_worksite_followup.py`. `--panel-only` omits the individual-rental ablation. Source/code hashes and cached 77-feature bundle hashes are recorded in results.json; input data and existing experiment outputs are preserved.
