# Company stock returns and HDB rents near engineering worksites

No site passes both the primary association screen and later forecast-improvement check. Keep company-specific location/return features out of the current submission on this evidence.

Scope: 11 documented sites, 6 parent-company stocks; 9 sites have enough earlier data for the primary 2 km test. ComfortDelGro, banks and dispersed consumer-service firms are excluded following the requested manufacturing/engineering focus.

## What is tested

For each site, compare rentals within 2 km with rentals in a disjoint 3-6 km ring. A model fitted on 2021 rentals only removes fixed building, flat type/model, area and lease-year effects from log rent. The response is the monthly CHANGE in this adjusted near-minus-ring premium, multiplied by 100. This tests local relative rent growth rather than national rent growth or expensive neighborhoods having high rent levels.

The primary company signal is the preceding calendar month's adjusted-close log return. Rent/stock predictors always precede the target approval month. The control model uses recent local relative changes, prior national adjusted rent growth, prior broad Singapore stock returns and an engineering/manufacturing peer basket excluding the focal company. Comparisons include a zero-change local-premium forecast. All models use the same eligible monthly cases; the better rent/market control is chosen using earlier forecasts only.

## Primary results: 2 km, lag 1

| site_id | development_baseline | baseline_rmse | stock_rmse | delta_rmse | improved_months | delta_ci_low | delta_ci_high | effect_per_10pct_stock | q_primary_tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| st_engineering_amk | rent_market_control | 2.873 | 2.778 | -0.094 | 7 | -0.237 | 0.012 | -0.257 | 1.000 |
| st_engineering_boon_lay | rent_market_control | 2.716 | 2.768 | 0.052 | 4 | 0.013 | 0.089 | -1.784 | 1.000 |
| st_engineering_paya_lebar | rent_market_control | 6.442 | 6.535 | 0.093 | 5 | -0.094 | 0.221 | -2.022 | 1.000 |
| venture_amk | rent_market_control | 1.822 | 1.836 | 0.013 | 6 | -0.097 | 0.070 | 0.667 | 1.000 |
| venture_marsiling | rent_market_control | 3.835 | 3.872 | 0.037 | 3 | 0.009 | 0.065 | -0.869 | 1.000 |
| venture_woodlands | rent_market_control | 2.180 | 2.185 | 0.005 | 6 | -0.006 | 0.022 | 0.484 | 1.000 |
| kulicke_soffa_serangoon | rent_market_control | 1.781 | 1.820 | 0.039 | 0 | 0.008 | 0.060 | 0.309 | 1.000 |
| flex_changi | rent_market_control | 1.700 | 1.762 | 0.062 | 6 | -0.026 | 0.121 | -0.759 | 1.000 |
| seatrium_admiralty | rent_market_control | 1.819 | 1.790 | -0.029 | 6 | -0.096 | 0.091 | 0.216 | 1.000 |

RMSE concerns monthly local relative growth, in log points (approximately percentage points), not individual-rental SGD RMSE. Negative deltas favor company stock. Intervals are paired three-month moving-block bootstrap over the ten June 2024-March 2025 validation target months, conditional on the fitted models. They omit model-fitting/selection uncertainty and are not adjusted for the sensitivity search or the five company-level outer comparisons. The development-only slope column is the estimated local relative rent-growth change for a +10% stock move; it is an association, not an intervention effect.

Association regressions use January 2022-May 2024 data with OLS and Newey-West covariance (three monthly lags, small-sample factor, approximate t reference). q_primary_tests applies Benjamini-Yekutieli multiple-testing correction across eligible 2 km/lag-1 sites; q_all_scans corrects across all eligible site/radius/lag scans. Development_associations.csv also shows early/late slopes. Small samples make these estimates fragile.

## Earlier-selected sensitivity candidates

Radii 1/2/3 km and company-return lags 1/3/6 are sensitivity scans, with 3-6 km comparison rings. The site/radius/lag for each company is chosen on earlier expanding-window forecasts only, using stock RMSE divided by the better control's RMSE. A forced best stock candidate can still fail to beat the control; development_supports_stock records this. Baseline competes with stock for adoption. This selection explores multiple localities and should not be interpreted as a discovered causal radius.

| company | site_id | radius_m | lag | development_stock_control_ratio | development_supports_stock | baseline_rmse | stock_rmse | delta_rmse | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flex | flex_changi | 2000 | 3 | 0.999 | True | 1.700 | 1.709 | 0.009 | -0.049 | 0.170 |
| Kulicke & Soffa | kulicke_soffa_serangoon | 1000 | 6 | 0.971 | True | 3.631 | 3.348 | -0.284 | -0.531 | -0.048 |
| ST Engineering | st_engineering_amk | 3000 | 3 | 0.925 | True | 2.416 | 2.340 | -0.076 | -0.354 | 0.269 |
| Seatrium / Sembcorp Marine | seatrium_admiralty | 2000 | 1 | 1.008 | False | 1.819 | 1.790 | -0.029 | -0.096 | 0.091 |
| Venture | venture_woodlands | 3000 | 3 | 0.936 | True | 1.872 | 1.787 | -0.084 | -0.136 | 0.129 |

1 earlier-supported sensitivity candidate(s) have a negative conditional outer RMSE interval. Report these alongside the primary specification; selecting a new winner after viewing outer errors would reuse the validation data.

None of the 63 exploratory association scans survives a 5% dependence-adjusted false-discovery threshold.

The K&S 1 km/six-month-lag specification is a useful follow-up hypothesis if it improves validation. Its 1 km ring is relatively noisy and the company is US-listed; neither a conditional bootstrap interval nor an uncorrected association p-value establishes a local employment effect.

| site_id | remote_sites | real_delta_pct | remote_median_delta_pct | remote_sites_improving_more |
| --- | --- | --- | --- | --- |
| flex_changi | 8 | 0.522 | 1.414 | 4 |
| kulicke_soffa_serangoon | 1 | -7.817 | -2.790 | 0 |
| seatrium_admiralty | 6 | -1.608 | 1.824 | 2 |
| st_engineering_amk | 4 | -3.150 | 1.080 | 0 |
| venture_woodlands | 5 | -4.508 | 0.721 | 0 |

## Coverage and geography

| site_id | rental_buildings | nearest_rental_km | development_near_median_n | development_valid_months | outer_valid_months | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| st_engineering_amk | 97 | 1.046 | 45.000 | 29 | 10 | True |
| st_engineering_boon_lay | 571 | 0.414 | 175.000 | 29 | 10 | True |
| st_engineering_paya_lebar | 143 | 0.792 | 42.000 | 29 | 10 | True |
| venture_amk | 430 | 0.485 | 131.000 | 29 | 10 | True |
| venture_marsiling | 281 | 0.262 | 55.000 | 29 | 10 | True |
| venture_woodlands | 318 | 0.163 | 93.000 | 29 | 10 | True |
| kulicke_soffa_serangoon | 506 | 0.122 | 132.000 | 29 | 10 | True |
| flex_changi | 346 | 0.800 | 120.000 | 29 | 10 | True |
| sia_engineering_changi | 3 | 1.545 | 1.000 | 0 | 0 | False |
| seatrium_tuas | 0 | 12.177 | 0.000 | 0 | 0 | False |
| seatrium_admiralty | 230 | 0.697 | 66.000 | 29 | 10 | True |

A near or comparison monthly cell needs at least 20 rentals; no target is filled across a missing month. Sites/radii require at least 24 valid local growth months in January 2022-May 2024, without using outer coverage to decide eligibility. Prediction fits begin after eight mature complete-case monthly labels. All valid cases are saved; sparse series are excluded explicitly. At least eight outer months are needed for confirmation.

The Tuas yard is over 12 km from the nearest HDB rental block in the supplied data, so no within-3 km HDB effect can be measured. SIA Engineering's airport site has extremely sparse nearby HDB support. These exclusions are lack of data, not estimates of zero effect. Their workers may commute farther or use other housing types.

Site_coverage.csv lists all radii, and neighborhood_overlap.csv measures overlapping 2 km rental-building sets. Venture's Ang Mo Kio/K&S neighborhoods and its Woodlands/Seatrium Admiralty neighborhoods overlap substantially; these are not independent firm-level experiments. Remote_site_negative_controls.csv applies each focal stock to unrelated-company sites at least 6 km away, with the same radius, lag and forecast procedure, for both the primary and earlier-selected specifications. Spatial_specificity_summary.csv compares percentage RMSE changes against these remote controls. These checks assess whether any apparent gain also occurs away from that company's operations; they are exploratory, not a randomized spatial design.

## Documented sites and sources

- **st_engineering_amk (S63.SI)**: 1 Ang Mo Kio Electronics Park Road, 567710. Group headquarters and industrial/commercial property. [Historical evidence (2021)](https://www.stengg.com/getmedia/1447e8e4-cb80-48fa-ac4f-1fc6745dad5c/st-engineering-annual-report-2021.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=567710&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **st_engineering_boon_lay (S63.SI)**: 249 Jalan Boon Lay, 619523. Land systems industrial site; major property in 2021 annual report. [Historical evidence (2021)](https://www.stengg.com/getmedia/1447e8e4-cb80-48fa-ac4f-1fc6745dad5c/st-engineering-annual-report-2021.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=619523&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **st_engineering_paya_lebar (S63.SI)**: 540 Airport Road, 539938. Aerospace operating/maintenance site. [Historical evidence (2020)](https://www.caas.gov.sg/docs/default-source/pdf/sar145-summary_02_nov_2020.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=539938&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **venture_amk (V03.SI)**: 5006 Ang Mo Kio Avenue 5, 569873. Principal place of business and group electronics companies. [Historical evidence (2019)](https://venture.listedcompany.com/newsroom/20200415_070135_V03_N19AQL37CRHVNIUW.1.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=569873&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **venture_marsiling (V03.SI)**: 28 Marsiling Lane, 739152. GES and Scinetic Engineering group operating address. [Historical evidence (2019)](https://venture.listedcompany.com/newsroom/20200415_070135_V03_N19AQL37CRHVNIUW.1.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=739152&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **venture_woodlands (V03.SI)**: 211 Woodlands Avenue 9, 738960. Univac design and precision engineering group operating address. [Historical evidence (2019)](https://venture.listedcompany.com/newsroom/20200415_070135_V03_N19AQL37CRHVNIUW.1.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=738960&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **kulicke_soffa_serangoon (KLIC)**: 23A Serangoon North Avenue 5, 554369. Singapore principal office of semiconductor equipment company. [Historical evidence (2021)](https://investor.kns.com/download/K%26S_2021.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=554369&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **flex_changi (FLEX)**: 2 Changi South Lane, 486123. Singapore business address and Changi manufacturing-group location; not assumed to be sole global headquarters. [Historical evidence (2021)](https://www.sec.gov/Archives/edgar/data/866374/000086637421000040/0000866374-21-000040-index.htm); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=486123&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **sia_engineering_changi (S59.SI)**: 31 Airline Road, 819831. Registered office and aircraft engineering/maintenance site. [Historical evidence (2021)](https://www.siaec.com.sg/content/dam/siaec/web-assets/pdfs/our-company/investor-relations/shareholder_meetings/2021/2021_Letter_to_Shareholders.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=819831&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **seatrium_tuas (5E2.SI)**: 80 Tuas South Boulevard, 637051. Principal Tuas yard and head-office address; 2023 group merger changes exposure. [Historical evidence (2021)](https://investors.seatrium.com/newsroom/20211020_185426_S51_ATZCOKMNZANUX51Y.3.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=637051&returnGeom=Y&getAddrDetails=Y&pageNum=1).
- **seatrium_admiralty (5E2.SI)**: 60 Admiralty Road West, 759947. Admiralty shipyard operating site; 2023 group merger changes exposure. [Historical evidence (2021)](https://investors.seatrium.com/newsroom/20211020_185426_S51_ATZCOKMNZANUX51Y.3.pdf); [OneMap geocode](https://www.onemap.gov.sg/api/common/elastic/search?searchVal=759947&returnGeom=Y&getAddrDetails=Y&pageNum=1).

Historical source documents around 2020-2022 establish addresses; the fixed-site analysis assumes continued operation through March 2025. The locations are not employment counts or an exhaustive factory inventory. A group headquarters can contain fewer operational workers than a plant. Exact facility boundaries, staff residence/commuting patterns and yearly occupancy have not been reconstructed. Current OneMap address pins locate these documented addresses and may differ from historical entrances or property centroids.

## Interpretation and feature decision

This design tests the proposed local predictive relationship more directly than a nationwide stock basket. It cannot establish that a company's stock price CAUSES nearby rents to change. Earnings, hiring, bonuses, global manufacturing demand, housing supply, COVID/reopening and transport access may drive both; stock prices may anticipate those drivers. Headquarters proximity alone is an imperfect exposure measure, especially for globally diversified companies. Seatrium's 2023 merger changes the scope of its historical stock exposure.

FLEX and KLIC are US-listed stocks in the supplied file. We use percentage/log returns, not mixed-currency price levels; separate US-market, currency and sector factors are not available here, so their firm-specific interpretation is weaker. Stock adjusted histories are later snapshots, not point-in-time archives. The rental data have 51 distinct months with serial dependence, not 150,000 independent stock observations. Building-specific rent premiums may evolve after the fixed 2021 adjustment. Publication timing is assumed month-end; real rental-release delays would need additional lags. This outer period has been examined in other project experiments and is not an untouched final test.

If a candidate is pursued, add distance-weighted company-return interactions together with standalone distance, company-return and local-history controls, rather than assigning raw stock prices to nearby homes. Require a new paired TabPFN evaluation and new rental periods before promotion. No final submission or TabPFN model was changed by this experiment.

## Reproduction

1. `.venv/Scripts/python.exe analysis/acquire_company_worksites.py` (public OneMap address geocodes, cached).
2. `.venv/Scripts/python.exe analysis/compare_company_worksite_stocks.py` (read-only inputs, single-thread CPU analysis).

Outputs include the adjusted monthly panel, full forward predictions, coverage/exclusions, all association scans, primary tests, earlier-selected sensitivities, spatial negative controls and charts. results.json records source/code hashes, selections, parameters and unchanged-input checks.
