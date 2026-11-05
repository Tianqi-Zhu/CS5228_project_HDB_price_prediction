# Follow-up findings: company stocks and rents near engineering sites

Recommendation: do not add a company-location stock feature to the current submission on this evidence. The tentative K&S relationship weakens under stronger market controls, is not unusual among matched distant neighborhoods, and does not deliver a stable incremental gain on saved TabPFN forecasts. This is a robustness study on already examined rental periods, not new causal identification or untouched confirmation.

The prior K&S hypothesis remains fixed: 23A Serangoon North Avenue 5, 1 km proximity, six-month-lag company return. ST Engineering, Venture, Flex and Seatrium retain their previously selected site/radius/lag combinations. No later-period radius/lag re-selection occurs.

## Aggregate local-rent check

| model | development_ratio | baseline_rmse | stock_rmse | delta_rmse_pct | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| original_controls | 0.971 | 3.631 | 3.348 | -7.817 | -0.531 | -0.048 |
| sector_fx_controls | 0.995 | 3.937 | 3.799 | -3.505 | -0.355 | -0.010 |

Adding US broad-market, semiconductor and SGD/USD controls reduces the incremental K&S gain relative to its matched control. The new control's absolute error is also larger: compare both the stock model and the control, rather than crediting a gain against a weaker comparator as proof of information. These are local-growth log-point errors, not SGD individual-rental errors.

| model | valid_placebos | true_delta_pct | placebo_median_delta_pct | placebos_improving_more | descriptive_rank_fraction |
| --- | --- | --- | --- | --- | --- |
| original_controls | 31 | -7.817 | 0.819 | 3 | 0.125 |
| sector_fx_controls | 31 | -3.505 | 0.633 | 6 | 0.219 |

Thirty-two placebo centers were matched on 2021-only housing characteristics and spread geographically; thirty-one have adequate later coverage. A descriptive rank fraction is not a randomized p-value. Six matched distant neighborhoods improve at least as much as the actual site after sector/currency controls.

| band | delta_rmse_pct | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- |
| band_0_1km | -3.505 | -0.355 | -0.010 |
| band_1_2km | 0.829 | -0.220 | 0.252 |
| band_2_3km | -8.146 | -0.496 | 0.236 |

The strengthened-control results do not show a reliable monotonic distance-decay pattern. The wrong-company diagnostic favors KLIC over the five substitute stocks at this particular site, but that alone does not establish workplace causation or geographic uniqueness.

## Individual-rental feature ablation with rent histories

CatBoost keeps the original 77 aligned rent-history/property inputs and temporal baseline fixed, then adds distance/global signals, local sector/currency signals, and finally local company-return interactions. Earlier origins have monthly-stratified queries with population weights; the reused later origin evaluates all 28,995 rentals, including 418 within 1 km of K&S. Both stock-availability scenarios are tested. Rental histories remain frozen at each fold origin.

| scenario | subset | candidate | baseline | earlier_delta_rmse | later_delta_rmse | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| monthly_updated | all_rentals | klic_local | klic_sector_local | -4.051 | -0.072 | -1.774 | 1.011 |
| monthly_updated | all_rentals | klic_local | history77 | 2.309 | 1.056 | 0.628 | 1.434 |
| monthly_updated | klic_within_1km | klic_local | klic_sector_local | -17.635 | -2.195 | -5.673 | 0.690 |
| monthly_updated | klic_within_1km | klic_local | history77 | -14.479 | 0.128 | -1.638 | 2.040 |
| fixed_origin | all_rentals | klic_local | klic_sector_local | -4.271 | -0.034 | -1.679 | 1.011 |
| fixed_origin | all_rentals | klic_local | history77 | 1.297 | 1.025 | 0.577 | 1.434 |
| fixed_origin | klic_within_1km | klic_local | klic_sector_local | -15.168 | -2.585 | -6.706 | 0.691 |
| fixed_origin | klic_within_1km | klic_local | history77 | -9.632 | -0.090 | -1.763 | 1.901 |

These deltas are individual-rental SGD RMSE. Sector-local controls distinguish company-specific interactions from a local industry-cycle effect. The 77-feature reference prevents a gain against an unnecessarily weak nuisance model from being mistaken for improvement over rent histories. Adding a feature can change CatBoost tree topology globally; a gain outside spatial support is not evidence that nearby employees caused rents to change. This is a single-seed controlled screen, not a matched TabPFN-backbone feature fit.

## Exactly local correction on saved TabPFN ensemble backtests

This second check preserves the saved TabPFN predictions. Correction inputs are multiplied by a 1 km Gaussian exposure truncated at 3 km. Weighted RMS scaling, no intercept and no centering make every correction exactly zero beyond 3 km. It cannot improve unrelated rows through a changed model topology. A first historical origin fits the correction; a second chooses among no correction, sector-only and company-local models; both earlier origins refit before the later reused period.

| scenario | candidate | baseline | control_rmse | candidate_rmse | delta_rmse | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_origin | company_local | sector_local | 428.332 | 428.248 | -0.085 | -0.282 | 0.065 |
| fixed_origin | company_local | uncorrected | 397.688 | 428.248 | 30.559 | 6.002 | 49.202 |
| monthly_updated | company_local | sector_local | 405.552 | 405.844 | 0.292 | -0.579 | 0.976 |
| monthly_updated | company_local | uncorrected | 397.688 | 405.844 | 8.156 | -2.610 | 18.751 |

The company-specific addition is negligible/inconclusive against sector/currency controls, and the full company-local correction does not beat the unchanged saved TabPFN near-site predictions in either availability scenario. Earlier local bias correction does not transfer reliably to the later regime. These are saved feature-ensemble backtests, not the all-150k submission model; no TabPFN backbone was retrained.

## Limits and next evidence

The original 63-scan search history still applies. Conditional bootstrap intervals omit model-fitting/search uncertainty, the later ten months have been reused, and workplace locations do not reveal employee headcounts, residences or commuting. Current geocodes and later-vintage stock/market histories are imperfect proxies for historical exposure. We need new rental months and evidence about actual local employment changes before claiming a causal company effect or promoting this feature.

Existing submissions and original experiment outputs are preserved. [Full follow-up methods](report.md), [strictly local TabPFN details](tabpfn_local_correction/report.md), [distance/placebo chart](01_distance_and_matched_placebos.png), and [row feature-ablation chart](02_individual_rental_increment.png).

Reproduce with `.venv/Scripts/python.exe analysis/acquire_worksite_market_controls.py`, `.venv/Scripts/python.exe analysis/compare_company_worksite_followup.py`, `.venv/Scripts/python.exe analysis/compare_worksite_tabpfn_correction.py`, then `.venv/Scripts/python.exe analysis/summarize_company_worksite_followup.py`.
