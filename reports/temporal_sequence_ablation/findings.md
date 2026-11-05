# What the temporal feature ablation shows

**Keep rental-history persistence as the conservative default. COE is a more promising sequence input than stock returns here, but no tested neural processor establishes a stable overall win.** All 86 target/specification combinations and 8,858 forecast cases completed; 17 availability/index tests passed. The later period has been examined by previous experiments.

The previous national-index analysis already compared Singapore mean-rent changes with a broad Singapore stock proxy: correlation 0.005 for the same month and -0.253 for the preceding month (95% month-block interval -0.517 to 0.067). It did not use LSTM.

## Which inputs help the LSTM?

Later June 2024–March 2025 RMSE of the property-adjusted monthly mean (SGD). Each row uses its own window chosen on earlier data. Rolling pools 1/3/6-month forecasts; fixed-origin forecasts all ten months from May 2024 information. Compare information sets within each scenario.

| group | fixed_origin | rolling |
| --- | --- | --- |
| Rent only | 258.483 | 126.409 |
| Rent + COE | 23.244 | 44.958 |
| Rent + stocks | 133.216 | 125.095 |
| Rent + stocks + COE | 31.828 | 61.503 |

Persistence scores **37.75 rolling / 45.39 fixed-origin**. The rent+COE LSTM uses six months of history for rolling forecasts and twelve for the fixed path; the joint LSTM uses three months. Adding stocks does not improve on the rent+COE LSTM in either scenario. This comparison permits each input set its earlier-selected window; all fixed-window results are in all_scores.csv.

The joint LSTM improves the later fixed path from 45.39 to **31.83**, while rent+COE reaches **23.24**. However, persistence won earlier model selection, and the joint LSTM worsens rolling RMSE to **61.50**. Simple rent momentum/EWMA also perform well on the later fixed path (21.24/23.02), despite losing earlier selection. Choosing among these after inspecting this period would overstate the evidence.

## Does it improve individual-rental prediction?

| scenario | candidate | model | gamma | corrected_rmse_sgd | delta_rmse_sgd |
| --- | --- | --- | --- | --- | --- |
| fixed_origin | joint | ridge_lags__rent_stock_coe__w12 | 0.318 | 470.375 | -1.256 |
| fixed_origin | joint_lstm | lstm__rent_stock_coe__w3 | 0.117 | 470.905 | -0.726 |
| fixed_origin | persistence | persistence__rent__w12 | 0.000 | 471.631 | 0.000 |
| rolling | joint | ridge_lags__rent_stock_coe__w12 | 0.494 | 470.049 | -1.582 |
| rolling | joint_lstm | lstm__rent_stock_coe__w3 | 0.411 | 470.271 | -1.360 |
| rolling | persistence | persistence__rent__w12 | 0.754 | 469.618 | -2.013 |
| rolling | rent_stock | ridge_lags__rent_stock__w12 | 0.435 | 469.878 | -1.753 |
| rolling | rent_coe | ridge_lags__rent_coe__w12 | 0.457 | 469.954 | -1.677 |
| rolling | rent_ridge | ridge_lags__rent__w12 | 0.454 | 469.666 | -1.965 |
| rolling | rent_lstm | lstm__rent__w12 | 0.396 | 470.096 | -1.534 |
| rolling | rent_stock_lstm | lstm__rent_stock__w12 | 0.463 | 470.180 | -1.451 |
| rolling | rent_coe_lstm | lstm__rent_coe__w6 | 0.400 | 470.284 | -1.347 |
| fixed_origin | rent_stock | ridge_lags__rent_stock__w12 | 0.129 | 469.387 | -2.244 |
| fixed_origin | rent_coe | ridge_lags__rent_coe__w12 | 0.353 | 470.160 | -1.471 |
| fixed_origin | rent_ridge | ridge_lags__rent__w12 | 0.141 | 469.666 | -1.965 |
| fixed_origin | rent_lstm | lstm__rent__w12 | 0.096 | 469.398 | -2.233 |
| fixed_origin | rent_stock_lstm | lstm__rent_stock__w12 | 0.112 | 470.064 | -1.567 |
| fixed_origin | rent_coe_lstm | lstm__rent_coe__w12 | 0.128 | 471.228 | -0.403 |

The uncorrected property/amenity tree scores **471.63** SGD RMSE. All correction scales above are learned from earlier errors only. Monthly-updated rent persistence scores **469.62**, beating the joint LSTM correction at **470.27**. The fixed-origin joint LSTM improves to **470.91** (a 0.73 SGD gain). These are controlled tree results; no TabPFN adapter or submission was refitted. The additional same-processor rent/stock/COE transfer rows are follow-up diagnostics added after viewing the aggregate ablation; models/windows still use earlier selection and gamma still uses earlier rental errors.

The matched fixed-origin rent-only LSTM correction scores **469.40**, better than the joint LSTM at **470.91**. Stock-plus-rent lag regression scores **469.39**, versus **469.67** for rent-only lag regression: a small exploratory difference. The best earlier row-correction candidate in the extended comparison is rent+stock LSTM; it scores **470.06** fixed-origin and **470.18** rolling later, without beating the simpler controls in this later period. Better national-mean forecasting does not guarantee better individual-rental predictions. Gamma shrinkage can also make a biased national forecast a useful small tree-bias correction; it does not establish that stocks predict rents.

## Robustness and practical decision

The joint fixed-path gain persists for seeds 43/44 (25.92/22.35) and shrinks with the equal-weight stock proxy (43.64). With an extra full month of publication delay it deteriorates to **99.35**. Input availability is therefore a material part of the result. Under the same delay, adjusted-mean persistence scores 54.52 on the fixed path, outperforming the delayed joint LSTM. The raw-mean fixed path remains favorable for the LSTM under delay, so even this sensitivity depends on which rental index is used.

| target | scenario | delayed_persistence_rmse_sgd | delayed_joint_lstm_rmse_sgd | delta_rmse_sgd |
| --- | --- | --- | --- | --- |
| mean_rent | rolling | 38.923 | 78.578 | 39.654 |
| mean_rent | fixed_origin | 55.362 | 49.099 | -6.263 |
| adjusted_mean_rent | rolling | 38.749 | 73.703 | 34.954 |
| adjusted_mean_rent | fixed_origin | 54.521 | 99.351 | 44.830 |

Additional earlier-selected rent+COE LSTM sensitivity:

| target | scenario | check | model | outer_rmse_sgd |
| --- | --- | --- | --- | --- |
| adjusted_mean_rent | rolling | seed43 | lstm__rent_coe__w6 | 46.840 |
| adjusted_mean_rent | rolling | seed44 | lstm__rent_coe__w6 | 35.829 |
| adjusted_mean_rent | rolling | extra_month_delay | lstm__rent_coe__w6 | 53.189 |
| adjusted_mean_rent | fixed_origin | seed43 | lstm__rent_coe__w12 | 24.333 |
| adjusted_mean_rent | fixed_origin | seed44 | lstm__rent_coe__w12 | 33.302 |
| adjusted_mean_rent | fixed_origin | extra_month_delay | lstm__rent_coe__w12 | 49.118 |

There are **51 independent calendar months at most**, with serial dependence, rather than 150,000 independent macro sequences. A tiny LSTM cannot establish a general advantage from these data. Treat COE plus short/long rent histories as the next candidate to confirm on new months, and keep stock inputs optional until they beat that control consistently. Do not replace the existing final ensemble from this study alone.

The detailed report uses a separate horizon-normalized MSE ranking. Its averages weight each horizon relative to that horizon's persistence error; they can exceed 1 even when pooled SGD RMSE improves. The SGD table above avoids conflating these two objectives.

## Files and reproduction

- [Full methods and results](report.md)
- [All input/processor RMSE comparisons](input_processing_rmse_sgd.csv)
- [Extended individual-rental transfer](row_transfer_extended_scores.csv)
- [Verification](verification.json)

Run `.venv/Scripts/python.exe analysis/compare_temporal_sequence_ablation.py`, then `.venv/Scripts/python.exe analysis/summarize_temporal_sequence_ablation.py`. The second command derives these findings and performs the explicitly labeled follow-up checks. Original input files and both final submission CSVs remain unchanged.
