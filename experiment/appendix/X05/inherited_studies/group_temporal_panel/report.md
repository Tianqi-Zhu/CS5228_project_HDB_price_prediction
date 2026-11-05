# Market trends by planning area and flat type

`PLANNING_AREA` from the HDB block table is used as a district proxy because the rental table has no district or postal-district column. The address join matched all 150,000 labeled rental rows. The analysis aggregates monthly mean and median rent by planning area, flat type, and their intersection. It requires at least 20 rentals in both the current and preceding month for a group-month to enter a comparison.

## Correlations

`group_correlations.csv` reports pre-2024-04 Pearson and Spearman correlations between *month-on-month group rent change* and prior-month COE A/C premium growth, COE C bids-to-quota ratio, or Singapore-stock returns. Separate planning-area heatmaps show **mean-rent growth** and **median-rent growth**, using the same color scale. Each reported pair has at least 24 valid months, with early and late-period correlations shown separately. This yields 24 planning areas, 5 flat types and 50 planning-area × flat-type groups; sparse groups are omitted. These are relationships across months, not 150,000 independent rental observations. Many group/signal combinations were examined, so a large isolated correlation is not a discovery claim.

For planning-area mean-rent growth, COE C bid pressure is positively correlated in 20 of 24 eligible areas; the median area correlation is 0.148. Punggol has `r=0.268` over 38 months, with early and late values 0.114 and 0.332. Singapore-stock one-month return has a median area correlation of -0.103. For median-rent growth, COE C bid pressure is positive in 17 of 24 areas, with a median area correlation of 0.089. Compare the [mean-rent](01_planning_area_correlations.png), [median-rent](04_planning_area_median_correlations.png), and [flat-type](03_flat_type_correlations.png) heatmaps; near-white is near zero.

## Temporal baseline

For each eligible group and each rent statistic, a Ridge model predicts the next month's group mean or median from the actual previous month's rent, a linear month index, and annual sine/cosine seasonality. Three variants add COE, stock, or both signal groups. Models are refitted on expanding time windows, with imputation and scaling learned only from fitting months. `group_forward_scores.csv` gives group-level RMSE; `pooled_forward_scores.csv` pools errors over eligible group-months, giving each month equal weight rather than each rental row.

At the planning-area × flat-type grain, adding COE changed pooled **median-rent** RMSE relative to the rent-history baseline by +S$22.63 (2023-01 validation), +S$9.78 (2023-07), and -S$17.50 (2024-04). Positive is worse. Adding stocks changed it by +S$16.78, +S$1.24, and -S$2.76. Mean-rent results show a similar reversal. The late COE gain is therefore not stable across forward windows and does not overturn the row-level result in `experiment/studies/economic_signals/` that market features have no consistent benefit.

This is a **monthly-updated one-step-ahead** exercise: validation month `t` may use actual group rent and market observations from `t-1`. It is not directly usable for a batch forecast of many future test months, because future previous-month group rent is unknown unless predictions are recursively fed back or new labels arrive. The block planning-area mapping is a supplied snapshot, and changes in property mix within groups can also affect their mean/median rent trends. Correlation and predictive improvement do not establish a causal COE or stock effect.

Run `python -m analysis.main group-temporal` from the repository root in the `cs5228` environment. The script also writes `monthly_panel.csv`, four PNG figures and `results.json` with methods and source hashes; it does not alter any source CSV.
