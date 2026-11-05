# COE and stock signals for HDB rent prediction

The target is monthly HDB rent, not resale price. This analysis joins each approval month to **the preceding calendar month's** COE and stock observations. It therefore assumes a forecast refreshed monthly after those observations are published; it does not represent a single forecast made at the training cutoff for every future month.

COE monthly premiums are medians of bidding rounds by category. Bid pressure is total bids received divided by total quota in that category-month. Stock returns are the median month-end adjusted-close return across symbols ending `.SI`; the three-month measure compounds those monthly median returns, and volatility is their rolling three-month standard deviation. Raw stock price levels are deliberately not compared across companies.

## Evidence

The pre-2024-04 correlation between mix-adjusted monthly rent growth and prior-month COE C premium growth is 0.306 (37 paired months). The same association moves from -0.009 in the early segment to 0.523 in the later segment. COE A/C premium levels even reverse their rent-growth correlation between the two segments. Prior-month Singapore-stock return has a pre-validation correlation of -0.181 (37 months). These are exploratory correlations, not causal evidence, and the number of independent months is small.

All forward comparisons use the same property, block and amenity baseline and the same gradient-boosting settings. The table shows the change in validation RMSE from adding each group; positive values are worse (SGD). The 2024-04 window is the project's existing, previously examined holdout.

| Validation start | COE A/C levels | COE A/C growth and bid pressure | SG stock returns/volatility | COE changes plus stocks |
| --- | ---: | ---: | ---: | ---: |
| 2022-07 | -0.21 | +1.97 | +2.79 | +3.54 |
| 2023-01 | -0.50 | +0.59 | +2.59 | +3.81 |
| 2023-07 | -0.05 | +0.07 | -0.18 | +0.02 |
| 2024-04 | +0.14 | +0.90 | +0.72 | +0.97 |

The baseline RMSEs are 658.14, 642.74, 567.69 and 478.82 SGD, respectively. These windows also differ in training history and calendar period; their absolute RMSEs are not a controlled measure of market-feature value.

## Recommendation

Keep the property/spatial/time baseline as the default. None of the tested COE or stock groups yields a stable, material out-of-time gain. If the team wants a small economic-feature candidate for further testing, use only lagged COE A/C premiums; their early improvements are below S$0.50 and disappear in the latest window, so they should not be presented as established gains. Do not include the stock basket by default. The supplied auxiliary snapshot contains preceding-month observations for all 17 months in the provided test file (2025-03 through 2026-07), so test-set coverage is not the obstacle. For a real fixed-origin prospective forecast, however, future realized market observations would not be known at the cutoff; use known-at-cutoff features or a separately validated macro forecast instead.

Reproduce from the repository root with `python -m analysis.main economic-signals` in the `cs5228` Conda environment. The generated `monthly_series.csv`, `rent_growth_correlations.csv`, `forward_scores.csv`, two PNGs and `results.json` are in this directory. Input CSVs remain unchanged; `results.json` records their SHA-256 hashes and modeling settings.
