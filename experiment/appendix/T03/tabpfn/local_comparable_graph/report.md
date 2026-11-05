# Block-level local comparable-rent graph

All scores are fixed-origin batch-forecast RMSE in SGD. A negative delta improves on the historical baseline in the same fold.

| Validation start | Historical | + local rent | Delta | + rent and premium | Delta | + local residual | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023-01 | 639.02 | 640.62 | +1.59 | 638.37 | -0.66 | 639.47 | +0.45 |
| 2023-07 | 563.35 | 564.61 | +1.26 | 564.11 | +0.76 | 562.86 | -0.49 |
| 2024-04 | 469.61 | 469.68 | +0.06 | 469.62 | +0.01 | 469.83 | +0.22 |
| 2023-12 | 489.60 | 490.72 | +1.12 | 489.26 | -0.34 | 489.97 | +0.37 |

The first three validation windows are disjoint (6, 9, and 12 months). The final 16-month window overlaps them and is a separate horizon check.

## Graph and features

Each HDB block connects to its 12 nearest *other* blocks within 1 km. Comparables must share flat type and be in the same or adjacent 10 m² floor-area bin. Distance and bin similarity weight the mean. For each 3-, 6-, and 12-month lookback, the raw variant receives local rent mean and count. The premium variant adds deviations from the observed monthly flat-type × area-bin market mean. The residual variant uses local out-of-time hedonic residual means and comparable counts. All variants retain the baseline property, spatial, amenity, and historical-rent features.

Residuals are generated in six-month forward blocks from the historical row model trained only on earlier months, starting in 2022-01. The residual bank is centered by observed month and flat type; residuals from 2021 are unavailable. A validation fold only consumes residuals from its own fitting period.

Every fitting row sees only earlier approval months. All validation months reuse the fitting-cutoff snapshot; validation targets never update graph features. For an overlapping prediction month, only strictly earlier training months are eligible.

## Coverage and interpretation

A 12-month comparable is available for 97.7%–98.1% of validation rows, and an out-of-time residual is available for 97.7%–98.1%. The median row has 17–18 raw comparables. Raw local rent worsens all four folds. See the fold table for the two normalized variants; retain the historical baseline unless a graph variant improves consistently across forward windows.

This is one fixed graph definition and model setting, not a general test of all graphs or a trained graph neural network. See `fold_metrics.csv`, `monthly_metrics.csv`, `coverage.csv`, and `results.json` for details. Run `python analysis/compare_local_comparable_graph.py` from the repository root.
