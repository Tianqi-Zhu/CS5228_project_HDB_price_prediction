[Chinese version](history_feature_optimization.zh-CN.md)

# Detailed feature-engineering experiments

This round completed 18 feature ablations, three compact-policy confirmations and four market-baseline decompositions, each covering three annual windows and
one sixteen-month window: 100 fixed-configuration LightGBM fits. Parameters, source hashes and complete results are in
`output/experiment_runs/features_history_optimization`.

Conclusion: the default shrinks from 31 to **27 inputs** with identical performance. Historical premiums show modest potential, but their compact
version is insufficiently stable for the default. Larger gains come from market-baseline decomposition, which needs separate testing at market turning points.

## 1. Confirmed duplicate information

The 150,000 training records contain 14,828 normalized-address/flat-type combinations. Within each, flat model,
area and lease commencement year are constant. Thus the four original `SIGNATURE` and `BLOCK_FLAT` history statistics
are identical row by row: mean, count, latest-month mean and months since the latest observation.

Exact numerical equality was checked across every training row; see
`output/experiment_runs/features_history_optimization/redundancy_audit.json`.
After deleting four columns, all four windows have exactly the same RMSE as the original policy.

The new default `RentalFeatureEngineer()` uses `compact_history_27`:

- Retain the 19 static fields.
- Keep four history statistics each for block/flat-type and planning-area/flat-type: eight in total.
- Reproduce the original 31 inputs with `RentalFeatureEngineer("compact_history")`.

This equivalence follows from this dataset's attribute structure. Recheck if future records have different areas, models or
lease years within the same block/type; it is not a general property-data rule.

## 2. Bottlenecks revealed by historical coverage

See `output/experiment_runs/features_history_optimization/history_support.csv`.
In validation from 2024-04, block/type history covers 98.83% of rows, with median support 11;
12.14% of rows have at most three earlier records. Planning-area/type coverage is complete, with median support
1,899. Median fine-grained history age is ten months, rising to twelve in the sixteen-month window.

Small support and stale histories therefore warrant attention; widespread missing history is not the main explanation. Counts
convey statistical reliability, while `RECENCY_MONTHS` explicitly measures staleness.

## 3. Eighteen matched-model ablations

All candidates use every original fitting row, identical LightGBM settings and identical windows. The three ranking windows are
2022-04 through 2023-03, 2023-04 through 2024-03 and 2024-04 through 2025-03. Report the sixteen-month window separately;
it overlaps annual windows and is excluded from pooling. History queries use only fitting labels before the prediction month.

| Policy | Features | Pooled three-year RMSE | Latest annual RMSE | 16-month RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original default | 31 | 591.96 | 468.49 | 491.22 |
| Remove duplicate history (new default) | 27 | 591.96 | 468.49 | 491.22 |
| Add same-type monthly premiums | 43 | 590.42 | 467.02 | 489.45 |
| Add historical market context | 37 | 586.95 | 468.98 | 491.68 |
| History mean with 12-month half-life decay | 31 | 590.64 | 469.85 | 494.18 |
| Retain only the latest 12 months of history | 31 | 593.69 | 475.41 | 493.96 |
| Shrink history means toward coarser-group means | 31 | 592.13 | 469.35 | 500.41 |
| Add area/room count, room count and completion-to-lease gap | 34 | 591.99 | 468.44 | 492.70 |
| Add amenity counts within 0.5km and 2km | 37 | 592.07 | 468.45 | 491.53 |
| Add street category | 32 | 600.09 | 475.53 | 496.83 |
| Add address category | 32 | 598.75 | 476.03 | 501.86 |

See all eighteen candidates in `pooled_metrics.csv`, `fold_metrics.csv` and `monthly_errors.csv`.

### Distinguish historical rent means from premiums

For each rent in an observed month m, calculate:

```text
premium = rent - same-flat-type mean rent in that month
```

Aggregate by address/type and other levels with hierarchical smoothing equivalent to 20 pseudo-observations. Encoding for month t includes only
fitting months m < t, so its inputs do not contain the same-month type mean. It represents property/location differences
relative to the market then; ordinary rent means also contain the overall market level.

Combining original histories with premiums improves slightly in all three annual windows and the sixteen-month window; replacing histories with premiums worsens results.
Price levels and relative premiums are therefore complementary in this experiment.

This does not show that all twelve added fields help: some duplicate original fields, and smoothing identical groups
can produce different shrinkage. After further compaction to 31 fields, pooled RMSE falls only from 591.9623 to
591.9048, while the early annual score worsens from 711.08 to 712.24. See
`output/experiment_runs/features_history_optimization/refined`.
Retain `compact_premium_research` as a reproducible research option; keep the simpler 27-input default.

### More recent history does not win consistently

A recent twelve-month window, six-/twelve-month half-lives and hierarchical smoothing help some annual periods but worsen the latest annual or
long-horizon window. History windows freeze at the fitting origin; future query dates do not gradually empty the historical pool.
These negative results therefore cannot be blamed on that implementation error.

Street and address categories worsen all four windows. Derived property features and multiple amenity radii also give no stable increment,
so they are excluded by default. These conclusions concern this learner/settings, not the universal value of those fields.

## 4. Larger potential: market-baseline decomposition

This is a separate target-representation experiment, not a pure feature ablation. Every arm excludes the same first six
fitting months, although those known months can support later histories. Each row's market context uses earlier months only.

Estimate a same-type market baseline, then let the tree predict the property's residual:

```text
training target = actual rent - market baseline forecast from past information
final prediction = market baseline + tree-predicted residual
```

Market context includes the latest observed month, three-/twelve-month means, six-month log slope, forecast baseline and
forecast horizon. The trend variant clips monthly log slopes to ±0.02 and damps by 0.85 each month. These constants are fixed across
windows, without fold-specific adjustment. Forecasts use neither future actual rents nor future economic observations.

| Target representation | Pooled three-year RMSE | Latest annual RMSE | 16-month RMSE |
| --- | ---: | ---: | ---: |
| Original target; matched-row original 31-input control | 592.61 | 469.19 | 497.24 |
| Original target; 37-input control with market context | 589.74 | 469.57 | 496.86 |
| Latest-observed-month market baseline + residual | 583.01 | 467.55 | 488.54 |
| Damped-trend market baseline + residual | 563.29 | 468.15 | 490.79 |

Trend decomposition improves the 2022-04 score from 703.06 for the matched 37-input control to 621.15, but worsens the 2023-04
score from 583.98 to 593.24. It reduces underprediction during rising markets but may overpredict at turning points.
The latest-month baseline gives a smaller gain; its 2023-04 difference from the matched 31-input control is only about 0.36.

These results support further work separating market levels from relative property prices. Simply supplying time indices or future market levels
to a tree does not guarantee extrapolation. Decomposition transforms both targets and predictions and must be validated as a separate
modelling route. It has not changed the default feature processor.

## Reproduction and implementation

```powershell
# All ablations, market decompositions, compact-policy confirmations and structural audits
python -m analysis.main feature-optimize

# Or run one stage only
python -m analysis.main feature-optimize --stage features
python -m analysis.main feature-optimize --stage market
python -m analysis.main feature-optimize --stage refined
python -m analysis.main feature-optimize --stage audit
```

The shared implementation remains in `analysis/pipeline/features.py`, defaulting to
`RentalFeatureEngineer("compact_history_27")`. `HistoryPolicy` explicitly records numerical representation,
windows, half-lives, smoothing and output statistics, preventing implicit algorithm differences across experiments.
New tests cover same-month/future-label isolation for every history policy, fixed origins, decay weights and market-trend boundaries.

These results reuse previously examined months and support policy selection only; no new independent test labels are available.
Small differences have not been assessed across models or with month-resampling uncertainty. Future trend-parameter tuning or more complex selection
should occur inside fitting data, followed by one evaluation on fixed outer windows, rather than continued tuning around the latest year.
