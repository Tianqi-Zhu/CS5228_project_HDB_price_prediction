# What the current TabPFN predictions still leave unresolved

[Open the interactive dashboard](dashboard.html). It compares the saved
**one-estimator TabPFN-3.5 Plus validation forecast** with local frozen Fast B
and the A–E equal ensemble on the same 28,995 labeled rentals. It also shows
forecast-horizon coverage of the active **eight-estimator, 150,000-context**
Plus submission. The submission's test labels are unavailable, and no
validation run used its exact eight-estimator configuration. Its RMSE and
residual patterns therefore cannot be measured from the submission CSV.

## Matched labeled comparison

| Forecast | June 2024–March 2025 RMSE (S$) | Mean prediction error (S$) |
| --- | ---: | ---: |
| Plus API, one estimator, original 77 features | 456.144 | +16.836 |
| Local frozen Fast B, two estimators, original 77 features | 455.396 | +10.792 |
| Local A–E equal ensemble | **454.704** | +0.988 |

The Plus and A–E prediction errors correlate **0.995**. More capable hosted
weights alone did not remove most errors on this one-estimator comparison.
The models also differ in estimator count, preprocessing, adaptation and/or
features, so the observed score differences cannot be assigned solely to the
pretrained weights. A retrospective best positive blend would use about 21%
Plus and lower this *same* development-window RMSE to S$454.594, only S$0.110
below A–E. That weight uses the evaluated labels and is **not** a validated
blend or submission recipe.

## Residual structure still visible

* **Sparse rental history:** Plus overpredicts by about **S$44.8** when the
  earlier block/flat-type count is 1–3, versus **S$24.1** for A–E. For count
  above 30, Plus underpredicts by about **S$16.9**. These are descriptive
  subgroup averages; sparse support also correlates with neighborhood and
  property composition.
* **Geography:** Plus overpredicts in all ten evaluated months in Bukit
  Panjang, Choa Chu Kang, Hougang, Punggol, Sembawang and Sengkang, among
  town-month cells with at least 30 rentals. A–E retains the same sign in
  Choa Chu Kang, Punggol, Sembawang and Sengkang. For Punggol, Plus's mean
  monthly error is about +S$93 and A–E's about +S$70.
* **Local clustering:** Block-mean residual Moran's I exceeds the
  support-stratified permutation upper reference in July and November 2024
  for both Plus and A–E. The reference is descriptive, not a multiplicity-
  adjusted significance test or proof of a missing causal spatial mechanism.
* **Feature-bin calibration:** The dashboard plots error against floor area,
  flat age, MRT distance and earlier block/flat-type support. Plus tends to
  overpredict younger flats and intermediate floor areas in this window.
  Binned residuals are associations with correlated features, not direct
  feature effects or an interpretation of individual TabPFN parameters.

### Could better temporal features fix the six-town pattern?

Possibly, especially **town-specific rental levels and changes**. The six
towns that overpredict in every month under the one-estimator Plus validation
have mean monthly biases of +S$46.6 to +S$92.8; within-town month-to-month
standard deviations are S$20.0 to S$33.6. The persistent level is therefore
substantial alongside a changing component. A national COE or stock series is
identical across towns at a fixed origin; it would need a reliable, learned
town-specific interaction to address this pattern.

As a posthoc descriptive check on these six towns, mean town bias is about
+S$65.8 for frozen B, +S$48.4 for relative-history C, +S$50.5 for seasonal D,
+S$48.7 for nearby-rent E, +S$34.4 for adapted A, and +S$49.6 for their equal
ensemble. Those towns were chosen after observing Plus errors, so this is
**not** an unbiased subgroup selection result. It shows that representation
and adaptation move the location biases, but all six views retain a positive
average there and several added-feature variants worsen overall RMSE. The
[town-by-view values](persistent_town_view_bias.csv) are saved for inspection.

Prior matched tests offer a strong caution against simply appending more
columns: seasonal D is S$0.408 worse than B; nearby-rent E is S$1.086 worse;
the 11-field town/street/building rental-dynamics addition raises an existing
frozen ensemble's later RMSE by S$1.074; and frozen TabPFN additions of
national rent forecasts or COE increments also worsen later RMSE. Earlier
windows sometimes improved, so the problem includes unstable temporal
transfer. These results test particular representations and cutoffs, not all
possible local temporal models. See the [A–E comparison](../../tabpfn_abcde_ensemble/report.md),
[targeted local-dynamics test](../../targeted_spatial_economic/report.md),
and [forecast/COE ablation](../../tabpfn_reliability_macro_80_20/report.md).

A targeted test would first decompose residuals into a shrunken town/flat-type
**level** and a town/flat-type **change with horizon**, using only labels
available before each origin. Compare (1) the unchanged TabPFN forecast,
(2) a static local-level correction, and (3) a support-gated local-level plus
local-trend correction on the *same* rolling origins. This isolates whether
time variation adds value beyond a stable geographic offset. Judge both
individual-rent RMSE and the six-town/month and nearby-block diagnostics.
National macro interactions should be a later, separate ablation because the
historical panel has few independent monthly paths.

At the single June 2024 forecast origin, target month and horizon move
together, so monthly bias cannot separate market movement, seasonality and
longer-horizon degradation. The ten monthly periods supply far less
independent temporal evidence than the 28,995 transaction count suggests.

## The active submission's untested range

The active Plus submission has **18,077 of 50,000 rows (36.154%) at horizons
11–16**. The labeled local validation covers horizons 1–10 only. The inherited
local temporal baseline plateaus after horizon 10, while horizon and
observation-age fields continue to change. The test has no rent labels, so
the eight-estimator Plus model's accuracy or bias in months 11–16 is unknown.
This is the clearest coverage gap for the current CSV.

The active Plus submission and the existing full-data ABCDE equal submission
are aligned by all 50,000 IDs. Their **mean absolute disagreement is S$20.17
at horizons 1–10 and S$33.24 at horizons 11–16**. Plus is on average S$44.2
lower at horizon 15. The widening difference pinpoints where model choice
matters most; it does **not** reveal which forecast is closer to the unknown
rent. [test_horizon_coverage.csv](test_horizon_coverage.csv) and
[test_town_disagreement.csv](test_town_disagreement.csv) give the full
unlabeled comparisons.

## What this does and does not say about TabPFN weights

These diagnostics identify forecast errors conditional on the supplied
features and context. They do not identify a deficiency in one weight tensor:
feature construction, incomplete point-in-time geography, scarce comparable
rentals, target noise, market shift and the fixed-origin forecasting policy
can all produce the patterns. This pipeline has no explicit monthly
local-rent sequence or geographic-distance attention mechanism. The
nearby-rent view E did not improve standalone main RMSE in the earlier A–E
study, so merely adding similar summary columns is not yet supported.

The next decisive check is an **11–16-month chronological backtest** using a
matched forecast policy, followed by several distinct origins. A second test
could compare grouped, origin-safe local-history perturbations or improved
comparable retrieval against the same TabPFN context. A validation run of the
exact eight-estimator Plus configuration would be required to attach a local
score to the active submission architecture; a one-estimator score cannot be
reused for it.

## Reproduction

Run `.venv/Scripts/python.exe analysis/analyze_tabpfn_remaining_gaps.py`.
The analysis aligns row IDs, months and labels exactly across the saved
forecasts, hashes all inputs, and writes [verification.json](verification.json),
[grouped_errors.csv](grouped_errors.csv), [town_month_bias.csv](town_month_bias.csv),
[spatial_autocorrelation.csv](spatial_autocorrelation.csv),
[feature_error_bins.csv](feature_error_bins.csv), and
[test_horizon_coverage.csv](test_horizon_coverage.csv). It makes no API calls
and changes no submission.
