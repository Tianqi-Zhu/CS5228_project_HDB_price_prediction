# Spatial, temporal and feature diagnostics for TabPFN A–E

[Open the interactive dashboard](dashboard.html). It contains monthly error
curves, town-by-month bias maps, a nearby-block residual check, variant
contrasts, and observed-versus-predicted feature associations. All figures use
saved forecasts; the script performs no TabPFN inference or model refit.

## What A–E encode

| View | Spatial information | Temporal information |
| --- | --- | --- |
| A | Baseline coordinates, town, planning area, building identity, amenities and rent histories | Baseline calendar, historical rents, forecast horizon and local temporal baseline; one-epoch DiagScale adaptation |
| B | Same baseline fields | Same baseline fields; frozen TabPFN-3.5-Fast |
| C | Same baseline fields | Re-expresses 19 absolute rent-history/forecast levels relative to the per-row baseline |
| D | Same baseline fields | Replaces categorical month with sine/cosine and adds six school/university seasonal proxies |
| E | Baseline fields plus 12 nearby-rent, support and local comparable summaries | Nearby-rent summaries use earlier labels; otherwise baseline time fields |

All five receive labeled earlier rental rows as TabPFN context and predict a
residual to the same local temporal baseline. Their tabular attention can use
interactions among columns and context rows, but none of these five variants
contains an explicit geographic-distance attention bias or a dedicated monthly
sequence encoder. E's geographic proximity is supplied as engineered columns.
The nearby-rent fields have an additional publication lag and can contain
earlier labels from the same building; they exclude the query and later labels.
The inherited static MRT/amenity snapshot is not fully point-in-time audited.

## What the saved forecasts show

| Model | Main development RMSE (S$) | Difference from B (S$) |
| --- | ---: | ---: |
| A, adapted | 455.071 | −0.325 |
| B, frozen baseline | 455.396 | 0 |
| C, relative history | 455.370 | −0.026 |
| D, seasonal view | 455.804 | +0.408 |
| E, nearby-comparable view | 456.482 | +1.086 |
| ABCDE equal | **454.704** | −0.692 |

The equal ensemble is the lowest *observed* RMSE on the repeatedly examined
June 2024–March 2025 window. The original earlier-transfer selection did not
promote it as an independently validated winner. D and E being worse alone
does not establish that seasonal or nearby-rent information is useless: both
change a representation, and E still contributes different errors in an
average. C improves B by only S$0.026 on this window. The historical windows
show different rankings, so the individual differences are not stable across
origins.

The town × month map shows that the ensemble still overpredicts all ten months
in **Punggol, Choa Chu Kang, Sembawang and Sengkang** (among cells with at
least 30 rentals). For example, Punggol's mean monthly bias falls from about
+S$92 for B to +S$70 for the equal ensemble, yet its sign remains positive.
After removing fitted additive town and month biases, the support-weighted RMS
of town × month cell errors is **S$43.30 for B, S$43.51 for E and S$43.33 for
the equal ensemble** across the 243 cells with at least 30 rentals. This
diagnostic does not show a material reduction in the *interaction* error from
the nearby-rent view or ensemble. The fitted additive terms and interaction
summary use the same development labels, so they describe residual structure
rather than a new validation score.
Block-mean residual Moran's I exceeds a support-stratified permutation upper
reference in **July and November 2024** for B, E and the ensemble (2 of 10
months each). This is a descriptive indication of remaining local structure,
not a causal or multiplicity-adjusted significance test. The map and the
Moran check do not by themselves identify an architectural defect.

The feature plots center both actual and predicted rent within each town,
flat-type and month before making eight marginal quantile bins. This reduces
some compositional differences while leaving other correlated property traits
uncontrolled. From the first to last bin, actual-versus-ensemble centered-rent
changes are approximately **−S$253 versus −S$244** for flat age, **−S$139
versus −S$136** for MRT distance, and **+S$35 versus +S$23** for floor area.
After town/month/type centering, the first-to-last CBD-distance contrast is
only about −S$5 observed and −S$7 predicted. That small conditional contrast
must not be read as a causal CBD-distance effect. Nearby support and block
history counts are also plotted in the dashboard.

## Interpretation limits and next test

These plots compare **observed associations** with **forecast associations**.
They do not show how rent would change if one feature were intervened on.
History, geography and the baseline are strongly correlated; moving one field
while keeping the others fixed can create impossible properties. TabPFN
attention weights likewise do not provide causal feature effects. At this
fixed origin, forecast horizon and target month coincide, so the monthly curves
cannot separate seasonal, market and horizon effects. There are ten forecast
months, not 28,995 independent temporal paths.

The next model-reliance diagnostic would use grouped, geographically and
temporally coherent perturbations on a fixed, label-independent query sample,
with the original TabPFN context frozen and the external baseline recomputed
when its inputs change. Compare B, C, D and E on several forecast origins.
Matched ablations and new chronological holdout data are needed before saying
the spatial–temporal relationship is adequately learned.

## Reproduction

Run `.venv/Scripts/python.exe analysis/interpret_tabpfn_abcde.py`. The script
checks query ID, label and month alignment; checks the ensemble mean; verifies
the cached feature hashes; and writes [verification.json](verification.json),
[monthly_scores.csv](monthly_scores.csv), [town_month_bias.csv](town_month_bias.csv),
[town_month_interaction.csv](town_month_interaction.csv),
[spatial_autocorrelation.csv](spatial_autocorrelation.csv),
[feature_associations.csv](feature_associations.csv), and
[variant_contrasts.csv](variant_contrasts.csv). No original forecasts or
submissions are modified.
