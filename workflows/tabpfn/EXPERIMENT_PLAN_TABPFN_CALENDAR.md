# Controlled calendar feature experiments

This plan is recorded before new outer scores are available. The fixed control
is the aligned TabPFN-3.5-Fast residual model, S$455.8629 RMSE on the existing
June 2024-March 2025 development holdout.

## Fixed components

- Context: all 121,005 January 2021-May 2024 transactions.
- Validation: all 28,995 June 2024-March 2025 transactions.
- Identical source row order, raw labels, residual baselines, rent histories,
  continuous elapsed-month index, and original categorical property inputs.
- Same Fast checkpoint, seed 42, one estimator and 200-row inference batches.
- Train on `y_train - train_baseline`; add the query baseline exactly once.
- Replay the first 400 control predictions with full context before reuse.

## Variants

1. Integer month: reuse the verified aligned-residual control.
2. Categorical month: change only month-of-year's pandas dtype, with all twelve
   categories consistently defined in context and query.
3. Cyclical month: replace only month-of-year with sine/cosine; preserve the
   continuous month index.
4. Calendar main effects: use the encoding selected on earlier folds; add six
   recurring monthly opening/lead-window proxies.
5. School interactions: add primary-school proximity/count interactions to
   variant 4. Existing flat type remains available to TabPFN for interactions.

The primary-school annual opening proxy is January, with December/November
one-/two-month leads. The university opening proxies are January and August,
with December/July and November/June leads. These are calendar hypotheses,
not exact term dates, actual tenant identities, holiday-day fractions, or
historical university-distance measurements. They are known from the target
month alone and require no future calendar announcement or rental outcome.

All added features must be finite for every context and query row, including
January 2021. The previously observed 31.1% reference-model warmup mismatch
is avoided by introducing no new fitted property-reference model. Existing
history warmup missingness remains identical to the control. If source school
location is missing, interaction inputs use zero access/count and an explicit
missing-location flag; current source school distances/counts are complete.

## Earlier selection and final confirmation

Use two disjoint ten-month query windows: September 2022-June 2023 and July
2023-April 2024. Every row before each origin remains in context. Query rent
histories and baselines are regenerated at the corresponding fixed origin.
To control GPU cost, earlier validation samples 200 source rows per month with
seed 42 independently of labels. Scores use month population/sample weights.
Only earlier query evaluation is sampled; context and final validation are full.

Select the base encoding on earlier pooled RMSE, preferring integer, then
categorical, then cyclical when within 0.1% of the best. Then evaluate the two
calendar variants on those same earlier folds. A proposed replacement must
improve earlier pooled RMSE by at least 0.25%, with neither earlier fold more
than 0.5% worse. Simpler variants win within 0.1% of the best eligible score.

Record that selection before new outer scoring. Run all four outer candidates
for a complete matched comparison. Retain the earlier-selected candidate only
if its outer RMSE improves on the control; do not choose a different candidate
by inspecting the outer ranking. The outer window has already been examined
and remains a development set. No 17-month generalization claim is made.

## Artifacts and checks

Run `.venv/Scripts/python.exe analysis/compare_tabpfn_calendar_80_20.py`.
The runner resumes completed jobs only with matching frozen settings. Metrics,
timings, selections, reports and availability audits go to
`reports/tabpfn_v3_5_fast_calendar_80_20/`. Full feature bundles go to ignored
`.cache/tabpfn_v3_features/calendar_80_20_<variant>.joblib`; row predictions are
also ignored. Unit tests cover feature completeness, category consistency,
annual wraparound, label/future-row independence and sampling weights.
