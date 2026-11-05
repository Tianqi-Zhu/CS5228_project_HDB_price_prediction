# TabPFN reliability and forecast-derived COE ablation

Locked before new scoring. This remains development evidence: the June
2024–March 2025 period and aggregate macro experiments have already been inspected.

1. Keep the retained 77-column categorical-month matrices, all context rows,
   labels, pseudo-origins and original additive temporal baselines unchanged.
   Plain V3.5-Fast, seed 42, two estimators, native estimator batching and
   200-row batches isolate the effect of inputs. Reuse the hash-verified plain
   control from the existing feature-diversity experiment, with identical IDs.
2. Nested input sets: control77; reliability82; rent_forecast84; coe_increment85.
   The five reliability indicators describe absent signature/block histories,
   block 3-month count below 2, block 12-month count below 5, and unknown BLOCK_ID.
   Add national rent-only predicted log growth plus an unavailable flag, then
   one COE incremental predicted-log-growth feature. No raw stock inputs.
3. Matched rent-only/rent+COE LSTMs both use 12-month input windows, hidden size
   8, seed42 and 100 fixed epochs from the prior sequence experiment. No new
   hyperparameter search. Target is the existing fixed-2021-mix-adjusted national
   arithmetic mean. The 2021 reference adjustment is only used after 2021 ends.
   Each forecast is refitted from observations through its cutoff; labels with
   later endpoints are masked. A head needs 3 mature monthly labels or returns
   native NaN with an explicit unavailable flag. Overlapping examples are not
   independent market observations.
4. Each row's macro cutoff is its original feature origin minus one month,
   conservatively buffering publication timing for both rent and COE. No realized
   future macro path is supplied. Direct heads forecast 1–10 months; macro growth
   is held at head10 when this extra buffer creates an eleven-month distance.
   Actual calendar/horizon and the original residual baseline remain unchanged.
5. Evaluate existing fixed-origin folds 2022-09 and 2023-07 (2,000 label-independent,
   population-weighted query rows each), then 2024-06 (all 28,995 rows). All context
   rows remain. Lock the minimum pooled earlier-MSE input set before later scoring;
   baseline participates and wins exact ties. Report every variant and adjacent
   input-set comparisons, not just the selected candidate.
6. If an added-feature input set wins earlier selection, also evaluate that input
   set against the numerical-best MSE-only DiagScale control on the later fold.
   Keep its existing pre-December-2023 adapter fixed, with one inference view;
   this tests feature transfer, not retraining the adapter on augmented inputs.
   Never apply that adapter to earlier folds preceding its training period.
7. Separately evaluate the LOCAL TEMPORAL BASELINE itself through 16 months:
   month10 plateau, extension of the existing damped trend/local-premium decay,
   and persistence of the latest shrunken local level. Origins 2022-09, 2023-01,
   2023-07 and 2023-12; retain monthly area/type cell rental counts as weights.
   Select only on target months before June2024; report June2024–March2025 and
   1–10 versus11–16-month bands separately. These are cell-mean forecasts,
   not a newly trained TabPFN residual model or measured test-set performance.
8. Add future-data poisoning, flag semantics and reconstruction checks. Record
   source/code hashes, feature coverage, weights, row IDs, costs and cutoff audits.
   Paired month-block intervals are conditional and omit model-selection fitting
   uncertainty. Existing datasets, reports and final submissions remain unchanged.
