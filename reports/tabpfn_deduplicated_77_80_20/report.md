# Deduplicated rental records: frozen TabPFN77-feature experiment

The fully rebuilt deduplicated-history experiment changes RMSE by **S$+0.028** against the original frozen77-feature baseline on the identical28,995 validation observations.

| population | model | n | rmse_sgd | mae_sgd | bias_sgd | delta_rmse_sgd | rmse_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_complete_validation | original_77 | 28995 | 455.814 | 329.748 | 0.768 | 0.000 | 0.000 |
| original_complete_validation | deduplicated_rebuilt_77 | 28995 | 455.841 | 330.178 | -2.134 | 0.028 | -0.006 |

## Duplicate rule and saved data

This is a duplicate-row removal experiment, with all77 model columns retained. Case/whitespace in town/block/street and existing flat-type/model aliases are canonicalized. A duplicate must match every original record attribute, including approval month, floor area, lease year, rent, furnished status and fee. Keep the first original row ID; different prices or dates are not collapsed. Source data lacks unit/tenancy identifiers, so this is a sensitivity policy, not proof that the removed rows were erroneous contracts.

The complete labelled file changes from150,000 to149,359 rows (641 removed). Within the original January2021-May2024 fitting period,244 rows are removed:121,005 becomes120,761. The June2024-March2025 validation population remains all28,995 rows for a directly comparable primary score. No random repartition or boundary shift is performed. A secondary table filters both models to the same28,598 unique validation rows without using it to select a model.

Processed records and model-ready CSVs are saved under `data/processed/tabpfn_deduplicated_77_80_20/`:

- train_canonical_deduplicated.csv:149,359 normalized records, all original11 fields plus stable source row_id.
- removed_duplicate_mapping.csv:641 removed IDs and their retained counterpart.
- train_features_77.csv/query_features_77.csv:77 model columns plus metadata row_id.
- train_targets.csv/query_targets.csv:aligned raw SGD rent, residual target, month and additive baseline.
- schema.json:feature list, categorical columns, loading and reconstruction policy.

## Feature rebuilding and inference

Retained context rows keep their original deterministic historical forecast origins. Every37 target-history input and9 temporal input is recalculated using only deduplicated fitting records available by that origin. The global/local temporal model is refitted and its existing earlier-only tuning rule reapplied. Validation history is frozen at May2024; no validation outcomes enter history construction or inference. Warmup rows without earlier data retain the original empty-history representation and zero additive baseline. Simply dropping TabPFN context rows while leaving duplicate-influenced histories was deliberately avoided.

The other31 inputs, categorical dtypes, original spatial source, month encoding and every feature name remain unchanged. In particular, the known future-MRT source issue is preserved in both arms to isolate this duplicate policy. This is frozen pretrained TabPFN-3.5-Fast inference, one estimator, seed42,200-row batches, every retained context row and native preprocessing. No DiagScale parameter optimization, CatBoost correction or new hyperparameter search is performed. Fit target is raw rent minus the recomputed temporal baseline; add the recomputed query baseline once.

## Unique-validation sensitivity

| population | model | n | rmse_sgd | mae_sgd | bias_sgd | delta_rmse_sgd | rmse_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unique_validation_sensitivity | original_77 | 28598 | 457.736 | 331.205 | 2.394 | 0.000 | 0.000 |
| unique_validation_sensitivity | deduplicated_rebuilt_77 | 28598 | 457.750 | 331.616 | -0.495 | 0.014 | -0.003 |

## Uncertainty and limits

Paired month-cluster bootstrap uses5000 resamples of the ten validation months. Candidate-minus-control RMSE interval: [-0.088, +0.150] SGD. This is descriptive development-period uncertainty; the holdout has been repeatedly inspected and is not a fresh competition-test estimate.

This primary ablation combines removing duplicate context rows with removing their contribution to histories and temporal forecasts. Feature changes are recorded in feature_changes.csv. It does not isolate those two mechanisms separately. The original MSE-only DiagScale checkpoint and Kaggle CSV are unchanged; this result should not be presented as a refitted MSE-only DiagScale result.

## Reproduction

Run `.venv/Scripts/python.exe -m unittest tests.test_canonical_rental_dedup tests.test_aligned_temporal_features`, then `.venv/Scripts/python.exe analysis/compare_tabpfn_deduplicated_77_80_20.py --stage all`. Prepared features and resumable inference batches live under ignored .cache/. No current Kaggle submission or raw source file is overwritten.
