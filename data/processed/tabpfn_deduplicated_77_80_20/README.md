# Deduplicated rental-data experiment

`train_canonical_deduplicated.csv` contains 149,359 labelled records: the original 150,000 rows minus 641 canonical duplicates. All 11 source fields are retained, with existing flat-type, model and address normalization. `row_id` identifies the original source row and is metadata, not a predictor.

Duplicates match every observed field, including approval month and rent. The first source occurrence is retained. This experimental policy cannot establish physical-unit identity because unit and tenancy IDs are unavailable. Removed and retained IDs are paired in `removed_duplicate_mapping.csv`.

The benchmark keeps the original June 2024 split:

- `train_features_77.csv`: 120,761 context rows, 77 predictors plus `row_id`.
- `query_features_77.csv`: all 28,995 original validation rows, 77 predictors plus `row_id`.
- `train_targets.csv` and `query_targets.csv`: aligned raw rent, month, additive temporal baseline and residual target.

All historical rent inputs and temporal baselines are rebuilt from deduplicated permitted history. The other 31 features remain unchanged. Validation duplicates remain in the primary comparison; the report also evaluates both models on the same 28,598 unique validation rows.

Load numeric CSV values with `pd.read_csv(path, index_col="row_id", float_precision="round_trip")`. Restore categorical dtypes using `categorical_dtypes.json`; monthly categories are integers. Preserve the source IDs and missing values. Fit on `residual_target_sgd` and add the query `additive_baseline_sgd` once to reconstruct SGD rent.

The lossless typed artifact is `.cache/tabpfn_deduplicated_77_80_20/features.joblib` relative to the repository root. The experiment runner is `analysis/compare_tabpfn_deduplicated_77_80_20.py`; results are under `reports/tabpfn_deduplicated_77_80_20/`.
