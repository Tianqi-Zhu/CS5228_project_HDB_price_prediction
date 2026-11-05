# Concat Transformer: same-feature architecture exploration

This independent model supplements M05. It does not replace ensemble F1 or the pending F2 interface.

- Development:114842fit through2024-03,35158validation2024-04..2025-03.
- Selected:128width,6layers,4heads,FF2,dropout.05; validationRMSE459.2188909646949 at epoch8.
- Final:150000labels,from-scratch8epochs,50000test predictions. Public Score418.38657 from the user-provided screenshot; not private score or local-validation RMSE.
- Frozen original submissionSHA25694142ea6ad6adca53d252f6eb11b44aafc8cf8117a41846810d2500df11562a6.

## Use from repository root

```bash
# Integrity, IDs, finite values; does not train
python -m analysis.experiments.concat_transformer verify
# Replay final checkpoint on archived prepared test inputs (CPU or CUDA)
python -m analysis.experiments.concat_transformer predict --device cpu
# Rebuild exact final features using shared main pipeline and original raw data
python -m analysis.experiments.concat_transformer prepare --stage final
# Refit from scratch on all150000 labels, fixed8epochs; CUDA recommended
python -m analysis.experiments.concat_transformer refit --device cuda
# Rebuild development split, train selected architecture with validation early stopping
python -m analysis.experiments.concat_transformer prepare --stage selection
python -m analysis.experiments.concat_transformer select --candidate w128_d6 --device cuda
```

`prepare --python /path/to/python` optionally uses a separate feature environment.
The model runtime requires torch2.8.0,numpy2.1.1; feature preparation uses the shared project requirements including pandas,sklearn1.9.1,LightGBM and joblib. Exact archived preprocessing was fitted with sklearn1.9.1. Use existing training environment or install requirements/concat-transformer.txt in an appropriate environment; no auto-install/network in commands.

The small frozen bank permits final-model replay without sklearn or rebuilding auxiliary features. It contains150000train labels already supplied in data/train.csv and unlabeled50000test features, no test labels. Training modes use separately rebuilt banks in runtime/reports, not silently the archived replay bank. `artifacts/preprocessing.joblib` stores encoder state/source for audit/future raw preparation. Input vocab/quantiles must be regenerated for new data; the bank is not a generic rental API.

Training outputs/checkpoints go to ignored runtime/reports; feature caches go to shared ignored.cache/output. Final refit writes runtime/reports/concat_final_v22/refit_w128_d6/submission.csv. Model replay writes ignored output/concat_transformer. Original submitted CSV/weights are immutable. Models and frozen artifacts are force-tracked intentionally, unlike regenerable caches.

Prediction replay across CPU/GPU may vary in floating-point arithmetic; tolerance0.10SGD absolute+1e-4relative is an integrity check, not a new score claim. The exact historical submitted file remains artifacts/submission.csv. Complete development and final-refit provenance is in protocol/report JSONs. Do not treat test predictions or training loss as test RMSE. Longer test horizons remain extrapolation.
