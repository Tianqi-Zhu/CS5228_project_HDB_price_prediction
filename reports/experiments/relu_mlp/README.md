# ReLU MLP

`historical_results.json` preserves the recorded source-branch experiment: RMSE 463.97 on the
January–March 2025 holdout after nested epoch selection. This three-month result is not directly
comparable with the common April 2024–March 2025 twelve-month validation.

Running `python -m analysis.main relu` writes a fresh `results.json`, `epoch_selection.csv` and
`monthly_errors.csv` here. Per-row holdout predictions, model binaries and submissions are ignored.

The verified `cs5228` reproduction selected the same two epochs and reached RMSE 463.78. Its
0.20-SGD difference from the recorded branch result is consistent with the different Python,
NumPy, pandas and scikit-learn patch versions; both values are retained in
`historical_results.json`.
