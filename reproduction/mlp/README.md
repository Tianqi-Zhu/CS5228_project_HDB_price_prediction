[Chinese version](README.zh-CN.md)

# MLP ensembles, full training and time weighting

Integrates `mlp-eda-ensemble` runtime, configuration, full-fit submissions and time-weighting experiments. Main's F1, Concat and TabPFN retain their implementations. MLP loads `runtime/analysis` in an independent process and uses the shared root `data/`.

| Existing MLP submission | Training rows | Kaggle screenshot score |
|---|---:|---:|
| [Full-fit uncleaned](artifacts/fullfit/submission_mlp_fullfit_uncleaned.csv) | 150,000 | 439.38193 |
| [Full-fit obvious-low-rent cleaning](artifacts/fullfit/submission_mlp_fullfit_cleaned.csv) | 147,983 | **437.34455** |

Scores come from original screenshots with public/private split unknown; this integration made no new Kaggle uploads. CSVs and [screenshot/score mapping](artifacts/fullfit/kaggle_results.json) retain original bytes. Full fitting uses known labels through 2025-03, three seeds and three epochs each to predict 50,000 rows. Interval models use an earlier cutoff; their old intervals cannot be applied directly to full-fit models.

## Run from the main repository root

```bash
# Verify files, raw data and four submissions without importing training dependencies
python -m analysis.main mlp verify
# MLP unit tests without full-data training
python -m analysis.main mlp test
# Retrain original full-fit MLPs, save/reload and generate two 50,000-row CSVs
python -m analysis.main mlp fullfit --output-dir output/mlp/fullfit-new
# Optional time-weighted candidate with a 36-month half-life
python -m analysis.main mlp fullfit --half-life-months 36 --output-dir output/mlp/weighted-new
# Select weights on two early windows, evaluate a later year, then refit all data
python -m analysis.main mlp recency-search --output-dir output/mlp/recency-new
# Original separate-calibration workflow for MLP/TabM intervals and test predictions
python -m analysis.main mlp intervals --output-dir output/mlp/intervals-new --predict-test
# Repeat both cleaning controls on the newly generated baseline
python -m analysis.main mlp cleaning --baseline-dir output/mlp/intervals-new --output-dir output/mlp/cleaning-new
```

Alternatively use `python reproduction/mlp/run.py ...`. Output directories must be new; new weights, preprocessing and results stay in ignored `output/`. Training requires CPU PyTorch and [MLP dependencies](../../requirements/mlp.txt), without CUDA. Tested versions are Python3.12/PyTorch2.14.1; use a separate environment to preserve F1's CUDA environment. On macOS without OpenMP, prefix commands with `DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib`; optionally set `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hdb-mpl`.

## Time-weighting result boundaries

Weights are `2 ** (-age_in_months / half_life)`, normalized to mean 1 and applied to regression and auxiliary-classification losses. Among 6/12/24/36-month candidates, 36 performs best; uniform weighting still wins on pooled early validation. On the same later annual validation, RMSE changes from 470.2343 to 469.7138 and MAE from 341.4548 to 341.4755, showing no stable advantage. [Weighting records and CSVs](artifacts/recency/) remain controls with no Kaggle scores.

MLP uses property/spatial fields, address embeddings, market-adjusted history and a residual target: 44 pre-encoding fields, differing from heterogeneous F1 and Concat inputs. This integration claims no same-input architecture ranking and does not add MLP to F1's original weights. Annual validation has been repeatedly used for development. The 380 test rows in 2025-03 overlap a training month; history excludes same-month labels, while full supervised fitting includes that month's known labels.

[Prediction report](../../docs/mlp_prediction_report.md) · [EDA](../../docs/mlp_eda.md) · [Provenance and path adaptations](provenance.json)

Source-branch and main raw CSVs differ only in LF/CRLF, with identical field values and row order. The [data identity record](data_identity.json) stores both byte hashes and normalized-LF hashes; verification accepts line-ending differences only and does not rewrite main data. Tests use one Torch thread to avoid local OpenMP conflicts; full training retains the original two threads.
