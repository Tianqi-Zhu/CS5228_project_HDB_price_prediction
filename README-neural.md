# HDB Monthly Rental Prediction

Predict Singapore HDB monthly rents using a grouped-mean baseline and a ReLU neural network. The pipeline includes data-quality checks, outlier removal, chronological validation, exploratory plots, and submission-file generation. Original data under `data/` is never overwritten.

The recorded ReLU model achieves **MAE 334.26** and **RMSE 463.97** on the January–March 2025 historical holdout. These are validation results, not scores on the unlabeled test set.

## Project structure

```text
.
├── README-neural.md
├── environment-neural.yml                 # Portable Miniconda specification
├── environment-neural-resolved.yml        # Recorded Apple Silicon macOS package versions
├── data_quality.py                 # Data audit, normalization and IQR filtering
├── baseline.py                     # Grouped-mean baseline
├── train_relu.py                   # Neural training, evaluation and prediction
├── test_data_quality.py            # Cleaning tests
├── test_neural.py                  # Preprocessing and inference tests
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── example-submission.csv
│   └── auxiliary/                  # Available but not used by this model
├── outputs/
│   ├── data_quality/               # Generated audit and cleaned data
│   └── neural_network/             # Generated metrics, figures and predictions
└── templates/                      # Supplied report templates
```

## Data

Place the supplied project CSV files under `data/` if they are not included in your checkout.

| File | Rows | Period | Purpose |
|---|---:|---|---|
| `train.csv` | 150,000 | 2021-01 to 2025-03 | Housing attributes and observed `MONTHLY_RENT` |
| `test.csv` | 50,000 | 2025-03 to 2026-07 | Housing attributes requiring predictions |
| `example-submission.csv` | 50,000 | Matches test rows | Submission IDs and `Id,Predicted` schema |

The input columns describe rental approval month, town, block, street, flat type, flat model, floor area, furnishing status, lease commencement year, and fees. Only `train.csv` contains the target. Example submission values are not test labels.

Auxiliary datasets cover HDB blocks, MRT stations, schools, shopping malls, COE prices and stock prices. They are available for future feature engineering; the current neural model uses only the main housing table.

## Miniconda setup and run

Install Miniconda, then run the following from the directory containing `train_relu.py`:

```bash
conda env create -f environment-neural.yml
conda activate cs5228-hdb
python data_quality.py
python -m unittest -v test_data_quality.py test_neural.py
python train_relu.py
```

If `cs5228-hdb` already exists, activate it without creating it again. Generated output directories are created automatically. The training command performs epoch selection, historical evaluation, final refitting and prediction.

To train without activating the environment:

```bash
conda run --no-capture-output -n cs5228-hdb python train_relu.py
```

`environment-neural-resolved.yml` records the conda package versions used on Apple Silicon macOS and may not resolve unchanged on other platforms. `environment-neural.yml` is the simpler, portable environment specification. Training uses Miniconda.

After one successful training run, regenerate test predictions from the saved model without retraining:

```bash
python train_relu.py --predict-only
```

The saved model is ignored by Git, so a fresh checkout needs to train first. The prediction-only command loads `outputs/neural_network/relu_model.joblib` and overwrites the submission and prediction examples.

Run the dependency-free baseline with:

```bash
python baseline.py
python baseline.py --raw
```

The default uses shared cleaning and outlier removal. `--raw` reproduces the earlier baseline without that filtering. Both use the same chronological holdout.

## Cleaning and chronological evaluation

- Normalize categorical text and flat-type spelling. A lease year after the rental approval year becomes missing, rather than guessing a replacement.
- Remove training rows outside either the floor-area or rent 1.5-IQR fence. Both fences are estimated once, before removal, on the applicable fitting partition. Full-training fences are `[5, 173]` square metres and `[450, 4850]` rent; they remove 874 rows, leaving 149,126.
- Do not independently deduplicate: the data has no apartment/unit identifier.
- Preserve all validation and test rows with valid evaluation labels; do not remove difficult holdout examples using their rent. Test labels are unknown.
- Fill missing numeric features with training medians and standardize numeric features and the target using fitting data only. Unknown categorical values are accepted by the one-hot encoder.
- Exclude constant `FURNISHED` and `FEE`. The first neural model uses town, street, flat type/model, area, lease year, lease age, missing-lease indicator, elapsed month and cyclical month features. `BLOCK` alone is not a unique address, and auxiliary datasets are not used in this first model.

There are no original blank values or recognized missing markers in the supplied training data. Normalizing the contradictory lease year introduces one missing lease value. Lease age is approximated as rental approval year minus lease commencement year.

Full-training outlier counts are:

| Rule | Rows flagged |
|---|---:|
| Rent outside `[450, 4850]` | 752 |
| Area outside `[5, 173]` square metres | 130 |
| Both rules | 8 |
| **Unique rows removed** | **874** |
| **Rows retained** | **149,126** |

The IQR rule is `[Q1 - 1.5 × (Q3 - Q1), Q3 + 1.5 × (Q3 - Q1)]`. Remove a row if either feature is outside its range; endpoints are retained. Fences are calculated once per fitting partition, without repeated trimming. Outlier removal is a modeling choice, not proof that the source record is incorrect.

Three separate chronological stages prevent selecting epochs on the final holdout:

| Stage | Fitting data | Evaluation / prediction data | Purpose |
|---|---|---|---|
| Epoch selection | 2021-01 to 2024-09 | 2024-10 to 2024-12, no IQR deletion | Select epoch count with temporal early stopping |
| Final historical evaluation | 2021-01 to 2024-12 | 2025-01 to 2025-03, no IQR deletion | Compare ReLU and baseline with MAE, RMSE, R² |
| Submission model | All eligible training rows through 2025-03 | All 50,000 test rows, 2025-03 to 2026-07 | Refit with selected epoch count, export predictions |

The file split overlaps in March 2025; no day-level transaction information is available. Within the labeled data, temporal validation has no month overlap. No random train/validation split is used. Mini-batch shuffling occurs only within the fitting partition.

Historical evaluation uses **140,280 fitting rows after filtering** and **8,924 holdout rows**. The holdout is not used to select the epoch count.

## Models

### Grouped-mean baseline

Use the latest six observed fitting months to estimate average rent for each `(town, flat type)` group. Fall back to the flat-type average, then the overall average, for unseen combinations.

### ReLU neural network

Implemented with scikit-learn's `MLPRegressor`:

```text
Encoded inputs → Dense(128) + ReLU → Dense(64) + ReLU → Dense(1)
```

ReLU is `max(0, x)`. Hidden ReLU layers model nonlinear relationships. The output is linear for continuous regression, and predictions are converted back from standardized units.

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Squared error on standardized rent |
| Learning rate | 0.001 |
| Batch size | 512 |
| L2 regularization, `alpha` | 0.01 |
| Random seed | 42 |
| Epoch-selection cap | 100 |
| Early-stopping patience | 12 |
| Selected epochs in the recorded run | 2 |

The selection run evaluated 14 epochs and chose epoch 2 using October–December 2024 RMSE. Subsequent models were trained from scratch for that duration. More epochs did not improve the temporal tuning score in this run.

To change the training limits explicitly:

```bash
python train_relu.py --max-epochs 100 --patience 12
```

Training overwrites generated outputs. Avoid repeatedly choosing settings based on the final holdout.

## Recorded results

### Historical holdout: January–March 2025

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| Cleaned grouped-mean baseline | 356.02 | 491.27 | 0.4241 |
| **ReLU neural network** | **334.26** | **463.97** | **0.4863** |

The neural network reduces MAE by **6.1%** and RMSE by **5.6%** relative to the cleaned baseline on the same holdout.

- **MAE:** average absolute error in rent units.
- **RMSE:** error in the same units, with more weight on large mistakes.
- **R²:** performance relative to predicting the holdout mean; not classification accuracy.

| Holdout month | Records | MAE | RMSE |
|---|---:|---:|---:|
| 2025-01 | 2,870 | 339.99 | 477.24 |
| 2025-02 | 3,022 | 332.49 | 460.01 |
| 2025-03 | 3,032 | 330.59 | 455.08 |

See [evaluation plots](outputs/neural_network/model_evaluation.png) and [EDA overview](outputs/neural_network/eda_overview.png). Both are regenerated by training if absent from the checkout.

### Test predictions

The final model predicts all **50,000 test rows**. The recorded prediction mean is **3,123.53**, the median is **3,115.10**, and the range is **1,839.33–4,520.88**.

First five submission rows:

```csv
Id,Predicted
0,2630.52
1,3601.65
2,3200.46
3,3047.63
4,3351.44
```

Test rents are unknown. These values are predictions; historical validation metrics must not be described as test scores. Numerical differences are possible across platforms and library versions.

## Generated outputs

`outputs/neural_network/` contains:

- `report.txt`: Chinese summary and validation results.
- `submission.csv`: all test predictions, exactly `Id,Predicted`, in submission-template order.
- `prediction_examples.csv`: first ten predictions with input housing attributes.
- `metrics.json`: exact metrics, split sizes, cleaning counts, feature configuration, input hashes and library versions.
- `validation_predictions.csv` / `validation_by_month.csv`: historical errors with source-row references and monthly breakdown.
- `epoch_selection.csv`: temporal tuning history.
- `eda_overview.png` / `model_evaluation.png`: source-data overview and neural evaluation figures.
- `relu_model.joblib`: final fitted preprocessing, target scaling and neural weights. Load only model files you trust.

`outputs/data_quality/` contains the normalized, IQR-filtered training CSV and an audit report. The training script rebuilds each chronological partition from the raw data, so full-data preprocessing never leaks into earlier evaluation fits.

## Interpretation and limitations

- **Errors are concentrated at price extremes.** On the holdout, 79.5% of predictions are within 500 rent units of the actual value, but low rents tend to be overestimated and high rents underestimated.
- **Outlier removal may exclude legitimate properties.** Its effect on the neural model still requires a controlled retain-versus-remove comparison.
- **Location is represented coarsely.** Town and street are used, but block coordinates, transport distances and nearby amenities have not yet been incorporated.
- **The tested forecast horizon is limited.** Three months of historical evaluation cannot establish accuracy through July 2026. Additional rolling temporal splits and longer-horizon validation are needed.
- **The data is observational.** Associations between features and rent do not establish causal effects.

Next experiments can add auxiliary location features, compare outlier policies, and evaluate additional chronological windows.

## Reproducibility and repository contents

The recorded run used Python **3.12.14**, NumPy **2.5.3**, pandas **3.0.6**, scikit-learn **1.9.1**, and joblib **1.6.0** in Miniconda environment `cs5228-hdb`. Full package versions are recorded in `environment-neural-resolved.yml`; model settings and source CSV hashes are stored in `metrics.json`.

Twelve tests cover cleaning boundaries, missing and invalid values, contradictory lease dates, outlier filtering, target exclusion from inputs, training-only preprocessing, and unseen-category inference. The submission was also checked for 50,000 correctly ordered IDs, finite positive predictions, and identical output after loading the saved model.

`.gitignore` excludes local virtual environments, caches, Python bytecode, `.DS_Store`, and the saved neural model. Include source code, environment specifications, tests, this README, and selected result summaries/figures in the repository. The pipeline recreates the model and predictions once the required CSV inputs are available.
