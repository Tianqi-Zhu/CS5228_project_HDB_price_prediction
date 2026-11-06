[Chinese version](midterm_report.zh-CN.md)

# CS5228 HDB monthly rent prediction: midterm experiment report

> This document retains midterm-stage definitions. For current feature increments, shared-input comparisons, error diagnostics and completion status, see
> the [current experiment report](../../experiment/02_feature_engineering/BASELINE_27_FEATURES.md); do not directly rank scores across stages.

## Research goal and current scope

Predict monthly rent (SGD) from Singapore HDB property attributes, addresses and rental approval time, evaluated by root mean squared error (RMSE). This report records completed EDA of the main and six auxiliary datasets and first-round traditional machine-learning experiments **without auxiliary features**. Scores are internal temporal-validation results; no Kaggle test scores are available yet.

Code is unified under [`analysis/`](../../analysis), separated into data, models, training, evaluation, EDA and experiments. Commands: [README](../../README.md); module roles: [project structure](../guides/project_structure.md).

The updated repository also retains [integrated EDA](../../experiment/eda/report.md) and a 15-page PDF covering drift, facility associations and recent-median baselines. Its statistical baselines use different windows from the four models here, so scores are not directly rankable. The unified pipeline supports full refitting and `Id,Predicted` generation; migration-check predictions test row count, order and model persistence, not new effectiveness or Kaggle scores.

## Main-data EDA (train/test)

`analysis/eda/audits/main_data.py` reads only `data/train.csv` and `data/test.csv`, without modifying them. It generates a [data-check/split summary](../../experiment/eda/supplementary/train_test/summary.json) and five presentation-ready figures:

1. [Monthly rent distribution](../../experiment/eda/supplementary/train_test/01_rent_distribution.png)
2. [Monthly median rents and validation period](../../experiment/eda/supplementary/train_test/02_monthly_median.png)
3. [Rent by flat type](../../experiment/eda/supplementary/train_test/03_flat_type.png)
4. [Rent by Town](../../experiment/eda/supplementary/train_test/04_town_median.png)
5. [Floor area and rent](../../experiment/eda/supplementary/train_test/05_floor_area.png)

### Initial findings

- Training contains 150,000 records from 2021-01 to 2025-03; test contains 50,000 from 2025-03 to 2026-07. Of those, 49,620 are later than the final training month.
- Neither file has missing values. Training rents range from S$300 to S$7,600 with median S$2,700; areas span 31–215 square meters. Rents and areas are all positive.
- `FURNISHED` is always `yes` and `FEE` always 0 in both files, so they cannot explain rent differences within these data.
- `FLAT_TYPE` includes variants such as `4 room` and `4-room`. Labels use spaces in 2021 and mostly hyphens afterward; normalize before comparisons or training.
- Training contains 425 exact duplicate records and one lease commencement year later than approval year. Without rental agreement IDs, identical records may represent different approvals; currently they are flagged rather than deleted or modified.
- Without controlling flat type, area and location, monthly median rent rises from S$2,000 in 2021-01 to S$3,200 in 2025-03. This alone does not measure appreciation for an individual flat.

### Temporal-validation design

Train on **2021-01 to 2024-03** approvals (114,842 records) and validate on **2024-04 to 2025-03** (35,158). No approval month appears on both sides. Score with RMSE, fit encoding/imputation only within training, and exclude Kaggle test data from model selection.

Almost all Kaggle test records follow the training period. Random splits can use later approvals to predict earlier ones, unlike the actual task. However, the 12-month validation period is shorter than the roughly 16-month future test horizon and may still underestimate long-horizon error. Consider longer windows or multiple temporal origins.

These figures describe training-data associations. Without other controls, differences across months, flat types or Towns do not isolate independent factor effects.

## Auxiliary-data EDA (auxiliary_data)

`analysis/eda/audits/auxiliary.py` checks six auxiliary CSVs for quality, coverage, matching and distances. Full statistics, three figures and per-file recommendations: [auxiliary EDA report](../../experiment/eda/supplementary/auxiliary/report.md) and [machine-readable summary](../../experiment/eda/supplementary/auxiliary/summary.json). Main findings:

- Normalized `TOWN + BLOCK + STREET` matches every training and test record. Of 9,464 buildings, 285 postal codes are `NIL` and 11 maximum-floor/completion-year entries are `-1`; convert these placeholders to missing.
- MRT, mall and school coordinates support straight-line distances, but historical opening dates are missing. The 243 MRT line records represent only 192 distinct locations, requiring deduplication for nearby-station counts.
- COE spans 2021-01 to 2026-08 with an incomplete final month; stocks span 2021-01 to 2026-07, with 85 entirely missing market rows and 72 anomalous OHLC ranges. Prices from different markets must not be averaged directly. At a fixed training cutoff, subsequently released economic data are not known features; see the report's two prediction scenarios.

These checks support starting auxiliary-feature experiments, not assuming they will reduce prediction error.

## Temporal validation: first feature comparison

Run in the `cs5228` environment:

```powershell
conda activate cs5228
python -m analysis.main baseline
```

`analysis/experiments/baselines/baseline.py` reads raw CSVs, constructs features in memory and checks source SHA-256 after training. Using the temporal split above and fixed CatBoost parameters, it compares three feature groups:

| Model | Features used | Validation RMSE (SGD) | Mean prediction minus actual rent (SGD) |
| --- | --- | ---: | ---: |
| Training-period mean rent | No property features | 846.75 | -531.44 |
| A: property attributes | Normalized flat type, flat model and area | 761.42 | -534.06 |
| B: property attributes + temporal features | A plus approval-month index, month of year and years since lease commencement | 529.96 | -49.91 |
| C: B + Town | B plus Town | **487.22** | -48.29 |

The large A→B gain is consistent with higher validation-period rents, but B adds approval time and lease years together, so neither can receive sole credit. C has the lowest CatBoost RMSE here, yet monthly mean underprediction remains S$5–S$84; see [monthly errors](../../output/experiment_runs/baseline_mean_catboost/monthly_validation_error.png). Among adequately represented types, Executive has the highest RMSE (S$594.62, 1,778 rows).

See [metrics](../../output/experiment_runs/baseline_mean_catboost/model_metrics.csv), [monthly errors](../../output/experiment_runs/baseline_mean_catboost/monthly_errors.csv), [flat-type errors](../../output/experiment_runs/baseline_mean_catboost/flat_type_errors.csv) and [parameters/full results](../../output/experiment_runs/baseline_mean_catboost/results.json). These come from one temporal validation, **not the Kaggle test set**.

### Traditional machine-learning comparison

Four models use C's **same seven features** and **same 35,158 validation rows**. Ridge and Random Forest use one-hot categories with training-only encoding/imputation. LightGBM uses categorical columns with training-defined levels; CatBoost reuses C above. Initial parameters are fixed, without hyperparameter search.

Fixed parameters: CatBoost 400 iterations, depth 6, learning rate 0.05; Random Forest 160 trees, depth 18; LightGBM 400 trees, 31 leaves, learning rate 0.05; Ridge regularization 10. Full settings are in both `results.json` files. They establish reproducibility, not optimality.

| Model | Validation RMSE (SGD) | Mean prediction minus actual rent (SGD) |
| --- | ---: | ---: |
| Ridge regression | 624.88 | +361.99 |
| Random Forest | 493.77 | -56.65 |
| CatBoost C | 487.22 | -48.29 |
| LightGBM | **485.09** | -42.68 |

Ridge substantially overpredicts later rents; monthly bias is consistent with excessive extrapolation of historical growth, though no separate experiment confirms this explanation. LightGBM has the lowest RMSE here, about S$2.1 below CatBoost. This small single-split gap does not establish general superiority; longer horizons may change rankings. See [monthly comparison](../../output/experiment_runs/baseline_classical_models/model_comparison_by_month.png), [metrics](../../output/experiment_runs/baseline_classical_models/family_metrics.csv) and [run details](../../output/experiment_runs/baseline_classical_models/results.json).

## Next steps: auxiliary data and model improvements

Best current validation RMSE is about S$485, but models only use basic property attributes, approval month and Town. Within-town location, transport and amenities may differ; trees also tend to underestimate later rents. Test **spatial information** and **time variation** separately for stable gains, without assuming auxiliary data help.

### Specific auxiliary-data hypotheses

Most auxiliary files have no monthly-rent labels. Join or aggregate them into main-model features rather than training a separate rent model per file.

| Priority and data | Proposed features | Hypothesis and checks |
| --- | --- | --- |
| 1. [`sg-hdb-block.csv`](../../data/auxiliary/sg-hdb-block.csv) | Building coordinates, postal code, maximum floor, completion year, Subzone and related fields. | Precise location and building attributes may explain within-Town variation. Address coverage is 100%; handle `NIL`/`-1` and compare completion year against the existing age feature for redundancy. |
| 2. [`sg-mrt-stations.csv`](../../data/auxiliary/sg-mrt-stations.csv) | Straight-line distance to nearest MRT and station counts within chosen radii. | Transport access may affect rent. Verify whether stations operated at approval time; current `STATUS` cannot treat planned stations as historically operating. Calculate coordinate distances first, then assess whether walking-time data are worthwhile. |
| 3. [`sg-shopping-malls.csv`](../../data/auxiliary/sg-shopping-malls.csv) and [`sg-schools.csv`](../../data/auxiliary/sg-schools.csv) | Nearest mall/primary-school distances and nearby counts. | Amenities may add information beyond Town and MRT distance. Check coordinates, school types and historical existence, then test future-month RMSE gains. |
| 4. [`sg-coe-prices.csv`](../../data/auxiliary/sg-coe-prices.csv) | COE prices, quotas or bids from the last published month. | Economic conditions may relate to rent changes. Join by approval month using only published information; define monthly aggregation across bidding rounds first. |
| 5. [`sg-stock-prices.csv`](../../data/auxiliary/sg-stock-prices.csv) | Lagged monthly returns or aggregate indicators for a small set of representative companies. | Stock prices may reflect economic conditions but many companies/dates can introduce noise. Define selection and aggregation rules, then test stable value for later months. |

### Experiment order and criteria

1. Use **C features + LightGBM/CatBoost** as controls. Add building information, then MRT, malls/schools and macro indicators separately. Record join coverage, missingness, RMSE, MAE and mean bias under matched validation rules to identify incremental value.
2. Extend the single split to multiple forward monthly origins and a longer horizon. Compare full history, recent years and recency weighting, checking whether later underprediction decreases.
3. Ablate approval time and age separately to identify the A→B gains. Then tune LightGBM and CatBoost narrowly; their roughly S$2.1 gap on one split is insufficient to select the final model alone.
4. After selecting features and model, refit on all labeled `train.csv` rows and predict Kaggle `test.csv`. Report gains, unsuccessful attempts and possible explanations.

Perform joins, imputation and feature calculations in new dataframes, leaving raw CSVs intact. If submodel predictions later become main-model inputs, use temporal out-of-fold predictions only to avoid target leakage.
