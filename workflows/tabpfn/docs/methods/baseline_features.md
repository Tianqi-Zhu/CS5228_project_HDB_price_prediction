[Chinese version](baseline_features.zh-CN.md)

# Recommended feature engineering: 19 static + 8 historical features

Construction rationale and temporal availability for each field are in the [27-column dictionary](../../experiment/02_feature_engineering/feature_dictionary.csv).
See [new ablation evidence](../../experiment/02_feature_engineering/FEATURE_ABLATION.md) for actual RMSE changes after individual/group deletions,
and the [current project report](../../experiment/02_feature_engineering/BASELINE_27_FEATURES.md) for conclusions and writing guidance. Deletion effects are conditional predictive increments, not causal contributions.

The recommended entry point is `analysis.pipeline.features.RentalFeatureEngineer()`, whose default policy is
`compact_history_27`. It handles deterministic processing, category vocabularies and rental histories. The initial seven-way comparison
selected 31 inputs; a deeper audit removed four duplicate statistics without changing any of four window scores. See
[detailed feature experiments](history_feature_optimization.md) for evidence; this is not a claim of global optimality.

## Why consolidate the features?

Original feature lists were spread across baselines, auxiliary ablations, deep learning and spatiotemporal experiments. The same distance had multiple
aliases, and year/age fields were highly redundant. Experiments also mixed three-month, twelve-month and multi-fold evaluations,
as well as monthly-updated and fixed-origin multi-month forecasts. New modelling should start from an explicit default policy.

Training spans 2021-01 through 2025-03; test data span 2025-03 through 2026-07. The recommended policy uses
fixed-origin forecasting: histories freeze at the fitting cutoff, without future economic observations or actual rents.

## Default feature inventory

| Group | Fields | Count |
| --- | --- | ---: |
| Property attributes | `FLAT_TYPE_CLEAN`, `FLAT_MODEL_CLEAN`, `FLOOR_AREA_SQM`, `FLAT_AGE_YEARS`, `MAX_FLOOR` | 5 |
| Time | `APPROVAL_MONTH_INDEX`, `APPROVAL_MONTH_SIN`, `APPROVAL_MONTH_COS` | 3 |
| Location | `TOWN_CLEAN`, `PLANNING_AREA_CLEAN`, `LATITUDE`, `LONGITUDE`, `CBD_DIST_KM` | 5 |
| Amenities | Nearest distance and count within 1km for MRT, malls and primary schools | 6 |
| Historical rents | Four statistics for each of two groups; see below | 8 |
| Total | Four categorical and 23 numerical features | 27 |

`MAX_FLOOR` is the building's highest floor. `FLAT_AGE_YEARS` is approval year minus lease commencement year.
Amenity distances use the Haversine formula; counts use deduplicated coordinates.

The four categorical fields are `FLAT_TYPE_CLEAN`, `FLAT_MODEL_CLEAN`, `TOWN_CLEAN` and
`PLANNING_AREA_CLEAN`. Vocabularies are learned only from fitting data; unseen prediction categories become missing.
LightGBM uses pandas categoricals and numerical missing values directly. This default needs neither one-hot encoding nor
StandardScaler, and does not learn imputation values from the full dataset.

Preserve row count, order, indices and duplicates; do not remove rents using IQR. `FURNISHED` and `FEE` are
constant and excluded. Keep one field per physical distance. Omit high-cardinality raw addresses and postal codes;
addresses serve only as history keys. Also omit completion year, duplicate age definitions, multiple amenity radii, manual interactions and economic signals.
These choices are supported by comparisons of feature bundles, not individual proof that each omitted field lacks value.

## Historical-rent construction

The two grouping levels are:

1. Same block and flat type.
2. Same planning area and flat type.

The original history for an identical property-attribute signature duplicates the first group on this dataset and was removed.

Each group outputs `MEAN`, `COUNT`, `LAST_MEAN` and `RECENCY_MONTHS`: the mean of all earlier rents,
record count, mean in the latest observed month and months since that observation.

A record uses rents strictly before its approval month; all records in one month share the same preceding history.
Validation/test statistics use fitting data only. Unseen fine-grained groups have missing means and zero counts; the model can also
use planning-area history and static features. Counts expose the support behind each statistic.

The test set includes 380 records in 2025-03, also present in training. Their histories use labels from
2025-02 and earlier. For predictions from 2025-04 onward, means freeze at fitting statistics through 2025-03,
while months-since-observation increases. This is stricter than ordinary leave-one-out target encoding,
which may still use labels from months after the current row.

## Comparison design and results

Fix LightGBM at 400 trees, learning rate 0.05, 31 leaves, minimum leaf size 30, L2 3 and
seed 42. Every candidate uses identical fitting/validation rows, without early stopping or tuning.

The three ranking windows are disjoint twelve-month periods: 2022-04 through 2023-03, 2023-04 through 2024-03,
and 2024-04 through 2025-03. Fit only before each origin; validation labels never update histories.
Report the overlapping sixteen-month 2023-12 through 2025-03 stress window separately, outside pooled rankings.

| Feature policy | Features | RMSE from 2022-04 | RMSE from 2023-04 | RMSE from 2024-04 | Pooled three-window RMSE | 16-month RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline C | 7 | 722.46 | 595.68 | 485.09 | 605.20 | 502.21 |
| Canonical-field version of the original spatial policy | 29 | 719.97 | 591.90 | 476.97 | 600.77 | 495.21 |
| Compact static | 19 | 719.59 | 589.76 | 477.02 | 599.90 | 493.89 |
| Rich static | 40 | 723.46 | 606.72 | 479.77 | 607.93 | 503.04 |
| Compact static + history (initial policy) | 31 | 711.08 | 582.59 | 468.49 | 591.96 | 491.22 |
| Current default after the deeper audit | 27 | 711.08 | 582.59 | 468.49 | 591.96 | 491.22 |
| Original spatial + history | 41 | 712.74 | 587.68 | 470.03 | 594.75 | 491.13 |
| Rich static + history | 52 | 721.03 | 610.34 | 485.28 | 609.77 | 512.10 |

The 31-feature policy outperforms the 29-feature policy in all three annual windows, lowering pooled RMSE by approximately 1.47%.
The 41-feature policy is about 0.09 SGD better in the sixteen-month window. Annual stability and feature count initially favored 31
features; the deeper audit then reduced this to 27 with identical scores. There is no evidence that the 0.09 SGD
difference represents a reliable long-horizon advantage.

Complete seven-policy, multi-fold and sixteen-month results are in
[`output/experiment_runs/features_selection/`](../../output/experiment_runs/features_selection): 
`fold_metrics.csv` for fold scores, `pooled_metrics.csv` for weighted three-window results,
`monthly_errors.csv` for monthly scores, and `results.json` for fields, parameters, versions and source hashes.

Existing evidence for omitting economic inputs: COE+stocks underperformed the spatial baseline in all four forward
windows, and monthly-updated inputs do not match this fixed-origin task. See
[`features_economic_ablation/forward_scores.csv`](../../output/experiment_runs/features_economic_ablation/forward_scores.csv).

## Reuse and reproduction

Rerun comparisons of baseline and currently registered feature policies:

```powershell
python -m analysis.main feature-select
```

The recommended policy works in a sklearn pipeline. Call `fit` on chronologically separated fitting data, then `predict` on validation;
fit all training data only after fixing the configuration. This example performs final fitting without writing a submission:

```python
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from analysis.pipeline.config import FAMILY_SETTINGS
from analysis.pipeline.features import RentalFeatureEngineer

train = pd.read_csv("data/train.csv", dtype={"BLOCK": "string"})
test = pd.read_csv("data/test.csv", dtype={"BLOCK": "string"})
model = Pipeline([
    ("features", RentalFeatureEngineer()),
    ("model", LGBMRegressor(**FAMILY_SETTINGS["lightgbm"])),
])
model.fit(train.drop(columns="MONTHLY_RENT"), train["MONTHLY_RENT"])
prediction = model.predict(test)
```

Save the entire pipeline with joblib so vocabularies and rent histories travel with the model.
Baseline `validate --features A/B/C` remains the historical entry point; the class above supplies the new default policy.

See `output/experiment_runs/features_history_optimization/runtime_validation.json`: the current 27-input
sklearn pipeline reproduces latest-window RMSE 468.486718, fits all 150,000 training rows and predicts all 50,000 test records
in memory. Test order is preserved, all predictions are finite, and source CSV hashes pass verification.

## Limits of the conclusions

All labelled months were examined in earlier experiments. This round evaluates policy selection without new independent test labels.
Three annual improvements and the duplicate-field audit support 27 features as a default starting point; they do not establish future Kaggle performance or
optimality for every model. Early windows still show substantial underprediction; these features alone do not solve market-trend changes.
Amenity tables are static snapshots without historical opening dates. The experiment evaluates spatial proxies under the supplied competition-data conditions.
