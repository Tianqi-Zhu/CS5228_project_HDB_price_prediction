[Chinese version](BASELINE_27_FEATURES.zh-CN.md)

# HDB monthly rents: baseline features and common-input models

This file preserves evidence for the baseline 27-input policy, field ablations and common-input comparisons.
See [model selection and later rounds](../01_final/README.md) for subsequent exploration,
and [FINAL_MODEL.md](../01_final/README.md) for the current final method.
Interpret "default" and "current result" here as the baseline stage, not the later final blend.

<!-- final-selection-link -->
Final method/exact submission scores are updated in [selection and report conclusions](../01_final/README.md). Recommended structural blend: displayed415.37529; history/macro 415.86912/415.78232. Earlier stage narratives remain below.

This report asks how raw fields become sensible inputs, how much each field/group adds,
and how classical/deep models compare under common information. Main uses competition labels and supplied auxiliaries,
excluding target-month rents from public sources. Public-target matching is a separate overlap audit outside predictive rankings.

## 1. Task, data and validation

Predict individual monthly rents in SGD; primary RMSE, with MAE and mean predicted-minus-actual bias.
Train150,000 rows(2021-01..2025-03); test 50,000(2025-03..2026-07).
Six auxiliaries cover blocks, MRT, malls, schools, COE and stocks; availability does not require including every source.
Retain duplicates; exclude constant FURNISHED/FEE; do not delete records solely by rent magnitude.

Three annual selection windows: 2022-04..2023-03,2023-04..2024-03,2024-04..2025-03;
fit only before each origin. Separately report sixteen-month2023-12..2025-03 stress evaluation,
which overlaps annual windows and is excluded from pooling. Pool sample-weighted squared errors before square root,
not RMSE values directly. Histories freeze at fitting cutoff without validation-rent updates.

Every labelled month was previously examined; these support selection, not independent testing.
Static facilities lack complete opening dates; spatial features are proxies under supplied competition information,
not evidence of complete historical-availability auditing.

## 2. Feature hypotheses and choices

Default27 inputs: five property, three time, five location, two each for MRT/malls/primary schools, and four histories at each of two levels.
See [field dictionary](feature_dictionary.csv) for definitions, sources, rationale and cutoffs,
and [feature guide](../../docs/methods/baseline_features.md) for implementation/usage.

Normalize flat-type formats; addresses support block joins/history keys. Building height is not unit floor,
and approval-minus-lease year only proxies age. Spherical distance proxies accessibility, not commuting time or causality.
Nearest distances/counts within1km describe nearest alternatives/local choice; group deletion tests incremental value.

Block/type and planning-area/type histories provide past mean, count, last-month mean and months since observation.
Fitting rows query strictly earlier months; validation queries fitting histories only. Unseen groups have missing statistics/count0;
coarser histories remain usable. Vocabularies/numerical transforms fit on fitting data only.

Original SIGNATURE and BLOCK_FLAT histories duplicate row by row; remove four fields from 31.
This structural audit is stronger than deleting by one importance ranking, but needs rechecking on new data.

## 3. From baseline attributes to historical features

Fix LightGBM at 400 trees, 31 leaves, learning rate0.05, seed 42 and identical fitting/validation rows.
Adjacent policies can change several fields; compare policies without attributing everything to one variable.


| Feature policy | 2022 annual window | 2023 annual window | 2024 annual window | 16-month stress window |
| --- | --- | --- | --- | --- |
| Seven basic inputs | 722.46 | 595.68 | 485.09 | 502.21 |
| 29 spatial inputs | 719.97 | 591.90 | 476.97 | 495.21 |
| 19 compact static inputs | 719.59 | 589.76 | 477.02 | 493.89 |
| 31 static/history inputs | 711.08 | 582.59 | 468.49 | 491.22 |
| 27 after duplicate-history removal | 711.08 | 582.59 | 468.49 | 491.22 |


Richer40-static/52-history policies do not consistently outperform compact inputs; more variables do not guarantee gains.
Historical windows, decay, smoothing, multiple radii, address categories and market decomposition received100 fixed-model fits;
see [detailed experiments](../../docs/methods/history_feature_optimization.md). Premium gains are small/unstable;
trend-residual decomposition helps some rising-market windows but can worsen turning points, so remains separate research.

COE/stocks have four-window controls; their joint bundle worsens4/4 under the earlier monthly-update protocol.
Those negative results use a different protocol and cannot be ranked against fixed-origin scores. Realized future macro observations
are unknown at the batch forecast origin and excluded by default. See [economic ablation](../../output/experiment_runs/features_economic_ablation/report.md).

## 4. RMSE contribution of each input/group

Add156 fits: full27,27 single deletions, eight groups, three joint deletions=39 configurations×four windows.
Refit after every deletion, fixing settings, rows and cutoffs. Define:

```text
Delta RMSE = RMSE after deleting specified inputs and refitting − full27-input RMSE
Positive means deletion worsens;negative means it improves.
```

Single deletion measures marginal contribution while others remain. Historical encodings still carry type/location/time information,
so removing area, location or type fields does not remove all related information.
Contributions are neither additive shares nor causal effects; source-level intervention requires redesigning derived features.

### Group and joint deletions


| Deleted input group | Pooled annual delta RMSE | Latest annual delta RMSE | Annual windows worsened | 16-month delta RMSE |
| --- | --- | --- | --- | --- |
| Time + all histories | +141.33 | +114.11 | 3/3 | +141.62 |
| All histories | +7.94 | +8.53 | 3/3 | +2.67 |
| Block/type histories | +6.18 | +9.27 | 3/3 | +5.79 |
| Time | +4.03 | +3.43 | 3/3 | +4.33 |
| Property attributes | +3.46 | +3.54 | 3/3 | +15.28 |
| Location | +2.51 | +0.24 | 2/3 | +4.16 |
| MRT | +0.74 | -0.01 | 1/3 | +1.55 |
| Primary schools | +0.61 | +0.37 | 2/3 | -0.88 |
| All amenities | +0.58 | +0.01 | 2/3 | +0.90 |
| Malls | -0.19 | -0.43 | 1/3 | +0.66 |
| Planning-area/type histories | -0.26 | -1.06 | 1/3 | -4.30 |


### Eight largest pooled single-deletion effects


| Deleted field | Pooled delta RMSE | Latest delta RMSE | Annual windows worsened |
| --- | --- | --- | --- |
| APPROVAL_MONTH_INDEX | +13.51 | +5.02 | 3/3 |
| HIST_BLOCK_FLAT_MEAN | +5.10 | +7.56 | 3/3 |
| HIST_BLOCK_FLAT_RECENCY_MONTHS | +3.35 | -0.30 | 1/3 |
| CBD_DIST_KM | +2.46 | -0.22 | 2/3 |
| MALL_DIST_KM | +2.20 | -0.38 | 1/3 |
| PLANNING_AREA_CLEAN | +1.59 | -0.39 | 1/3 |
| LONGITUDE | +1.28 | -0.53 | 2/3 |
| HIST_AREA_FLAT_COUNT | +1.16 | -0.27 | 1/3 |


See all 27 [single-field results](single_feature_effects.csv),including latest/stress differences.
Complete folds:[fold_effects.csv](../../output/experiment_runs/features_ablation_evidence/fold_effects.csv).
![Group deletions](../../output/experiment_runs/features_ablation_evidence/group_deletion.png)

Time/history inputs partly substitute; examine joint deletion, not single rankings alone.
A slightly beneficial deletion does not prove universal uselessness or justify combining individual gains into a new optimum.
Keep the 27-input default rather than continuing stepwise selection on examined outer labels.

Latest-window paired three-month contiguous-block resampling uses 2000 draws and 2.5/97.5 percentiles.
It describes month sensitivity conditional on fixed predictions; only twelve months, without fitting randomness/future-market uncertainty
or multiplicity correction. It is not formal significance evidence.

## 5. Common-information models versus complete pipelines

Models test distinct hypotheses rather than increase architecture counts:

| Control | Question tested |
| --- | --- |
| LightGBM, CatBoost, XGBoost, HistGradientBoosting | Nonlinear ensembles for mixed numerical/categorical tables; main predictive references |
| ExtraTrees | Whether randomized trees provide useful contrast with boosting |
| Embedding MLP / ResNet | Whether jointly learned embeddings/numerical inputs outperform trees under identical information |
| FT-Transformer 96/192 | Whether token attention and greater width add value |
| TabNet | Whether sequential attentive feature selection helps this table under limited budget |

Sources:[LightGBM paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html),
[FT-Transformer/tabular deep baselines](https://arxiv.org/abs/2106.11959),
[TabNet paper](https://arxiv.org/abs/1908.07442). Project scores come from local experiments below,
not rankings transferred from other datasets.

### Same27 inputs, 114,842 fitting rows and 35,158 validation rows

Select parameters/epochs only from original compact-input inner2023-10..2024-03 trials;
refit all pre-April2024 rows. No market-residual offset or warm-up row deletion.
Reuse exactly matched old outer fits; fit others anew. Model-adapted preprocessing is allowed, but all 27 raw inputs match.


| Model | Same27-input RMSE | MAE | Delta RMSE versus LGB | Month-block sensitivity interval |
| --- | --- | --- | --- | --- |
| XGBoost | 467.03 | 338.87 | -1.84 | [-2.48, -1.28] |
| HistGradientBoosting | 468.02 | 339.25 | -0.85 | [-1.90, +0.44] |
| CatBoost | 468.46 | 343.51 | -0.40 | [-2.78, +2.53] |
| ExtraTrees | 468.47 | 341.08 | -0.40 | [-1.66, +0.93] |
| LightGBM | 468.87 | 340.71 | +0.00 | [+0.00, +0.00] |
| FT-Transformer 96 | 476.00 | 339.92 | +7.13 | [+4.35, +9.76] |
| FT-Transformer 192 | 476.25 | 342.47 | +7.38 | [+4.77, +9.86] |
| Embedding MLP | 483.72 | 358.21 | +14.85 | [+10.45, +19.09] |
| Embedding ResNet | 493.88 | 381.85 | +25.01 | [+8.07, +43.07] |
| TabNet | 498.53 | 383.79 | +29.67 | [+18.74, +39.56] |


This better approximates an architecture control, but unequal search budgets, preprocessing and one seed remain limitations.
LightGBM here selects settings/duration internally, unlike section4's fixed 400-tree468.49 reference;
identical27 inputs need not imply identical hyperparameters, so differing scores are not contradictory.
Small differences do not establish universal model-family superiority. Paired-block intervals retain the previous interpretation.

### Full pipelines with model-specific features/targets

Original search covers30 inner configurations and ten specifications, allowing compact, premium, entity and market-residual inputs.

Lowest observed latest-annual RMSE is **465.66**,an inner-weighted blend.

This compares pipelines, not architecture alone. See [original model comparison](../../docs/methods/model_comparison.md).
Early/16-month checks cover selected trees; one recent neural year cannot support all-period conclusions.

## 6. Sources of error

These latest-window diagnostics use the same fixed 27-input LightGBM as section4's deletion control.
Bias is prediction minus actual. True-rent grouping is post-hoc; unknown query rents cannot select models/corrections.


| Actual rent band (diagnostic only) | Samples | RMSE | Predicted-minus-actual bias | Share of window SSE |
| --- | --- | --- | --- | --- |
| 2000-3000 | 14059 | 407.49 | +161.91 | 30.3% |
| 3000-4000 | 16629 | 325.73 | -147.05 | 22.9% |
| <=2000 | 2107 | 1143.13 | +1023.21 | 35.7% |
| >4000 | 2363 | 604.82 | -453.30 | 11.2% |
In the latest window, 2,107 records with true rent≤2000(6.0%) contribute35.7% of squared error. Their systematic overprediction merits study more than adding further amenity fields.



Interpret low-rent overprediction/high-rent underprediction alongside mean regression, missing unit details and market change,
not as direct causal mechanisms or permission for target-leaking group calibration.
Further breakdowns by month, horizon, town, type, history support and staleness are in
[error_diagnostics.csv](../../output/experiment_runs/features_ablation_evidence/error_diagnostics.csv).
Tables include counts/SSE shares to avoid focusing only on small high-RMSE groups.
Monthly diagnostics locate problems without creating future-label inputs.

## 7. Final reproduction and information boundaries

Retain fixed 27-input, 400-tree LightGBM as the simplest reproducible reference,
latest RMSE 468.49; it is not the 465.66 search-selected blend.
`python -m analysis.main canonical-submit` fits all 150,000 labels,
saves vocabularies/histories with the model, reloads/checks it and predicts50,000 test rows.

Maximum reloaded-prediction difference is0; 50,000 output rows.

Outputs:`output/submissions/baselines/baseline_27_features/`,without automatic upload; check summary in
[baseline_27_feature_submission/summary.json](../../output/experiment_runs/baseline_27_feature_submission/summary.json).
March2025 query histories exclude same-month labels, but the final supervised model fits all known labels,
so those380 queries are not completely pre-month fitted predictions.

Public-rent matching is isolated in `experiment/public-rent-lookup`.
Approximately180 is public-target reconstruction error, and 173 a candidate-variance estimate, not ordinary forecasting scores.
Other teams' methods are unknown; course/competition rules determine whether reconstruction is allowed.

## 8. Completion and remaining work

| Item | Current status |
| --- | --- |
| Data quality, join coverage and main/auxiliary EDA | Reports available; raw hashes protected |
| Explicit default inputs and per-field rationale | 27-field dictionary/shared implementation complete |
| Feature groups, history representations and redundancy | Multi-window comparisons and duplicate-deletion confirmation available |
| Actual per-field/group RMSE increments | Added156 refits and fold results |
| Same-input ML/DL comparison | Completed ten common-input specifications |
| Search with model-adapted inputs | Completed 30 inner configurations under a separately described protocol |
| Error analysis and reproducible export | Added history-support/staleness/horizon diagnostics and final reload checks |
| Multiple neural seeds/more forward windows | Outstanding; avoid universal architecture rankings; bounded future work |
| Historical opening dates for every facility | Missing; retain limitations or run no-facility sensitivity |
| Genuinely untouched test/actual Kaggle results | Not available; current outer data cannot be relabelled untouched |
| Final course format, pages, workload and references | Finalize against original course rubric; the complete rubric is absent here |

Do not add latent-variable models merely for complexity. Extend only with diagnostic-driven hypotheses, clear input provenance and validation.
Prioritize an evidence-based report/presentation over endlessly adding fields/architectures.

## 9. Rerun entry points

```powershell
# Per-field/group deletion refits and error diagnostics
python -m analysis.main feature-evidence
# Common27-model controls require original model-search inner trial results
python -m analysis.main common-models
# Refit simple default on all data,reload and export local submission CSV
python -m analysis.main canonical-submit
# Read existing results to regenerate this report,dictionary and index
python -m analysis.main project-review
```

See [model guide](../../docs/methods/model_comparison.md) for environments/original searches. Historical entry points remain in
the [running guide](../../docs/guides/running_experiments.md); reading main results does not require running each.
Machine-readable gaps/evidence paths:[output/experiment_runs/foundation_feature_review/README.md](README.md).
