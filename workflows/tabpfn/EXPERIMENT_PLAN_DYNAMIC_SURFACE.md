# Experiment plan: test a dynamic rental surface efficiently

This plan revises the supplied 71-section *Dynamic Rental Surface + Static Spatial Context + Temporal Factor Forecasting + CatBoost Residual Model* proposal against experiments already in this repository. The research question remains: **does a forecast spatial market state improve future transaction-level rent predictions beyond a strong model using property, location, amenities, and earlier rents?** The surface and residual model are hypotheses to test, not the predetermined final model.

## Decisions from completed experiments

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Earlier rental history is the strongest tested addition. | The historical HistGradientBoosting model has pooled RMSE S$543.10 across the disjoint 6-, 9-, and 12-month folds, versus S$549.10 for the static spatial baseline. On the latest 12-month fold it scores S$469.61 versus S$478.82. [Results](reports/spatiotemporal_hybrid/report.md). | Retain this **historical** variant as the minimum row-level comparator. Give new models the same history features. |
| Block and amenity information is already joined and tested. | On the latest fold, adding block features changed RMSE from S$485.72 to S$480.22; adding amenities reached S$478.82. [Ablation](reports/auxiliary_ablation/ablation_metrics.csv). | Reuse `analysis/data.py` and its spatial features. Test only genuinely new spatial features individually. |
| A recursive planning-area × flat-type residual forecast did not help the historical model. | Pooled RMSE S$543.13 versus S$543.10; latest fold S$469.90 versus S$469.61. [Results](reports/spatiotemporal_hybrid/report.md). | A new temporal component must show that it improves on this simpler attempt. |
| A block-level comparable graph did not yield consistent gains. | Raw local rent worsened all four folds, by S$0.06 to S$1.59 RMSE; normalized premium and residual versions had mixed, sub-S$1 changes. [Results](reports/local_comparable_graph/report.md). | Avoid another nearest-block graph or graph neural network in the primary path. |
| The Bayesian amenity model and economic features did not beat the tree. | Best Bayesian model with history: S$484.64 on the latest fold versus S$469.61 for the tree. COE plus stock features worsened all four complete forward windows of the static model. [Bayesian](reports/bayesian_amenity_factors/report.md); [economic](reports/economic_feature_ablation/report.md). | Keep their interpretation in the report; do not repeat broad macro or Bayesian tuning to optimize RMSE. |
| The existing model-family comparison is incomplete for the strongest feature set. | Basic-feature LightGBM scored S$485.09; basic CatBoost scored S$487.22 on the latest fold. [Results](reports/model_families/family_metrics.csv). | Compare LightGBM and CatBoost on the **full historical feature set** before crediting a new surface with a gain. |

The first three historical folds are disjoint. The separate 16-month check overlaps them; its historical RMSE is S$489.60, and it must not be included in their pooled score.

## One validation and information contract

1. Use the repository's expanding, fixed-origin folds: 2023-01 to 2023-06, 2023-07 to 2024-03, and 2024-04 to 2025-03. Use the existing 2023-12-origin, 16-month horizon as a separate stress check. Keep the same target rows, feature availability, and metric definitions for every candidate.
2. At origin `T`, every validation month uses only rental labels dated **before `T`**. Neither rental histories nor market-surface forecasts are refreshed with validation targets. Static block and amenity files are snapshot features, not dated evidence that a facility was open in a historical month.
3. For training rows, construct target-derived features out of time: a row in month `t` may use labels only from months `< t`. A final residual learner must receive market predictions generated from earlier origins, never a surface containing its own target.
4. Treat the March 2025 train/test overlap explicitly. A strict single-origin forecast for **all** test months can use data through February 2025. A separate competition-oriented variant may use a February cutoff for March test rows and a March cutoff for April 2025 onward, with separate fitted models. A model trained on March rents must not predict March test rows under the strict historical claim.
5. Select using RMSE in SGD on the three disjoint folds. Also report MAE, bias, per-month error, flat-type/town error, and performance by forecast horizon. The 16-month check is diagnostic, not a tuning fold. The latest fold has already been examined repeatedly, so no existing fold is an untouched final test.

## Stage 1 — establish the strongest affordable row model

Reuse the current spatial and history builders. Compare the existing historical HistGradientBoosting model with LightGBM and CatBoost using the **same property, block, amenity, and prior-rent features**. Normalize flat-type labels and block keys with the existing code. Keep `FURNISHED` and `FEE` out because they are constant. Use a small, fixed configuration budget for each new family; tune on an inner chronological slice of each fitting period if early stopping is needed. Do not choose hyperparameters using a fold's reported validation target.

As a focused feature experiment, test recent 3-, 6-, and 12-month same-signature and block × flat-type summaries, counts, and recency, with sparse groups shrunk toward planning-area × flat-type estimates. The current history uses expanding means, counts, last-month means, and recency; recent-window and shrinkage behavior remain untested. Keep each new history feature strictly prior-month. Select the best validated row model as the comparator for all subsequent stages.

**Gate:** do not claim a structured-market improvement against the weaker static baseline or a basic-feature CatBoost model. Record fit time and memory as well as error so that a tiny gain can be judged against its cost.

## Stage 2 — test whether a forecast market index adds information

Start at the coarsest reliable resolution: a national index, flat-type indices, then pooled planning-area × flat-type indices where monthly counts support them. Approximately 150,000 rentals over 51 training months gives about 2,900 records per month **across all areas and types**; a fine grid subdivides these rapidly. The existing panel found 50 planning-area × flat-type groups with sufficient observations for its analysis. [Panel](reports/group_temporal_panel/report.md).

Build a **composition-adjusted index**: estimate expected rent from flat type, area, age, and broad location using only pre-origin data, then aggregate residuals by month and group with counts and shrinkage toward wider groups. This asks whether market movement is forecastable after removing changes in the mix of properties rented. Compare forecasts that freeze the last observed level, average recent levels, and use a regularized damped/local-level trend. Avoid a large VAR or neural sequence model with only a few dozen monthly states.

Measure both (a) forecast error of observed monthly indices, with observation counts reported, and (b) the effect of the **forecast** index as an added feature to the selected row model. For fitting rows, generate this feature in forward blocks from earlier months, then cache it by fold and origin. A one-step model that uses actual prior validation-month rents answers a different question and is not interchangeable with the batch forecast.

**Gate:** advance to a more detailed surface only if a simple index forecast beats persistence on forward months **and** its forecast feature improves transaction RMSE across forward folds. Otherwise stop the prediction-oriented dynamic-surface branch and report the negative result. A good-looking latent trajectory or lower in-sample reconstruction error is not enough.

## Stage 3 — a minimal spatial-surface proof of concept

Run this stage only if Stage 2 gives evidence of useful market movement, or if a separate scientific deliverable explicitly requires a surface experiment. The supplied plan's fixed 32 × 32, four-plus rental channels would create up to 4,096 cell × flat-type slots per month before masking, compared with roughly 2,900 transactions per month. Measure actual supported-cell occupancy **before** choosing a grid.

First compare one or two coarse resolutions selected from observed occupancy, not an automatic 16/32/48 × multiple-bandwidth search. Construct a single composition-adjusted premium channel with an observation mask, transaction count, and shrinkage to the Stage 2 area index. Compare its local prediction directly with the area index. Use an origin-specific fit: no rent from month `t` may enter a feature for a transaction in `t`.

Only if the coarse surface improves forecast and row-level error, fit a low-rank representation such as PCA on **training months only**. Evaluate reconstruction on observed cells and forecast performance separately. Compare factor persistence with one regularized damped-trend or local-level forecast. Limit latent dimensions to a small prespecified set, for example 3 and 8. Static MRT, mall, school, and HDB geometry remain separate row-level features; they do not belong in temporal PCA.

**Gate:** promote a surface only when its out-of-origin transaction RMSE improves on both the best row model and the Stage 2 index hybrid. A surface can be an informative visualization even if it fails this prediction gate.

## Stage 4 — compare only justified hybrids

Try **direct feature integration first**: append the forecast group or local market state, count, and confidence to the selected row model. If this helps, compare the proposed additive residual formulation `forecast_market_rent + residual_model(property, location, amenities, confidence)`. Give both hybrids the same training origins and feature availability. Check whether the additive model miscalibrates low- and high-rent flats; its market baseline may be less relevant for unusually sized or modeled properties.

There is no need to repeat global-median, Ridge, Random Forest, all existing static ablations, COE/stock PCA, graph variants, or a ConvLSTM unless a new question specifically calls for them. Their existing outputs remain part of the project evidence.

## Efficient run order and artifacts

| Pass | Runs | Advance when |
| --- | --- | --- |
| Reproduction | Reuse saved baseline predictions and verify fold/feature definitions; no retraining without a reason. | Baseline and new candidates share target rows and origins. |
| Row-model screen | One bounded LightGBM and CatBoost search on the full feature set, plus the recent-history ablation. | A candidate is competitive on at least the contrasting rising and later market folds. |
| Index screen | Persistence, recent-level mean, and one damped/local-level forecast; direct row-level feature test. | Improvement survives the three disjoint folds. |
| Surface screen | Occupancy report, one coarse composition-adjusted surface, persistence. | It improves beyond the group-index hybrid. |
| Factor/hybrid confirmation | At most two latent sizes, one temporal alternative, then direct versus additive hybrid. | A practically meaningful, consistent gain remains on all folds and the separate 16-month check. |

Predeclare a **practical target of about 1% pooled RMSE improvement** (roughly S$5 from the current S$543 pooled baseline) and inspect paired monthly errors rather than treating thousands of correlated rental rows as independent replications. This is a decision threshold, not a claim of statistical significance. Report every attempted candidate and its runtime. Preserve the simpler model if a more complex method wins only one fold or depends on an implausibly fine, mostly imputed surface.

Implement in the repository's existing `analysis/experiments/` and `analysis/` runner pattern, with outputs under `reports/`. Cache static block/amenity features once and target-derived artifacts by source hash, fold origin, and configuration. A wholesale `src/`, `scripts/`, and `outputs/` reorganization from the supplied plan would duplicate working infrastructure.
