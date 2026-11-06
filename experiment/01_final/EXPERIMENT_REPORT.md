[Chinese version](EXPERIMENT_REPORT.zh-CN.md)

# Report model selection and historical branches (through 2026-11-05)

<!-- final-choice:start -->
**Final choice (2026-11-05, user screenshot):** Recommend `kaggle_structure_blend.csv`,displayed Kaggle **415.37529**,versus macro 415.78232/history 415.86912. See [FINAL_MODEL.md](README.md) for method, local controls and report conclusions, and [final/submission.csv](../../output/submissions/final/submission.csv) for the identical copy. Selection is among these three submitted candidates.
<!-- final-choice:end -->

<!-- tenth-wave-status:start -->
**Round 10 architecture update (2026-11-05):** Actually trained numerical-embedding FT, Deep & Cross, conditional experts and asymmetric trees: 19 neural and four tree fits, with fixed round 9 blending controls. Lowest later blend scores 441.8345 recent/451.0295 annual; user submitted it, screenshot 415.37529, and selected it among three submissions. See [section14](#14-actual-architecture-training-comparisons) for full tables/selection scope.
<!-- tenth-wave-status:end -->

Final model/submission selection: [FINAL_MODEL.md](README.md). Snapshot through 2026-11-04: [REVIEW.md](../../output/experiment_runs/00_report_and_final/REVIEW.md). This file retains complete evidence and subsequent rounds.

<!-- ninth-wave-status:start -->
**Round 9 blending update (2026-11-05):** Reuse models and compare 31 blend configurations. Lowest development candidate **441.9591** recent/**450.8650** annual, improving442.7032/451.4310 by 0.7441/0.5660. Nonnegative weights use 24-/36-month support groups, month-centered errors and shrinkage toward global weights; all ten recent months improve. It was selected after viewing later development results and needs Kaggle checking. [New CSV](../../output/submissions/candidates/kaggle_blend_support_centered.csv); see [section13](#13-further-results-from-changing-only-the-blend).
<!-- ninth-wave-status:end -->

<!-- eighth-wave-status:start -->
**Round 8 actual-training update (2026-11-05):** New recent RMSE **442.7032**,versus 442.9025, improves0.1993; annual 451.4310 versus 451.1810. Remove five36-month FT fields, train three epochs on four seeds, equally average, and replace the FT member at its original 15.9872% weight. Completed 65 new development fits, 52 saved-model replays and 17 tests. Modest candidate improvement awaits Kaggle; see [section12](#12-small-improvements-and-actual-training-results).
<!-- eighth-wave-status:end -->

<!-- seventh-wave-status:start -->
**Round 7 Transformer update:** Measured TabDPT1.3 including supervised adaptation, Nori1.5/6M and strong-history TabICLv2. New blend annual/recent 450.4823/442.9704. See [section11](#11-updated-transformer-routes) for results/budgets. Target 430 remains unmet.
<!-- seventh-wave-status:end -->

<!-- sixth-wave-status:start -->
**Round 6 update:** Measured four additional deep-model families. New blend annual 450.4924 versus 451.1810, but recent 444.6888 versus 442.9025. Earlier item 8 remains a control; see [section10](#10-further-evaluation-of-stronger-deep-tabular-models). Target 430 remains unmet.
<!-- sixth-wave-status:end -->

## 1. Recommended model set

**Keep seven same-input standalone models in the body, then separately compare history/macro/structural pipelines and add a mean reference.** Models: Ridge, ExtraTrees, LightGBM, XGBoost, CatBoost, Embedding MLP, FT-Transformer. History blend annual 451.18/recent 442.90 is the strong reference; final structural blend451.03/441.83 has the lowest displayed Kaggle score 415.37529 among three submissions. Retain round 8(442.70/451.43) and round 9(441.96/450.87) as process controls, including final annual deterioration versus round 9. See [FINAL_MODEL.md](README.md).

Compare models on 27 shared inputs, then explain feature increments with fixed models, and finally compare enhanced representations/blends. Do not combine different inputs/cutoffs into an architecture ranking.

| Model | Annual RMSE with 27 shared inputs | MAE | Bias (predicted−actual) | Role in the report |
|---|---:|---:|---:|---|
| Ridge | 513.03 | 363.23 | +194.11 | Interpretable linear reference; tests need for nonlinear interactions |
| ExtraTrees | 468.47 | 341.08 | +4.79 | Randomized tree ensemble; mechanistic contrast with boosting |
| LightGBM | 468.87 | 340.71 | +11.67 | Main feature-ablation learner; fast nonlinear iteration |
| XGBoost | 467.03 | 338.87 | +20.64 | Alternative regularized boosting reference/blend member |
| CatBoost | 468.46 | 343.51 | -3.71 | Native categorical handling; checks historical-feature gains on another tree learner |
| Embedding MLP | 483.72 | 358.21 | -11.83 | Neural baseline; distinguishes embeddings from attention |
| FT-Transformer | 476.00 | 339.92 | +87.64 | Feature attention; tests complexity and blend complementarity |

### History-feature blend: 442.90 strong reference preceding final submission

Retain this strong reference alongside the final method. The first seven models compare architectures on shared inputs; this item explains enhanced histories/complementarity. Final performance conclusions use the top structural blend/three-submission comparison. Explain the tables separately because inputs differ.

| Validation scope | Rows | RMSE (SGD) | MAE (SGD) | Mean bias |
|---|---:|---:|---:|---:|
| 2024-04—2025-03 | 35,158 | **451.1810** | 329.4085 | -2.8559 |
| 2024-06—2025-03 | 28,995 | **442.9025** | 323.7295 | -20.6662 |

Blend components (predictions restored to SGD monthly rent before weighting):

| Component | Weight | Main information |
|---|---:|---|
| Round 2 history-tree ensemble (seven LightGBM/CatBoost members) | 76.2602% | Market-relative rents, 12/24/36-month anniversary histories; exact/neighbor months, block/coarser groups |
| FT-Transformer residual model | 15.9872% | 62 history fields + five past-market/shrinkage fields |
| Depth5 XGBoost residual model | 5.5224% | Same inputs; predicts residuals from a historical baseline |
| Depth7 direct XGBoost | 2.2302% | 62 history inputs; direct SGD rent prediction |

This is a **four-branch blend containing ten underlying models**,not one442.90 standalone model. See `best_ensemble_members.csv` for expanded weights, input counts, iterations and checkpoint locations.

Anniversary histories first improve individual models; tree/FT error differences then reduce RMSE through blending. Merely replacing a27-input model with FT does not produce442.90.

This policy uses train/supplied auxiliary data. Groups were selected after later development inspection, so call it the best development policy. The earlier-locked full blend443.0052 is a robustness control. Overlapping windows do not add independent samples.


Annual training-mean RMSE is846.75. All27-input standalone models fit114,842 rows through 2024-03 and predict identical35,158 April2024–March2025 rows. Learn vocabularies, imputation and scaling on fitting data only; valid preprocessing may differ by model.

The old common27 comparison included every model except Ridge. This round adds Ridge, selecting alpha from 1/100/10000 on 2023-10 through 2024-03. Other models retain their matched-input inner temporal selection. Unequal budgets, one seed and reused validation mean controlled development evidence, not a universal architecture ranking.

**If space is tight**,move ExtraTrees/Embedding MLP to appendix and retain Ridge, LightGBM, XGBoost, CatBoost, FT. For neural-structure emphasis, keep MLP and shorten XGBoost details. Put history/macro/final complete pipelines in a separate three-row table.

## 2. Explain feature engineering with fixed learners and iterations

Inputs nest strictly: 19 property/time/location/amenity fields; 27 adds block/type and planning-area/type mean, count, last mean and staleness; 47 adds market-relative histories at five levels; 62 adds block/type support, raw rent, market ratio, dispersion and low-rent share around12/24/36 months earlier(±one month).

Add16 fits: LightGBM353 trees/15 leaves/lr0.03/L2=10; GPU CatBoost535 iterations/depth6/lr0.04/L2=10. Settings/durations come from earlier windows and stay identical across four levels. Retain all validation rows without rent updates.

These settings differ from section1's model-selection settings: 31-leaf/108-tree LightGBM and CPU CatBoost there, versus 15-leaf/353-tree LightGBM and GPU CatBoost here. Different27-input scores reflect configuration differences, not contradictions. Attribute effects within each table only.

| Model / window | 19 inputs | 27 inputs | 47 inputs | 62 inputs | Interpretation |
|---|---:|---:|---:|---:|---|
| LightGBM / 2024-04—2025-03 | 481.60 | 467.80 | 466.97 | 457.84 | First ordinary-history gain, then anniversary-history gain |
| CatBoost GPU / 2024-04—2025-03 | 479.33 | 466.43 | 465.99 | 455.17 | First ordinary-history gain, then anniversary-history gain |
| LightGBM / 2024-06—2025-03 | 474.18 | 459.15 | 458.43 | 447.17 | First ordinary-history gain, then anniversary-history gain |
| CatBoost GPU / 2024-06—2025-03 | 473.11 | 458.60 | 458.06 | 446.16 | First ordinary-history gain, then anniversary-history gain |

Each delta is a predictive feature-group increment for this fixed learner/recipe, not a causal effect or detected contract renewal. There are no unit/tenant IDs, only block/type. Early rows lack two-year history (count0, statistics missing); distant queries use months observed by the forecast origin only.

**Strongest feature conclusion:** time/past rents matter most; market normalization yields smaller model-dependent gains; 12/24/36-month cohorts add clearer information. More fields do not guarantee improvement:±three-month windows, local normalization and tail statistics improve depth8 GPU CatBoost early526.08→525.21 but worsen annual 454.77→458.71 and recent 445.58→449.53. Retain this overfitting/validation counterexample.

Early fixed-LightGBM A→H ablation supports the basic narrative: property758.24,+time526.11,+town485.09,+block480.20,+amenities476.97; COE479.22/stocks480.54 do not consistently help. Economic availability differs and is excluded from this fixed-origin batch. The19/27/47/62 table more cleanly isolates histories.

Existing156 deletion refits supplement interpretation: removing time plus all histories adds141.33 pooled annual RMSE; all histories alone adds7.94; all amenities only 0.58. Correlated inputs substitute, so do not sum individual contributions or equate field deletion with removing source information. See [feature evidence](../02_feature_engineering/FEATURE_ABLATION.md).

## 3. Performance progression and RMSE targets

| Policy | Annual35,158 rows | Recent28,995 rows | Evidence type |
|---|---:|---:|---|
| Initial search blend | 465.66 | — | Earlier development reference |
| Frontier blend | 463.76 | 456.82 | Weights selected on earlier windows, then fixed |
| Round 2 anniversary-history blend | 452.94 | 443.86 | Weights selected on earlier windows, then fixed |
| Round 3 complete blend | 451.31 | 443.01 | Prespecified on earlier windows |
| **Recommended item 8:best history-feature blend** | **451.18** | **442.90** | Round 3 with newly added model group only; best development policy |
| Current regularized extended blend | 451.31 | 443.01 | Earlier window retains original full blend; all added weights0 |

The27-input table addresses architecture; this table permits feature, target and blending changes and evaluates full pipelines. Do not infer an architecture's gain across them. Default400-tree LightGBM468.49 differs from inner-stopped108-tree468.87; label both settings accurately.

Target 400–430 remains unmet. Round 3 full-blend annual/recent MAE331.05/324.82 worsens round 2's328.44/322.92; new-model-only recent MAE323.73. RMSE-oriented selection has tradeoffs rather than improving every metric.

Ten extra configurations include CatBoost depths4/5/7/9, stronger-L2 depths6/8, CPU Ordered depths4/6, and two regularized random-threshold LightGBMs. Evaluate first on 2023-10 through 2024-03; select weights with fixed 0.001 shrinkage and 0.25 maximum per new model. None receives weight, so no later refits or claimed record from a chance inner minimum. Complete results in `optimization/screen/`.

Annual/recent windows overlap and cannot be pooled as independent samples. Separate pre-origin selection for the 16-month stress window yields round 2 RMSE 479.64 versus old XGBoost487.58. Replacing it with shorter-window442.90 would falsely imply long-horizon forecasting is solved.

## 4. Review of all GitHub branches

On2026-11-04,`git fetch origin --prune` refreshed all ten actual remote branches;`origin/HEAD` is a symbolic main reference, not an eleventh branch. Git objects were read without switching workspaces or overwriting edits. Commits/evidence blobs are in `branches.json`,`branch_inventory.csv`; deduplicated small reports in `branch_evidence/`.

| Branch | Commit | Unique content | Report treatment |
|---|---|---|---|
| `codex/rnn-lstm-market-features` | `cb167e9f` | RNN/LSTM market state, window/cutoff search, three seeds | Monthly updates with actual preceding-month data; does not beat its own tree control overall |
| `experiment/dynamic-rental-surface-plan` | `469e34e3` | Dynamic market index, rental surface, PCA, AR/state-space and CatBoost hybrid | Explain decomposition; separate transaction and smoothed-surface errors |
| `experiment/public-rent-lookup` | `b1429600` | Normalized matching of public target records | 181.25 measures target reconstruction; outside train+auxiliary-only forecasting |
| `experiment/tabpfn` | `ae9da8e6` | TabPFN3/3.5, context/calendar/alignment/adaptation; official TabM; spatial extensions | Main pretrained route; frozen77 scores 455.81, MSE adaptation455.07; official TabM463.20 |
| `feature/midterm-eda-models` | `e56cbf7b` | CatBoost A/B/C, Ridge/RF/LGB, auxiliary features and early deep models | Evidence for basic attributes, time, space and original models |
| `feature/neural-tabm-experiment` | `21e4ec69` | Custom TabM-style and history-tree blend | Network annual 561.71; tree+6% network468.83; not official TabM |
| `feature/temporal-auxiliary-analysis` | `c3c339eb` | Historical spatial trees, lagged macro and grouped panels | Separate fixed-origin histories from monthly-updated economic protocols |
| `main` | `b05b17f7` | Code scaffold | No evaluable model or score |
| `relu-rental-prediction` | `dde07705` | sklearn ReLU MLP, nested epoch selection and fitting-only IQR | 2025Q1 only: 463.97; not ranked with annual results |
| `ziyuan——spatialfeatures` | `7caf2cb7` | LightGBM/CatBoost/KNN spatial references and controlled analyses | 26-input LGB477.20; fixed effects/matching explain associations, not predictive rankings |

New local feature_selection, features_ablation_evidence, comparison_model_search, comparison_same_27_features, round_01_history_target_search, round_02_anniversary_history and third_wave results are not yet in the current remote commit. This report separately reads local results without presenting them as uploaded GitHub work. Trace every path through `local_run_index.csv`.

## 5. Additional models and why not put everything in the body

| Route | Representative result/control | Recommendation |
|---|---|---|
| Official TabM | Recent463.20 versus frozen77-input TabPFN455.81; three-seed mean462.95, SD2.46 | Illustrate modern parameter-sharing ensembles; distinguish old TabM-style |
| TabPFN | 68-input Fast459.81; 77-input frozen455.81, DiagScale455.21, MSE-only 455.07 | Extend pretrained-model discussion; inputs differ from main27-input table, so no shared architecture ranking |
| TabPFN adaptation, sampling and spatial/external augmentation | Q/K LoRA, native heads, multiple views and local contexts do not consistently beat frozen controls; MSE-only minus matched loss control is only−0.051 with interval crossing zero | Summarize negative results briefly; external income/land-cover data exceed this train+auxiliary scope |
| TabICLv2 | Annual488.18 with 47 inputs/24000 context | Retain as a counterexample to automatic foundation-model superiority |
| HistGradientBoosting | Annual468.02 on common27 inputs | Overlaps the boosted-tree narrative; include in full supplementary table |
| ResNet / TabNet / FT-192 | Common27 inputs: 493.88 / 498.53 / 476.25 | Complexity need not help; FT-192 and FT-96 need not each occupy a main-text row |
| ReLU MLP | 2025Q1 score 463.97; local reproduction463.78 | Three months only; appendix result. Main MLP uses common27 inputs and 483.72. |
| RNN/LSTM market features | Monthly-updated control478.52; RNN42 scores 478.89, LSTM42 scores 479.24 | Negative sequence-representation result, not a fixed-origin comparison |
| PCA dynamic rental surface + state space | Annual residual hybrid467.84 versus full-history CatBoost466.23; pooled gain concentrates in early growth | Explain decomposition; intermediate surface error differs from transaction error |
| KNN / Random Forest | Old-feature sampled KNN539.58, RF493.77 | Different mechanisms/budgets; retain archives without repeating main-text comparisons |
| Public target matching | Annual known-label reconstruction181.25; 173.41 is an assumed error scale | Source audit only; exclude from forecasting rankings and target-achievement claims |

`model_catalog.csv` gives independent entries/original evidence for these routes. `branch_metric_tables.json` and `all_branch_scores.csv` index within-branch values, including grouped, calibration, pooled and intermediate-target errors; do not automatically rank the entire table by minimum RMSE.

## 6. Draft narrative for the report

> We first compare linear models, randomized ensembles, boosted trees and neural networks using27 shared inputs and fixed chronological validation. Ridge supplies a linear reference, ExtraTrees tests randomized ensembles, LightGBM/XGBoost/CatBoost compare boosting implementations, and Embedding MLP/FT test embeddings and feature attention. Under the limited search budget, tested trees generally outperform tested neural configurations; the finding applies to these data and budgets.

> We then fix LightGBM/CatBoost settings, iterations and samples while adding static attributes, ordinary histories, market-relative histories and anniversary cohorts. Both learners support historical/cohort information, while more elaborate windows, spatial and macro representations do not consistently improve. Finally, we blend complementary errors and distinguish earlier-locked recipes from later-selected development minima.

> These experiments use competition training and supplied auxiliary data. Repeatedly inspected development targets are not new independent tests. Final comparisons use all validation records within common chronological windows and SGD RMSE, with MAE, bias and long-horizon stress results to avoid highlighting only favorable scores.

## 7. Verification and reproduction

Independently checked31 new row-prediction files and ten existing common27-model files: IDs, months, labels, counts and recomputed metrics match. Reloaded18 Ridge/ladder models reproduce predictions; 31 tests pass and input hashes match history. GitHub results remain source-branch records; TabPFN/official TabM were not retrained locally. Archive checking is not complete rerunning.

```powershell
python -m analysis.experiments.report_model_catalog
python -m analysis.experiments.report_selection_trials ridge
python -m analysis.experiments.report_selection_trials ladder --stage outer
python -m analysis.experiments.report_selection_trials ladder --stage june
python -m analysis.experiments.report_selection_trials optimize --stage screen
python -m analysis.experiments.report_selection_blend select
python -m analysis.experiments.report_selection_blend outer
python -m analysis.experiments.report_selection_blend june
python -m analysis.experiments.report_selection_summary
```

Chart:[models/features](../03_model_comparison/model_feature_comparison.png). Machine-readable main table:`recommended_models.csv`; ladder:`controlled_feature_ladder.csv`; branches:`branch_summary.csv`; full catalog:`model_catalog.csv`.

## 8. Further exploration toward local RMSE 430


Completed 66 earlier-window configurations with identical validation rows/history cutoffs. Without future macro observations, the new blend's lowest recent RMSE 443.3215 does not beat442.9025.

Separately, supplied prediction-month COE/stock observations yield best recent 442.8892, under different information availability from prospective forecasting.

Target 430 remains unmet. User screenshots/local scores for three submissions are now in FINAL_MODEL.md; do not assume a fixed 20-point deduction. Details/negative results:[round 4](../04_optimization_history/04_macro_spatial_diagnostics.md).

### Submission-ready predictions

Refit150,000 training records and export50,000 queries: [history CSV](../../output/submissions/components/history_macro_ensemble/submission.csv), [supplied-macro CSV](../../output/submissions/components/history_macro_ensemble/submission_given_macro.csv). User submitted both; screenshots show415.86912/415.78232 versus local442.9025/442.8892. See [round 4 export](../../output/experiment_runs/round_04_macro_spatial_diagnostics/submission_export.json).


## 9. Further models and feature-group controls


Add three fixed learners×six inputs×three windows=54 fits, testing historical distributions and neighboring-building structure with anniversary-field deletions.

| Recent-window model | Original62 inputs | + Historical distribution | + Building structure | Both additions | Remove36-month fields | Retain24-month anniversary only |
|---|---:|---:|---:|---:|---:|---:|
| lgb | 447.1701 | 447.2156 | 447.1111 | 446.9913 | 447.3701 | 448.6101 |
| cat | 446.1608 | 446.2064 | 446.0444 | 446.1903 | 446.0275 | 447.3196 |
| xgb | 445.6576 | 445.8572 | 446.0440 | 445.9626 | 445.8147 | 447.5220 |

Fix settings, iterations, rows and cutoffs within learners; no validation stopping. Annual/early controls, negative results and boundaries:[round 5 feature report](../04_optimization_history/05_feature_interactions.md).

## 10. Further evaluation of stronger deep tabular models


Four new families and 19 earlier-window configurations: official TabM, numerical-embedding MLP, authors' standalone RealMLP and ModernNCA. Protocol:[round 6](../04_optimization_history/06_deep_models.md).

| Route | Annual RMSE | Recent RMSE |
|---|---:|---:|
| Official TabM | 456.1612 | 461.0924 |
| Authors' standalone RealMLP | 458.0752 | 448.4527 |
| MLP-PLR | 475.9378 | 448.0631 |
| Authors' ModernNCA | 461.3778 | 463.9001 |
| Original history blend | 451.1810 | 442.9025 |
| Blend with new deep models | 450.4924 | 444.6888 |

Earlier-locked new blend recent 444.6888 versus 442.9025; target 430 unmet. Later windows do not select epochs/weights but remain repeatedly inspected development data.

Retain common27-input baselines in the body, then separately compare deep models with enhanced histories and old/new blends. Extended tables hold every route/configuration without confusing different inputs with architecture effects.
Because recent RMSE does not improve, retain history blend442.9025 and existing submission CSVs. Report the annual gain separately rather than claiming universal improvement.

## 11. Updated Transformer routes


Add TabDPT1.3, Nori1.5/6M and strong-history TabICLv2. Protocol/all negative results:[round 7](../04_optimization_history/07_pretrained_transformers.md).

| Route | Annual RMSE | Recent RMSE |
|---|---:|---:|
| Frozen TabDPT1.3 | 462.9712 | 476.9916 |
| Supervised-adapted TabDPT1.3 | 464.1707 | 463.0065 |
| Nori | 459.5105 | 462.0088 |
| TabICLv2 | 460.6730 | 454.6275 |
| FT-Transformer (previous67-input policy) | 471.5338 | 458.6012 |
| Original history blend | 451.1810 | 442.9025 |
| Round 6 blend | 450.4924 | 444.6888 |
| Blend with new Transformers | 450.4823 | 442.9704 |

Target 430 remains unmet. Capacity, preprocessing/context budgets differ; reused local development results are not Kaggle scores. TabPFN-3.5 was not measured because model-license acceptance requires the user's login.

<!-- eighth-wave-details:start -->
## 12. Small improvements and actual training results

Retain even modest actual-training gains. Completed 65 development fits: 39 second-level residual models, nine CatBoost seeds, eight neural feature deletions, nine extra FT seeds. Also refit four FT models on 150,000 labels for 50,000 test predictions. Later training durations are fixed.

| Policy | Annual RMSE | Recent RMSE | Kaggle feedback |
|---|---:|---:|---|
| Original history blend | 451.1810 | 442.9025 | User reports about 415 |
| Original supplied-macro augmentation | 451.2081 | 442.8892 | User reports about 415; only decimal-level difference |
| **Pruned FT + four-seed replacement** | **451.4310** | **442.7032** | New file not yet submitted |
| New FT replacement + original macro weights | 451.5252 | 442.7710 | New file not yet submitted |

Change only the main blend's FT member: remove five36-month cohort fields, three fixed epochs, equal seeds 42/43/44/45, original 15.9872% leaf weight; retain all others. Recent28,995-row RMSE improves0.1993 but annual 35,158-row RMSE worsens0.2500. TabM pruning, second-level models and CatBoost seed averaging do not beat the recent reference; retain these negative results.

Five of ten recent months improve; descriptive month-block interval[-0.3017,0.0065] crosses0. No stable-gain claim; windows overlap. All52 saved development models replay and 17 tests pass. Four full refits also reload/replay; raw hashes remain unchanged.

On2026-11-05, user reports history/macro Kaggle scores near415 with decimal differences, normal-method best near411, and scores near170 described as cheating, excluded from valid comparisons. These are user reports without exact decimals/leaderboard partition; infer no fixed local-to-Kaggle gap. New candidates need same-board submission.

New submission candidates: try [history 442.7032](../../output/submissions/candidates/kaggle_history_ft_no36_mean.csv) first; also [macro 442.7710](../../output/submissions/candidates/kaggle_macro_ft_no36_mean.csv). Original files remain. Score labels identify temporal-validation experiments, not known scores on the 50,000 test rows.

Evidence:[round 8](../04_optimization_history/08_feature_pruning_seeds.md), [metrics](../../output/experiment_runs/round_08_feature_pruning_seeds/comparison.csv), [replay](../../output/experiment_runs/round_08_feature_pruning_seeds/verification.json), [Kaggle feedback](../../output/experiment_runs/round_08_feature_pruning_seeds/kaggle_feedback.json), [full refits/exports](../../output/submissions/components/round_08_ft_seed_average/summary.json).
<!-- eighth-wave-details:end -->

<!-- ninth-wave-details:start -->
## 13. Further results from changing only the blend

Reuse ten original leaf predictions plus round 8 four-seed FT mean, eleven channels total; no base-model retraining. Test22 initial and nine grouped-shrinkage configurations, 31 total, retaining complete controls/negative results.

| Blend policy | Annual RMSE | Recent RMSE |
|---|---:|---:|
| Original main blend | 451.1810 | 442.9025 |
| Round 8 fixed four-seed FT replacement | 451.4310 | 442.7032 |
| Global month-centered weights | 450.9816 | 442.2495 |
| 24-month-support grouping, weak shrinkage | 450.6221 | 442.0146 |
| **Joint24-/36-month support grouping, weak shrinkage** | **450.8650** | **441.9591** |

The lowest recent candidate has at most four24-/36-month support groups. Subtract each month's mean model error before fitting weights to reduce shared level effects. Weights are nonnegative, sum1, and each≤0.6. Global weights shrink toward the original blend, group weights toward global, both regularization0.003. Groups below500 fitting rows/unseen groups use global weights. Inference applies learned weights without query rents.

Annual weights use historical out-of-time predictions before 2024-04; June weights may add April–May forecasts from annual models, with labels through 2024-05. Versus round 8, ten/twelve annual months and all ten recent months improve; deltas−0.5660/−0.7441. Overlapping, repeated development windows are not independent tests.

Earlier inner selection chose `support24_mse`,later recent 442.7381. The441.9591 minimum follows later-result inspection and added shrinkage controls; it is not an earlier-locked independent-generalization result. Reloaded93 global/group blend JSONs reproduce full predictions; metrics/constraints checked; five tests pass.

Submission reuses base models trained on 150,000 labels and their50,000 test predictions; refit blend weights only on historical out-of-time labels through 2025-03. [New CSV](../../output/submissions/candidates/kaggle_blend_support_centered.csv) passes IDs, numerical/reload consistency and data hashes. Preserve originals; not uploaded, Kaggle score unknown.

Evidence:[round 9](../04_optimization_history/09_grouped_blending.md), [all configurations](../../output/experiment_runs/round_09_grouped_blending/comparison.csv), [development weights](../../output/experiment_runs/round_09_grouped_blending/weights_june.csv), [full-data weights](../../output/experiment_runs/round_09_grouped_blending/weights_full.csv), [replay checks](../../output/experiment_runs/round_09_grouped_blending/verification.json), [submission record](../../output/submissions/components/round_09_grouped_blend/summary.json).
<!-- ninth-wave-details:end -->

<!-- tenth-wave-details:start -->
## 14. Actual architecture training comparisons

Neural models share62 inputs: FT with linear/piecewise-linear/periodic numerical tokens, full-matrix Deep & Cross, and four/eight conditional experts. FT/embeddings use existing author packages; cross/expert models are project implementations. Nine earlier configurations select epochs/five routes, then fixed later durations: 19 neural fits. Two asymmetric CatBoost variants select one earlier for later evaluation: four fits. Their62 window inputs differ from neural62, so no cross-group fairness claim.

| Representative network | Annual RMSE | Recent RMSE |
|---|---:|---:|
| cat_symmetric_saved | 456.4077 | 446.3793 |
| cat_depthwise6 | 460.1803 | 448.0859 |
| ft_ple_128_3 | 458.2931 | 449.8158 |
| ft_plr_128_3 | 462.0714 | 451.0982 |
| dcn_plr_3 | 461.5278 | 452.1061 |
| ft_linear_96_2 | 461.9852 | 458.9254 |
| moe_plr_8 | 458.6529 | 478.9720 |

Fix round 9 objective, regularization and grouping; compare no new model, separate additions and all additions: nine inclusion routes. Baseline reproduces441.9591/450.8650.

| Inclusion route | Annual RMSE | Recent RMSE |
|---|---:|---:|
| only_ft_plr | 451.0295 | 441.8345 |
| baseline | 450.8650 | 441.9591 |
| only_tree | 450.8034 | 441.9752 |
| only_ft_ple | 450.7508 | 441.9845 |
| only_dcn | 450.8395 | 442.1377 |
| only_ft_linear | 450.7687 | 442.2048 |
| all_new | 450.8987 | 442.5100 |
| all_new_plus_tree | 450.8121 | 442.5193 |
| only_moe | 450.8200 | 442.8225 |

Lowest architecture-augmented blend recent 441.8345/annual 451.0295; user submitted, screenshot 415.37529, selected among three submissions.

Earlier neural choice was `all_new`; tree additions are extra exploration. Later minima come from reused development, not locked independent tests. Later base models do not stop on query labels; blend labels respect origins; windows overlap. Reloaded19 networks, four trees and 27 weight sets reproduce predictions; nine tests pass.

[Full architecture report](../04_optimization_history/10_plr_transformer_final.md), [standalone metrics](../../output/experiment_runs/round_10_plr_transformer_final/model_metrics.csv), [blend controls](../../output/experiment_runs/round_10_plr_transformer_final/blend_comparison.csv), [replay checks](../../output/experiment_runs/round_10_plr_transformer_final/verification.json).

[New structural-blend CSV](../../output/submissions/candidates/kaggle_structure_blend.csv),50,000 rows, user-submitted, screenshot Kaggle 415.37529.
<!-- tenth-wave-details:end -->
