[Chinese version](main-five-item-completion.zh-CN.md)

# Main five-item improvement and acceptance record

This page preserves the original five-item acceptance record. Later main integrates F2 candidates/MLP materials; follow [current status](main-execution-status.md) and new M07–M09 checks rather than old "missing generator" statements below.

This round completes experiment materials/commands, not manuscript prose. Raw data, original submissions, integrated Concat architecture and others' work are retained; F2 remains pending as previously decided.

| Item | Completed work | Evidence / entry point |
|---|---|---|
| 1. Common feature definitions | Register C7,19/27/47/62, adjusted62, F1 heterogeneous members and raw/encoded Concat inputs separately; restore five fields omitted in old PLE/PLR metadata | [Schema inventory](../../experiment/main/M03/feature_schemas/README.md); `python -m analysis.main schemas` |
| 2. F1 Y/J generation chains | Train30 new fixed-recipe base models from raw data, save/reload all, and reconstruct pre-origin weights without new parameter/weight selection | [New generation chain](../../reproduction/f1/README.md), [acceptance table](../../reproduction/f1/validation_evidence/validation_results.csv); `python -m analysis.main validation-refits` |
| 3. Prediction scenario | Explicit supplied-snapshot/fixed-history-origin main task; separately disclose fitting/query boundaries, same-month test rows and macro usage | [Scenario](../methods/prediction-scenario.md); `planned` checks raw counts/months |
| 4. Reuse spatial EDA | Import frozen sources; select six raw/adjusted associations and 18 controlled facility-bin comparisons; distinguish two adjustment models and historical availability | [Spatial evidence](../../experiment/main/M01/spatial_evidence/README.md) |
| 5. Reuse cleaning controls | Import nine matched summaries; independently recompute three row-level predictions; retain slight cleaned-MLP gains and contrary results | [Cleaning evidence](../../experiment/main/M01/cleaning_evidence/README.md); `python -m analysis.main evidence` |

## New-fit identity and results

| Window | Fitting / query rows | Base models / logical components | RMSE (SGD) | MAE (SGD) | Recorded end-to-end time |
|---|---|---|---|---|---|
| Y | 114,842 / 35,158 | 15 / 12 | 451.029502492 | 328.729828598 | 384.97 seconds |
| J | 121,005 / 28,995 | 15 / 12 | 441.834534884 | 321.820758266 | 391.51 seconds |

Generator commit is `29ea4ac`; full IDs are in each window's `generating_commit`. All39 training sources per window match commit objects/current bytes. Maximum difference from old F1 predictions is about `9.09e-13` SGD in both windows; RMSE matches. New CSV bytes reconstruct from archived components. Preserve new-run and old-cache provenance separately; the new commit is not the old cache's generator.

Large models, feature caches and complete new training CSVs stay in ignored `output/five_item_validation/`. Compact component arrays, run records and hashes are tracked for CPU review; the CUDA fixed recipe retrains from raw data.

## Structure and duplicate-content handling

| Content | Primary entry point | Action |
|---|---|---|
| Data | Root `data/` | Do not duplicate nine raw CSVs or change their newline bytes |
| Scenario | `docs/methods/prediction-scenario.json` | Markdown explains; audits read JSON and check sample scope |
| Features | `M03/feature_schemas/` | Ordered columns/transforms; retain old metadata with explicit corrections |
| Provenance | `M01/imported-evidence.json` | Original branch, commit, path and hash; verification regenerates selected main tables |
| Training | `reproduction/f1/validate.py` | Reuse frozen runtime without a second model implementation or data copy |
| Offline acceptance | `python -m analysis.main planned` | Integrate three focused checks; standalone commands help locate issues |
| Execution records | [review/CHECKS.md](../../review/CHECKS.md) | Retain history; separate new acceptance directories |

## Remaining boundaries

| Item | Status and implications |
|---|---|
| F2 | Actual generator, 87-input schema, member weights, submission CSV and Y/J paired predictions remain outstanding; corresponding M07/M08/M09 work is incomplete |
| Identity of old generators | New F1 generation chains exist; exact original commits for all 26 old cached groups cannot yet be independently recovered |
| Independence | Y/J overlap and were repeatedly inspected; fixed-recipe refits do not create an independent test |
| Spatial statistics | Source bytes, input versions and definitions verified; original regressions not independently reestimated this round; facility associations are not causal |
| Cleaning evidence | Three row-level recomputations; the other six verify source summaries only. Different fitting cutoffs prevent treating these as F1 cleaning gains. |
| Execution scope | New Y/J pipelines ran end to end; existing-model replay/tests have separate acceptance. No claim of rerunning every search, full eight-epoch Concat training or fresh Conda installation. |

The five items are independently reviewable, but missing F2 evidence prevents declaring all two-endpoint P0 work complete.
