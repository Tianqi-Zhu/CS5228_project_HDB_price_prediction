[Chinese version](README.zh-CN.md)

# Ten optimization rounds

Read in order; one summary per round. Full runs:`output/experiment_runs/round_*/`.

| Round | Main question | Report |
| --- | --- | --- |
| 01 | Distribution histories, target decomposition, space/horizons | [History and targets](01_history_target_search.md) |
| 02 | Whether anniversary records capture stable rent patterns | [Anniversary histories](02_anniversary_history.md) |
| 03 | Whether trees/Transformers complement each other | [Heterogeneous blending](03_tree_transformer_ensemble.md) |
| 04 | Remaining macro/spatial/error-diagnostic gains | [Macro and space](04_macro_spatial_diagnostics.md) |
| 05 | Whether feature additions help across learners | [Feature/model interactions](05_feature_interactions.md) |
| 06 | Whether stronger deep tabular structures help | [Deep models](06_deep_models.md) |
| 07 | How newer pretrained Transformers perform | [Newer Transformers](07_pretrained_transformers.md) |
| 08 | Sparse-field pruning/seed averaging | [Feature pruning and seeds](08_feature_pruning_seeds.md) |
| 09 | Blend weights conditioned on historical support | [Grouped blending](09_grouped_blending.md) |
| 10 | Whether PLR/other structures complement the blend | [Final architecture controls](10_plr_transformer_final.md) |

Preserve each round's contemporary development conclusions/negative results; submission status follows [final conclusions](../01_final/README.md).
Round 9 annual 450.8650 beats final 451.0295; final recent 441.8345 is lower and wins among three screenshot submissions.
Do not claim the final is lowest on every local window.

Reused windows supply development evidence. Counts/checks record historical runs, not complete retraining during reorganization.
