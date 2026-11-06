[Chinese version](README.zh-CN.md)

# Model comparisons

Compare models on 27 shared inputs before enhanced features/blends; these answer separate questions.

| Content | File |
| --- | --- |
| RMSE, MAE and bias for common27-input models | [same_input_models.csv](same_input_models.csv) |
| Shared inputs/selection protocol | [PROTOCOL.md](PROTOCOL.md) |
| Models recommended for the report | [recommended_models.csv](recommended_models.csv) |
| Complete routes/provenance | [model_catalog.csv](model_catalog.csv) |
| Model/feature comparison plot | [model_feature_comparison.png](model_feature_comparison.png) |
| Recommended set/main narrative | [Full experiment report](../01_final/EXPERIMENT_REPORT.md) |
| Complete-pipeline/submission comparison | [Final method](../01_final/README.md) |

Retain linear, randomized-tree, boosted-tree, Embedding MLP and FT references,
then separately compare history/macro/structural complete pipelines.
Network complexity does not guarantee lower RMSE; blend gains need fixed-rule controls.

Tuning, window fitting and full searches:`output/experiment_runs/comparison_*/`.
