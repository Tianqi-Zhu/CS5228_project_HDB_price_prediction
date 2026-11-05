# 十轮优化历程

按顺序阅读，每轮只保留一份汇总报告；全部配置与运行明细在 `output/experiment_runs/round_*/`。

| 轮次 | 主要问题 | 阅读报告 |
| --- | --- | --- |
| 01 | 历史分布、目标分解、空间与跨度 | [历史与目标表示](01_history_target_search.md) |
| 02 | 同期记录能否识别稳定租金模式 | [同期历史](02_anniversary_history.md) |
| 03 | 树与Transformer是否互补 | [异质模型融合](03_tree_transformer_ensemble.md) |
| 04 | 宏观、空间与误差诊断还能改善多少 | [宏观与空间](04_macro_spatial_diagnostics.md) |
| 05 | 特征扩展是否在不同模型上一致有效 | [特征与模型交互](05_feature_interactions.md) |
| 06 | 更强深度表格结构是否有效 | [深度模型](06_deep_models.md) |
| 07 | 较新预训练Transformer表现如何 | [较新Transformer](07_pretrained_transformers.md) |
| 08 | 稀疏字段删减与种子平均 | [特征删减与多种子](08_feature_pruning_seeds.md) |
| 09 | 按历史覆盖情况学习融合权重 | [分组融合](09_grouped_blending.md) |
| 10 | PLR数值嵌入等结构能否提供互补 | [最终结构对照](10_plr_transformer_final.md) |

各轮保留当时的开发结论和负结果，旧提交状态以[最终结论](../01_final/README.md)为准。
第九轮年度450.8650低于最终方案的451.0295；最终方案近期441.8345更低，并在三份截图提交中胜出。
不能把最终方案说成所有本地窗口都最低。

这些窗口曾被反复查看，属于开发证据。各轮数量和核验记录是当时运行的记录，不等于目录整理时全部重训。
