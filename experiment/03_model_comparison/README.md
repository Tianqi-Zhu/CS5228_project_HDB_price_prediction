# 模型对比

先比较同一组27个输入下的模型，再讨论增强特征与融合，避免把两种问题混在一起。

| 内容 | 文件 |
| --- | --- |
| 同27输入模型的RMSE、MAE和偏差 | [same_input_models.csv](same_input_models.csv) |
| 共同输入与选择协议 | [PROTOCOL.md](PROTOCOL.md) |
| 报告推荐保留的模型 | [recommended_models.csv](recommended_models.csv) |
| 完整模型路线与来源 | [model_catalog.csv](model_catalog.csv) |
| 模型与特征对照图 | [model_feature_comparison.png](model_feature_comparison.png) |
| 推荐组与正文解释 | [完整实验报告](../01_final/EXPERIMENT_REPORT.md) |
| 完整融合与提交比较 | [最终方案](../01_final/README.md) |

建议保留线性参照、随机化树、提升树、Embedding MLP与FT-Transformer，
再单独对照历史／宏观／结构融合三条完整管线。
单个网络更复杂并不保证更低RMSE；融合收益需要结合固定规则的对照解释。

调参、分窗口训练与完整模型搜索记录在 `output/experiment_runs/comparison_*/`。
