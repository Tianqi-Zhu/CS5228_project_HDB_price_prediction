# 特征工程：从基础属性到历史同期信息

先看[基础27列阶段报告](BASELINE_27_FEATURES.md)，再用固定模型的受控阶梯解释增强信息的作用。

| 内容 | 文件 |
| --- | --- |
| 基础清理、特征假设与27列方案 | [BASELINE_27_FEATURES.md](BASELINE_27_FEATURES.md) |
| 基础27列字段定义与来源 | [feature_dictionary.csv](feature_dictionary.csv) |
| 删除单列并重训 | [FEATURE_ABLATION.md](FEATURE_ABLATION.md)、[single_feature_effects.csv](single_feature_effects.csv) |
| 删除整组并重训 | [group_feature_effects.csv](group_feature_effects.csv) |
| 固定模型比较19／27／47／62列 | [controlled_feature_ladder.csv](controlled_feature_ladder.csv) |
| 阶梯的配置和结果解释 | [完整报告第2节](../01_final/EXPERIMENT_REPORT.md#2-特征工程用固定模型与固定轮数讲清楚) |

19列覆盖房屋、时间、区位和设施；27列增加过去租金统计；47列加入相对市场历史；
62列加入12／24／36个月同期统计。不同表格的训练配置分别注明，不能跨表直接归因。

删除重训的增量是条件预测作用，不能相加或称为因果贡献。
历史特征遵守信息截止点，验证期不接收真实租金更新。
完整逐折指标、参数和误差诊断在 `output/experiment_runs/features_*/`。
