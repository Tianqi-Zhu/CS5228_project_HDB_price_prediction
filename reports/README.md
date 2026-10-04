# Reports index

报告按用途分成三类，避免把描述性研究和预测模型分数混在一起比较。

| 目录 | 内容 | 是否可按 RMSE 排名 |
| --- | --- | --- |
| [`eda/`](eda/) | 主数据、辅助数据审计和阶段 PDF | 否；其中基线仅用于描述 |
| [`studies/`](studies/) | 时间信号、分组面板和空间控制研究 | 否；主要解释关联和稳定性 |
| [`experiments/`](experiments/) | 固定验证规则下的预测模型实验 | 仅在验证窗口和目标一致时 |

大型模型、逐行预测、提交文件和训练历史默认不进入 Git。每个实验目录保留小型
`results.json`、指标 CSV、摘要或必要图表。不同验证窗口的分数不能直接排序，具体
冲突检查见 [`docs/result_consistency.md`](../docs/result_consistency.md)。
