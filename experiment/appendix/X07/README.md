# X07 匹配外部种子鲁棒性

当前状态：**planned**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 两终点的小幅差距能否跨完整训练随机种子保持？ |
| 对照与口径 | F1/F2 配方冻结，Y/J各42/43/44；已有合格42可复用，43/44不重选超参和权重规则。 |
| 观察 | 当前不存在这组匹配完整重复。子成员四种子平均和 MLP 自身多种子都不能替代。 |
| 取舍依据 | 保留为后续计算任务；需F2年度原点资料、checkpoint、明确GPU时间/显存，再做完整重复。 |
| 服务 main | M08；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [appendix-execution-status.md](../../../docs/planning/appendix-execution-status.md)
