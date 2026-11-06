# X08 设施时间/来源与特征敏感性

当前状态：**partial**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | STATUS=open 是否能证明历史时点已开通？ |
| 对照与口径 | 来源快照坐标去重，与仅排除JRL/CRL-only坐标的特征级对照；没有修改旧模型输入。 |
| 观察 | 34个坐标；最近站点受影响训练20557/测试7153行，复现原审计；距离增量并非RMSE增益。 |
| 取舍依据 | 采用明确快照场景；缺完整开通日期，严格历史重训保持未完成，不认证剩余站点。 |
| 服务 main | M01/M08；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [availability_audit.json](evaluation/availability_audit.json)
- [feature_sensitivity.csv](evaluation/feature_sensitivity.csv)
