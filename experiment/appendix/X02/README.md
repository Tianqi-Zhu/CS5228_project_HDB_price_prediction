# X02 清洗敏感性与极端标签

当前状态：**archived**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 低租金或重复输入是否应删除？ |
| 对照与口径 | MLP/TabM/hybrid 各与自己的未清洗版本比较，保留全部验证行；ReLU/IQR 另有协议。 |
| 观察 | 轻度清洗只改善部分模型；更强删除未稳定降低 RMSE，不能凭低租金就断定标签错误。 |
| 取舍依据 | 主线保留全部记录，附录记录删除代价、实际行数与独立校准设置。 |
| 服务 main | M01/M04；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [comparison_metrics.csv](../X06/mlp_explain/cleaning/comparison_metrics.csv)
- [protocol.json](../X06/mlp_explain/cleaning/protocol.json)
- [README.md](relu_historical/README.md)
