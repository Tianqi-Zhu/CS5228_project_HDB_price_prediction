# X06 神经模型、基础模型与误差解释

当前状态：**archived**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 不同深度方法在各自实际输入和预算下提供什么证据？ |
| 对照与口径 | MLP原协议、Concat同输入架构对照、TabPFN各自匹配实验分别阅读；官方TabM与自定义style分开命名。 |
| 观察 | MLP 有低分歧却高误差的案例；Concat/TabPFN探索包含预算与输入限制，不能汇总成跨协议架构排名。 |
| 取舍依据 | 保留远端已有的MLP三组图、原始逐行预测和运行入口；模型优点必须限定于实际对照。 |
| 服务 main | M04/M05；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [README.md](mlp_explain/README.md)
- [README.md](concat_explorations/README.md)
- [README.md](../T05/README.md)

## Ziyuan concat exploration archive

See [methods and evidence](concat_explorations/README.md) for the newly imported local experiments. This addition archives historical source/results; it does not mark every planned experiment complete.

## MLP解释性实验

[报告、逐行结果与运行入口](mlp_explain/README.md)：误差分组、相同输入、confidence、清洗、风险修正和历史深度实验。
