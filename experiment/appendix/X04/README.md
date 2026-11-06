# X04 种子、覆盖分组与结构负结果

当前状态：**archived**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 复杂结构单独好、加入融合就一定好么？ |
| 对照与口径 | 第8/9/10轮的固定规则、不同成员集合、早期锁定和后期探索分开；内部bagging不作外部重复。 |
| 观察 | PLR 相对无新增对照年度 RMSE +0.164455、近期 -0.124548；同一模型可给出窗口依赖的结论。 |
| 取舍依据 | 保留年度更好的无新增/PLE对照和负结果；禁止只报告挑中的最终分数。 |
| 服务 main | M04/M05；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [blend_comparison.csv](round_10_plr_transformer_final/blend_comparison.csv)
- [README.md](round_09_grouped_blending/README.md)
