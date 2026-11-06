# X09 预测跨度与共同月份边界

当前状态：**partial**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 近期10个月验证能说明全部查询跨度么？ |
| 对照与口径 | 按原始日期审计全量上下文，保留既有独立长窗参考；F1/F2同2023-12原点L仍须重拟合。 |
| 观察 | 380条查询与最新标签同月，18077条查询超过已配对的10个月范围；并无这些行的真实标签。 |
| 取舍依据 | 只报告样本覆盖与未验证边界；不能从Y/J删月制造新L，不能把重叠窗口当独立测试。 |
| 服务 main | M08；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [query_horizons.csv](evaluation/query_horizons.csv)
- [cutoff_checks.json](evaluation/cutoff_checks.json)
- [existing_multi_window_reference](existing_multi_window_reference)
