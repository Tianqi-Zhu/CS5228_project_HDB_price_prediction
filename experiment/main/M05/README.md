# M05 实验审查卡

| 字段 | 内容 |
|---|---|
| 归属 | main |
| 目的 | 共同输入神经对照与PLR固定融合增量 |
| 归档状态 | archived（已有归档不表示本轮新训练） |
| 完整要求/对照/验收 | docs/planning/main-appendix-experiment-plan.md中同ID要求卡 |
| 既有证据 | 本目录内文件；文件来源/哈希见review/manifest.json；无文件则尚未实施 |
| 执行状态 | 历史证据按本卡归档；本轮实际运行范围另见 review/checks/push-validation/summary.json，不代表重跑全部历史搜索 |
| 复用限制 | 不同窗口、schema、宏观可用性、选择阶段分开比较；负结果不删除 |

## Independent concat Transformer architecture exploration

See [method, capacity selection and results](concat_transformer/README.md). It reuses the final PLR member's adjusted_history_no36_62 schema and adjusted residual target. Validation459.2189; final all-label submission Public418.38657. These supplement the controlled architecture discussion and do not replace F1/F2.
