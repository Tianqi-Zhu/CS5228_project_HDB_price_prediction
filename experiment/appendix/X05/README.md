# X05 时间序列与动态租金面替代路线

当前状态：**archived**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 把市场动态压缩成 latent 或 PCA 租金面能替代现有历史表示么？ |
| 对照与口径 | RNN/LSTM 对同记录的无latent基线，月更新；动态面固定原点，并为丢失2021拟合行设置2022+ CatBoost控制。 |
| 观察 | RNN/LSTM 未稳定改善匹配基线；PCA历史重建好不等于未来面预测好，残差hybrid增益随折变化。 |
| 取舍依据 | 作为替代表示和失败/条件增益归档；月更新、面诊断、逐行固定原点三种成绩不混排。 |
| 服务 main | M03/M05；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [rnn_lstm_report.md](external_studies/rnn_lstm/docs/rnn_lstm_report.md)
- [report.md](external_studies/dynamic_surface/reports/factored_rental_surface/report.md)
- [README.md](concat_explorations/README.md)

## Ziyuan concat exploration archive

See [methods and evidence](concat_explorations/README.md) for the newly imported local experiments. This addition archives historical source/results; it does not mark every planned experiment complete.
