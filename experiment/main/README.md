# main 实验索引

[执行状态与缺口](../../docs/planning/main-execution-status.md) · [本轮结果摘要](evaluation-summary.json) · [要求卡](../../docs/planning/main-appendix-experiment-plan.md)

| ID | 当前状态 | 内容 |
|---|---|---|
| [M00](M00/README.md) | partial | 26 组预测身份、行 ID、schema 和指标审计；精确旧生成提交/F2 仍缺失 |
| [M01](M01/README.md) | completed for snapshot | 共同 EDA 与预处理政策；连接不增行；设施时间按快照解释 |
| [M02](M02/README.md) | archived/verified | C7 均值、Ridge、RandomForest、LightGBM、CatBoost 基线 |
| [M03](M03/README.md) | completed | 两窗固定模型 19/27/47/62 特征阶梯与配对增量 |
| [M04](M04/README.md) | completed | F1 与两个 62 列控制的月份、户型、覆盖、尾部和相同输入诊断 |
| [M05](M05/README.md) | completed within recorded scope | 27 列架构对照；62 列 PLE/PLR 和固定规则融合增量；Concat 独立探索 |
| [M06](M06/README.md) | archived/verified | 三个既有提交中选取 F1；其他版本保留在 appendix |
| [M07](M07/README.md) | partial | TabPFN 背景可查，实际 F2 生成链继续待补 |
| [M08](M08/README.md) | pending F2 | 两个最终管线的同行配对评价尚不能执行 |
| [M09](M09/README.md) | partial | F1 verify/CPU 重组通过；F2 独立复现未完成 |

每张卡的 `evaluation/` 保存本轮新分析，原有目录保存此前实验。相同数据、配置和协议内才解释配对增量；Y/J 不作为两个独立测试。完整 main P0 尚未全部完成，缺口按卡列明。
