# main 实验索引

[执行状态与缺口](../../docs/planning/main-execution-status.md) · [本轮结果摘要](evaluation-summary.json) · [要求卡](../../docs/planning/main-appendix-experiment-plan.md)

| ID | 当前状态 | 内容 |
|---|---|---|
| [M00](M00/README.md) | F1 generation and F2 recent identities verified | 旧 26 组预测审计；F1 固定生成链；F2 行 ID/标签与候选组件来源核验 |
| [M01](M01/README.md) | completed for snapshot | 共同 EDA、空间受控关联、清洗匹配对照；显式快照场景 |
| [M02](M02/README.md) | archived/verified | C7 均值、Ridge、RandomForest、LightGBM、CatBoost 基线 |
| [M03](M03/README.md) | completed | 特征阶梯与配对增量；有序 schema、异质组件及两套 62 列输入 |
| [M04](M04/README.md) | completed | F1 与两个 62 列控制的月份、户型、覆盖、尾部和相同输入诊断 |
| [M05](M05/README.md) | completed within recorded scope | 27 列架构对照；62 列 PLE/PLR 和固定规则融合增量；Concat 独立探索 |
| [M06](M06/README.md) | archived/verified | 三个既有提交中选取 F1；其他版本保留在 appendix |
| [M07](M07/README.md) | candidate reconstructed; partial identity | 七树＋TabPFN87 等权公式和 CSV 可核验；完整有序字典、截图关联待补 |
| [M08](M08/README.md) | J complete; Y pending | 近期 28,995 行同行配对；年度预测和完整信息政策等价审计待补 |
| [M09](M09/README.md) | component replay passed | F1 verify/CPU 重组与生成链通过；F2 组件重组通过，未在本轮新跑完整 GPU 推理 |

每张卡的 `evaluation/` 保存本轮新分析，原有目录保存此前实验。相同数据、配置和协议内才解释配对增量；Y/J 不作为两个独立测试。完整 main P0 尚未全部完成，缺口按卡列明。

MLP 全量清洗／未清洗提交、时间加权和区间流程通过 [M05 的独立入口](M05/README.md)导航，不作为第三个最终终点。运行 `python -m analysis.main final-pair` 可单独审查 F1/F2 近期配对和 F2 组件身份。
