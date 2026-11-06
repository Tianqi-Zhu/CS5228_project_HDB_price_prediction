# HDB 月租预测实验

预测单条 HDB 的 `MONTHLY_RENT`（SGD）。`main` 保存必要对照与最终方法，`appendix` 保存替代方案、消融、负结果和完整探索。比赛原始数据、环境文件和项目架构均保留。

| 从这里开始 | 位置 |
|---|---|
| 实验索引与当前状态 | [M00–M09](experiment/main/README.md) |
| 新增要求卡的执行结果与缺口 | [执行状态](docs/planning/main-execution-status.md) |
| 五项补齐与本次验收 | [五项优化](docs/planning/main-five-item-completion.md) |
| 预测情景与信息可用边界 | [场景定义](docs/methods/prediction-scenario.md) |
| 原实验要求卡、main/appendix 规划 | [统一规划](docs/planning/main-appendix-experiment-plan.md) |
| 环境、训练与离线验证命令 | [运行指南](docs/guides/running_experiments.md) |
| 代码结构 | [analysis/](analysis/README.md) |
| 目录职责与中间产物清理 | [仓库结构](docs/guides/repository_layout.md) |
| 课程评分依据 | [SKILL.md](skills/cs5228-final-report/SKILL.md) |

```powershell
python review.py list
python review.py verify
python -m analysis.main planned
python -m analysis.main schemas
python -m analysis.main evidence
python -m analysis.main validation-refits
python -m analysis.main final-pair
```

`planned` 重算冻结开发预测、分组误差和结构对照，核验 F1 Y/J 生成链、特征清单、M01 来源，并完成 F2 候选组件重组与近期配对评价，不训练、不搜索。`final-pair` 可单独执行最后一项。新结果统一进入忽略的 `output/`，已有 run 不覆盖；重复运行需使用新的 `--output-dir`。Y/J 均为已查看过的开发期。

| 方法 | 状态与证据 | 入口 |
|---|---|---|
| F1：分组融合＋PLR-FT | 原提交显示 415.37529；年度/近期开发 RMSE 451.0295/441.8345；冻结 CSV、组件重组与全量训练路径独立保留 | [F1 复现包](reproduction/f1/README.md) |
| Concat Transformer | 独立同输入架构探索，年度开发 RMSE 459.2189；原团队记录 Public Score 418.38657；保留他人合并的代码、容量比较、权重与重放入口 | [方法](experiment/main/M05/concat_transformer/README.md)、[复现](reproduction/concat_transformer/README.md) |
| MLP独立集成 | 原全量清洗／未清洗提交显示437.34455／439.38193；CPU训练、预测区间、时间加权及测试入口已接入；不改变F1配方 | [MLP复现包](reproduction/mlp/README.md)、[报告](docs/mlp_prediction_report.md) |
| TabPFN87 单模型 | 近期 RMSE 441.9593；远端已记录独立 fit/inference 验证；本机仍需 SDK 与原 checkpoint | [独立运行](docs/guides/tabpfn_best.md) |
| F2：七树＋TabPFN87 等权融合 | 近期 RMSE 440.9520；50,000 行候选组件重组及 28,995 行 F1/F2 配对可离线核验。年度结果、完整有序字典和截图 413.19635 的关联仍待补 | [F2 状态](experiment/main/M07/README.md)、[配对结果](experiment/main/M08/README.md) |

完整性检查用 `python -m analysis.validation.repository_audit`，实际验证范围和日志见 [review/CHECKS.md](review/CHECKS.md)。截图分数不作为新运行结果；不同信息条件、窗口和 schema 分开比较。

`data/` 保存 150,000 行训练记录、50,000 行测试记录、提交模板及六份辅助表。`analysis/` 是共享实现；`reproduction/` 保存固定模型与配方；`workflows/tabpfn/` 保存隔离工作流；`experiment/main/` 保存科学证据。可重建缓存、训练输出和大型临时目录统一在忽略的 `output/` 或 `.cache/`。

## 独立工作流

| 工作流 | 不训练的核验入口 | 实际训练或推理 |
|---|---|---|
| F1 | `python reproduction/f1/verify.py` | [固定配方](reproduction/f1/README.md) |
| Concat | `python -m analysis.main concat verify` | [原模型重放与训练](reproduction/concat_transformer/README.md) |
| MLP | `python -m analysis.main mlp verify` | [CPU 全量、区间、时间加权](reproduction/mlp/README.md) |
| TabPFN | `python review_tabpfn.py verify`；`python run_tabpfn_best.py preflight` | [SDK/checkpoint 前提与独立运行](docs/guides/tabpfn_best.md) |

各工作流使用同一份根 `data/`，隔离的运行时不能相互覆盖。完整探索和历史来源保留，不将 MLP 或 Concat 直接加入 F1/F2。TabPFN 合并记录见[来源与边界](docs/guides/anniversary_merge.md)。已有[报告材料](reports/final_report/README.md)仅保留，本轮不继续改稿。
