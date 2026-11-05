# HDB 月租预测实验

预测单条 HDB 的 `MONTHLY_RENT`（SGD）。`main` 保存必要对照与最终方法，`appendix` 保存替代方案、消融、负结果和完整探索。比赛原始数据、环境文件和项目架构均保留。

| 从这里开始 | 位置 |
|---|---|
| 实验索引与当前状态 | [M00–M09](experiment/main/README.md) |
| 新增要求卡的执行结果与缺口 | [执行状态](docs/planning/main-execution-status.md) |
| 原实验要求卡、main/appendix 规划 | [统一规划](docs/planning/main-appendix-experiment-plan.md) |
| 环境、训练与离线验证命令 | [运行指南](docs/guides/running_experiments.md) |
| 代码结构 | [analysis/](analysis/README.md) |
| 课程评分依据 | [SKILL.md](skills/cs5228-final-report/SKILL.md) |

```powershell
python review.py list
python review.py verify
python -m analysis.main planned
```

`planned` 重算已冻结开发预测的指标、对齐、分组误差与结构对照，不训练、不搜索；结果写入 `output/planning_validation/main/<内容摘要>/`。已有 run 不覆盖，重复运行可指定新的 `--output-dir output/planning_validation/recheck`。所有 Y/J 结果均为已查看过的开发期证据。

| 方法 | 状态与证据 | 入口 |
|---|---|---|
| F1：分组融合＋PLR-FT | 原提交显示 415.37529；年度/近期开发 RMSE 451.0295/441.8345；冻结 CSV、组件重组与全量训练路径独立保留 | [F1 复现包](reproduction/f1/README.md) |
| Concat Transformer | 独立同输入架构探索，年度开发 RMSE 459.2189；原团队记录 Public Score 418.38657；保留他人合并的代码、容量比较、权重与重放入口 | [方法](experiment/main/M05/concat_transformer/README.md)、[复现](reproduction/concat_transformer/README.md) |
| F2：TabPFN87 相关提交 | 截图显示 413.19635；实际生成器、schema、成员、CSV 和配对验证仍待补。现有 77 列背景已归档，不能替代 F2 身份 | [状态](experiment/main/M07/README.md)、[TabPFN 工作流](docs/guides/tabpfn_workflow.md) |

完整性检查用 `python -m analysis.validation.repository_audit`，实际验证范围和日志见 [review/CHECKS.md](review/CHECKS.md)。截图分数不作为新运行结果；不同信息条件、窗口和 schema 分开比较。

`data/` 保存 150,000 行训练记录、50,000 行测试记录、提交模板及六份辅助表。`analysis/` 是共享实现；`reproduction/` 保存固定模型与配方；`workflows/tabpfn/` 保存隔离工作流；`experiment/main/` 保存科学证据。可重建缓存、训练输出和大型临时目录统一在忽略的 `output/` 或 `.cache/`。
