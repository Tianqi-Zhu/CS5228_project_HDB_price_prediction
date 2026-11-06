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
```

`planned` 重算已冻结开发预测的指标、对齐、分组误差与结构对照，并核验新增的 F1 Y/J 生成链、统一特征清单及 M01 来源证据，不训练、不搜索；结果写入 `output/planning_validation/main/<内容摘要>/`。三个专项命令也可单独运行。已有 run 不覆盖，重复运行可指定新的 `--output-dir output/planning_validation/recheck`。所有 Y/J 结果均为已查看过的开发期证据。

| 方法 | 状态与证据 | 入口 |
|---|---|---|
| F1：分组融合＋PLR-FT | 原提交显示 415.37529；年度/近期开发 RMSE 451.0295/441.8345；冻结 CSV、组件重组与全量训练路径独立保留 | [F1 复现包](reproduction/f1/README.md) |
| Concat Transformer | 独立同输入架构探索，年度开发 RMSE 459.2189；原团队记录 Public Score 418.38657；保留他人合并的代码、容量比较、权重与重放入口 | [方法](experiment/main/M05/concat_transformer/README.md)、[复现](reproduction/concat_transformer/README.md) |
| MLP独立集成 | 原全量清洗／未清洗提交显示437.34455／439.38193；CPU训练、预测区间、时间加权及测试入口已接入；不改变F1配方 | [MLP复现包](reproduction/mlp/README.md)、[报告](docs/mlp_prediction_report.md) |
| F2：TabPFN87 相关提交 | 截图显示 413.19635；候选生成器、schema、成员及 CSV 已合并；与该截图分数对应的文件身份及配对确认仍待核实。现有 77 列背景不能替代 F2 身份 | [状态](experiment/main/M07/README.md)、[TabPFN 工作流](docs/guides/tabpfn_workflow.md) |

完整性检查用 `python -m analysis.validation.repository_audit`，实际验证范围和日志见 [review/CHECKS.md](review/CHECKS.md)。截图分数不作为新运行结果；不同信息条件、窗口和 schema 分开比较。

`data/` 保存 150,000 行训练记录、50,000 行测试记录、提交模板及六份辅助表。`analysis/` 是共享实现；`reproduction/` 保存固定模型与配方；`workflows/tabpfn/` 保存隔离工作流；`experiment/main/` 保存科学证据。可重建缓存、训练输出和大型临时目录统一在忽略的 `output/` 或 `.cache/`。

## Latest TabPFN anniversary merge

The latest anniversary branch is integrated through the existing isolated workflow. Main's canonical training/review entry points and frozen F1/Concat evidence remain in place.

- [Merged workflow, source mapping, and validation boundaries](docs/guides/anniversary_merge.md)
- [Final report PDF and editable source](reports/final_report/README.md)
- Run `python review_tabpfn.py verify`, `python review_tabpfn.py prepare`, and `python review_tabpfn.py test` to inspect and test the imported sources without overlaying main's `analysis` package.

The 87-input candidate export and its source/provenance are now included. Association with the historical 413.19635 screenshot score remains unverified; this merge does not assign that score to a newly validated model.

## MLP完整接入

`python -m analysis.main mlp verify`核对MLP提交与数据；`python -m analysis.main mlp fullfit --output-dir output/mlp/new-run`从原始数据重训。完整命令和环境见[MLP入口](reproduction/mlp/README.md)。MLP的历史完整探索保留在源分支，本main提供所需实现、配置、原提交和加权对照。
