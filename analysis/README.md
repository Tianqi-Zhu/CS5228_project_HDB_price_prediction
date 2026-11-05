# 代码导航

| 想看什么 | 位置 |
| --- | --- |
| 数据、特征、训练与预测的共享实现 | `pipeline/` |
| 模型与特征实验 | [experiments/](experiments/README.md) |
| 主表和辅助数据分析 | `eda/` |
| 经济、空间与面板研究 | `studies/` |
| 基础命令与训练入口 | `main.py` |
| 不执行训练的实验索引 | `experiment_catalog.py` |
| 从既有结果更新阅读报告 | `report_bundle.py` |
| 基础27列阶段汇总生成器 | `project_review.py` |
| 已保存结果的一致性核查 | `verification.py` |
| 按任务检查缺失资源与包版本来源 | `resource_preflight.py` |
| 关联已保存权重、融合参数与指标 | `artifact_catalog.py` |

```powershell
python -m analysis.main experiment-list
python -m analysis.main experiment-list --group wave10
python -m analysis.main experiment-refresh
python -m analysis.main preflight --profile final
```

实验代码按用途和轮次分组，旧模块名与命令仍可使用。
阅读汇总在[experiment/](../experiment/README.md)，完整运行记录在 `output/experiment_runs/`。
训练与依赖说明见[运行指南](../docs/guides/running_experiments.md)。
