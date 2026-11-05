# main 代码入口

| 任务 | 实现 | 命令 |
|---|---|---|
| main 要求卡审计与误差诊断 | `validation/` | `python -m analysis.main planned` |
| 文件完整性 | 根目录 `review.py` | `python review.py verify` |
| 全 main 静态审查 | `validation/repository_audit.py` | `python -m analysis.validation.repository_audit` |
| EDA | `eda/` | `python -m analysis.main eda --scope all` |
| C7 基线与传统模型 | `experiments/baselines/` | `python -m analysis.main baseline`、`compare` |
| 固定特征阶梯 | `experiments/reporting/report_selection_trials.py` | `python -m analysis.main ladder --stage outer` |
| 共同 27 列模型 | `experiments/model_comparison/` | `python -m analysis.main common-models` |
| Concat 模型重放 | `experiments/model_comparison/concat_transformer/` | `python -m analysis.main concat verify` |

训练命令的参数见 `python -m analysis.main --help` 和[运行指南](../docs/guides/running_experiments.md)。`planned` 只分析冻结预测，不训练；输出是开发期分析，F2 缺失项继续待补。

`pipeline/` 提供原始数据、特征和学习器的共享实现。`experiments/` 按用途与历史轮次保留主线依赖，`studies/` 保留测试与预处理需要的研究模块。旧模块别名供检查点重载使用，不把依赖模块视为新的 main 实验。

F1 固定配方独立放在 `reproduction/f1/runtime/`，避免当前代码修改影响旧提交复现。TabPFN 另存 `workflows/tabpfn/`，通过根目录 `review_tabpfn.py` 在 `output/` 中组装独立工作目录。

[实验结果与状态](../experiment/main/README.md) · [当前要求卡与规划](../docs/planning/main-appendix-experiment-plan.md)
