# CS5228：HDB 月租预测

使用比赛提供的训练集和辅助表预测月租金（SGD）。当前完整项目分支为 `zch-complete-proj`，由原 `feature/midterm-eda-models` 延续提交历史。

## 先看什么

| 你的目的 | 从这里进入 |
| --- | --- |
| 了解最终方法与成绩 | [最终结论](experiment/01_final/README.md) |
| 从基础到final阅读完整研究过程 | [完整实验报告](experiment/01_final/EXPERIMENT_REPORT.md) |
| 理解特征工程为什么有效 | [特征工程](experiment/02_feature_engineering/README.md) |
| 比较不同模型 | [模型对比](experiment/03_model_comparison/README.md) |
| 查看每轮探索及负结果 | [优化历程](experiment/04_optimization_history/README.md) |
| 运行代码 | [运行指南](docs/guides/running_experiments.md) |
| 了解原始数据及使用边界 | [数据说明](data/README.md) |
| 检查实现与既有结果 | [测试说明](tests/README.md) |
| 查看已保存模型、参数及指标来源 | [产物清单用法](docs/guides/running_experiments.md#基础特征与模型对照) |

## 最终三份提交

| 方法 | 本地年度RMSE | 本地近期RMSE | Kaggle显示分数 |
| --- | ---: | ---: | ---: |
| 历史特征主融合 | 451.1810 | 442.9025 | 415.86912 |
| 所给宏观增强 | 451.2081 | 442.8892 | 415.78232 |
| **分组融合＋PLR-FT（final）** | **451.0295** | **441.8345** | **415.37529** |

本地年度为2024-04至2025-03，近期为2024-06至2025-03；窗口重叠且已用于开发。
Kaggle数值来自用户截图，未标注公榜／私榜。final是三份提交中的选择，第九轮的年度分数仍更低。
最终方法和信息边界见[最终报告](experiment/01_final/README.md)。

## 目录怎么读

```text
experiment/       给人读：最终结论、特征、模型、优化历程、数据分析
analysis/      看实现：共享管线与按用途分组的实验代码
docs/          查用法：运行、目录、依赖说明与历史维护
data/          原始比赛数据，保持不变
output/        运行产物：试验明细、模型、逐行预测、提交文件
requirements/  分层依赖
tests/         自动检查
```

`experiment/`只保留汇总与阅读材料。完整试验配置、逐次指标与选择记录在
`output/experiment_runs/`，模型在 `output/models/`，最终提交在
`output/submissions/final/submission.csv`，候选在 `output/submissions/candidates/`。
这些运行产物不随Git分发。详细目录见[结构说明](docs/guides/project_structure.md)。

## 环境与常用命令

在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda activate cs5228
python -m analysis.main --help

# 只查看入口，不训练
python -m analysis.main experiment-list --group wave10

# 只检查环境与资源，不安装、不训练
python -m analysis.main preflight --profile final

# 从已有结果更新阅读报告，不训练
python -m analysis.main experiment-refresh
```

仅查看索引可用标准库入口 `python -m analysis.experiment_catalog --group final --json`。
GPU及额外作者模型依赖见[requirements说明](requirements/README.md)和[运行指南](docs/guides/running_experiments.md)。

最终三份方案使用比赛训练标签与所给辅助数据；测试表作为查询输入，不使用其真实租金。
其他探索中的公开预训练权重在相应轮次注明。原始数据和预测数值未因目录整理而改变。
