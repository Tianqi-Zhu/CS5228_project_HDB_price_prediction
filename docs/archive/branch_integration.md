# 五个工作分支整合说明

本文件记录 `feature/midterm-eda-models` 中的整合范围。整合只修改该分支；其他远程分支保留原样，作为历史来源。

## 来源与落点

| 来源分支 | 保留的独有内容 | 统一后的落点 |
| --- | --- | --- |
| `feature/midterm-eda-models` | 主数据 EDA、A/B/C 特征、四模型比较、时间验证、提交管线 | `analysis/pipeline/`、`analysis/eda/`、基础实验 |
| `feature/temporal-auxiliary-analysis` | 辅助特征消融、滞后经济信号、分组时间面板、固定预测原点的历史空间模型 | `analysis/experiments/baselines/auxiliary_ablation.py`、`analysis/studies/`、`spatiotemporal_hybrid.py` |
| `feature/neural-tabm-experiment` | TabM 风格网络以及与历史树模型的融合 | `analysis/experiments/baselines/neural_tabm.py` |
| `relu-rental-prediction` | 嵌套时间验证、仅拟合集 IQR 清理的 ReLU MLP | `analysis/experiments/baselines/relu_mlp.py` |
| `ziyuan——spatialfeatures` | 空间特征字典、描述分析、固定效应、设施联合控制、重叠分层匹配 | `analysis/studies/spatial/` 与统一入口 `spatial-controls` |

`main` 没有相对上述祖先提交的独有项目内容，因此不复制空壳结构。

## 去重原则

- 所有命令统一由 `python -m analysis.main` 进入。
- 原始数据只从 `data/*.csv` 和 `data/auxiliary/*.csv` 读取；不提交第二份 processed CSV。
- 空间分析共用 `analysis/pipeline/auxiliary.py`，由 `spatial-controls` 在内存中构建一次规范数据定义。
- 未复制重复的包装脚本、重复进度报告、验证/测试逐行预测、模型二进制及大体积中间表。
- 小型指标、图和汇总表保留，便于审阅；可再生的大型产物由 `.gitignore` 排除。

## 指标可比性

| 路线 | RMSE | 验证口径 | 解释 |
| --- | ---: | --- | --- |
| 主表 LightGBM | 485.09 | 2024-04 至 2025-03 | 统一 12 个月基准 |
| 空间／设施消融 E | 476.97 | 同上 | 静态楼栋与设施快照带来主要增益 |
| 深度＋辅助树＋历史先验固定融合 | **466.47** | 同上 | 当前同口径最佳已保存结果；MAE 341.14 |
| 历史空间树 | 469.61 | 固定原点首折 | 三折 pooled RMSE 543.10，远期外推明显变难 |
| TabM＋历史树 | 468.83 | 固定原点首折 | 仅 6% 神经权重；三折 pooled RMSE 542.78 |
| ReLU MLP | 463.97 | 2025-01 至 2025-03 | 只有 3 个月且采用嵌套选 epoch，不能与 12 个月数字直接排名 |

空间固定效应和匹配分析是解释性研究，不产生可与预测 RMSE 排名的分数。原空间分支的模型二进制和逐行预测未复制，但 LightGBM、CatBoost 与 KNN 的精简指标已保存在 `output/experiment_runs/baseline_spatial_models/`。ReLU 原分支的精简历史指标保存在 `output/experiment_runs/baseline_relu_mlp/historical_results.json`。

## 推荐复现顺序

```powershell
conda activate cs5228
python -m unittest discover -s tests -v
python -m analysis.main aux-ablation
python -m analysis.main spatial-controls
python -m analysis.main deep-summary
```

`tabm`、`spatiotemporal` 和 `relu` 是进一步模型实验，验证窗口不同；比较时应读取各自结果文件中的切分日期，不应只比较单个 RMSE 数字。
