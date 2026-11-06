# 实验代码：按用途查找

| 文件夹 | 内容 |
| --- | --- |
| `baselines/` | 基础树模型、早期深度网络、MLP及空间基线 |
| `feature_engineering/` | 特征选择、历史表示与删除重训 |
| `model_comparison/` | 共同输入模型、搜索和相关准备工具 |
| `optimization/` | 第一至十轮探索，每轮一个有含义的文件夹 |
| `reporting/` | 模型筛选、报告汇总和历史分支盘点 |
| `submissions/` | 基础参照、历史／宏观与最终选择导出 |

`optimization/`使用 `round_01_history_target_search` 到
`round_10_plr_transformer_final`，模型和运行明细沿用相同轮次名称。

## 查命令和真实文件位置

```powershell
python -m analysis.main experiment-list --group wave10 --json
python -m analysis.experiment_catalog --group final --json
```

JSON中的 `source` 给出新的真实文件位置，`command`保留原有模块命令。
例如第十轮仍可运行 `python -m analysis.experiments.tenth_wave_structures screen`。
这样已有脚本调用和模型检查点的模块引用仍可使用。

索引只解析源码，不会训练。显示的是命令前缀，所需参数与前置结果见[运行指南](../../docs/guides/running_experiments.md)。
报告阅读入口在[experiment/](../../experiment/README.md)，无需逐个运行历史实验。
