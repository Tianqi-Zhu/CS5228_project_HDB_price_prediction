# 测试与运行核查

在仓库根目录、激活 `cs5228` 环境后执行：

```powershell
conda activate cs5228
python -m unittest discover -s tests -v
python -m analysis.main verify
```

第一条发现全部 `test_*.py`，检查数据处理、时间边界、模型适配和输出行为。
第二条核对当前环境、既有报告指标、数据来源哈希和最终融合组件。
两者用途不同；都通过也不代表已经从零重训十轮实验，或验证了新机器的GPU兼容性。

## 按问题找测试

| 问题 | 测试文件 |
| --- | --- |
| 共享管线、模型保存、输出位置 | `test_pipeline.py`、`test_project_paths.py`、`test_output_defaults.py` |
| 空间辅助表、经济信号与特征选择 | `test_auxiliary_features.py`、`test_economic_feature_ablation.py`、`test_economic_signals.py`、`test_feature_selection.py`、`test_feature_evidence.py` |
| 模型搜索、TabM与时空混合 | `test_model_search.py`、`test_neural_tabm.py`、`test_spatiotemporal_hybrid.py` |
| 十轮探索中的时间截止、特征与融合 | `test_frontier_search.py`、`test_second_wave.py` 至 `test_tenth_wave.py` |
| 面板、空间与相关研究 | `test_group_temporal_panel.py`、`test_spatial_controls.py`、`test_temporal_correlations.py` |
| 来源审计的键匹配 | `test_source_overlap.py`，属于独立研究，不为最终方法提供测试租金标签 |
| 实验索引、阅读报告与组件完整性 | `test_experiment_catalog.py`、`test_report_bundle.py`、`test_artifact_verification.py` |
| 资源缺失、哈希变动、路径越界与无建模依赖的预检 | `test_resource_preflight.py` |
| 当前产物覆盖、作废状态、指标上下文与最终组件追溯 | `test_artifact_catalog.py` |

不移动已有测试模块，便于沿用历史命令。只检查一类时可以使用发现模式：

```powershell
python -m unittest discover -s tests -p test_project_paths.py -v
python -m unittest discover -s tests -p 'test_*wave.py' -v
python -m unittest discover -s tests -p test_artifact_verification.py -v
```

## 依赖与判读

- 完整集合需要比赛数据和核心环境。既有结果一致性是一项本地集成检查：新克隆没有 `output/experiment_runs/` 时明确跳过；目录存在却不完整或指标不一致时仍报错。完整结果核查须先按[运行指南](../docs/guides/running_experiments.md)准备产物。
- 可选TabM、RealMLP、ModernNCA与TabDPT适配测试在对应包／作者源码未准备时明确跳过，并给出准备入口；文件存在却导入或行为异常时仍报错。核心环境中的RTDL等依赖缺失不会被当作可选测试跳过。
- 一部分模型适配测试使用小型临时样本或模拟训练器；它们检查实现行为，不衡量模型的实际RMSE。
- 文件重载和哈希核验检查产物是否完整，不能证明某个特征的泛化收益，也不能替代时间验证或Kaggle成绩。
- 测试只使用受控的小样本和已保存结果，不会上传Kaggle，也不会启动十轮完整训练。

环境入口见[依赖说明](../requirements/README.md)，数据角色见[数据说明](../data/README.md)。
