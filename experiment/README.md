# 报告阅读导航

这里放适合阅读和引用的汇总材料。单次训练配置、分窗口明细、完整分支快照不在这里堆放，
需要追溯时进入 `output/experiment_runs/`。

| 阅读顺序 | 文件夹 | 内容 |
| --- | --- | --- |
| 1 | [01_final/](01_final/README.md) | 最终方法、三份提交、完整实验报告 |
| 2 | [02_feature_engineering/](02_feature_engineering/README.md) | 特征思路、字段字典、消融与受控阶梯 |
| 3 | [03_model_comparison/](03_model_comparison/README.md) | 同输入模型、推荐模型及完整模型目录 |
| 4 | [04_optimization_history/](04_optimization_history/README.md) | 十轮探索，各轮一份阅读报告 |
| 补充 | [eda/](eda/README.md) | 主表与辅助数据质量 |
| 补充 | [studies/](studies/README.md) | 经济、空间与面板关联研究 |

阅读报告由 `python -m analysis.main experiment-refresh` 从已有结果更新，不重新训练。
来源清单在[维护记录](../output/checks/records/reading_bundle_sources.json)；
本地运行明细和模型不随Git分发，但本目录的选定报告与表格可独立阅读。

同输入比较、完整管线比较和Kaggle分数分别解释，不能混成一个架构排名。
本地时间窗经过反复开发，年度与近期窗口重叠；最终Kaggle截图没有注明榜单类型。
