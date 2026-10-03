# 租金预测模型基准（时间留出验证）

用 `MONTHLY_RENT` 预测月租（新元），训练至 2024-03，验证期为 2024-04 至 2025-03。训练 114,842 行，验证 35,158 行。RMSE 越低越好。

| 模型 | 验证 RMSE（SGD） | MAE（SGD） | 训练行数 |
|---|---:|---:|---:|
| lightgbm_spatial | 477.20 | 355.32 | 114,842 |
| catboost_spatial | 480.03 | 358.68 | 114,842 |
| lightgbm_base | 484.49 | 361.01 | 114,842 |
| catboost_base | 487.05 | 363.76 | 114,842 |
| knn_base_sample | 539.58 | 405.22 | 12,000 |
| train_mean | 846.75 | 690.60 | 114,842 |

LightGBM 加入空间及建筑特征后，RMSE 从 484.49 降至 477.20，下降 7.29 SGD。CatBoost 的同项下降为 7.03 SGD。这个对比固定了数据划分和各模型超参数；新增特征同时含位置、周边设施和建筑属性，不能单独归因于设施。

KNN 仅使用固定随机种子抽出的 12,000 条训练样本，作为轻量参考，不用于和全量训练的树模型做严格公平排名。所有模型在完整验证集评估。空间设施快照的历史可得性尚未逐期核实；这是单次探索性时间留出验证，且项目此前 EDA 已查看所有标签，不能当作完全未触碰的最终测试分数。

运行：`DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib .venv-model-benchmark/bin/python scripts/benchmark_rent_models.py`（本机 LightGBM 需 libomp；其他环境可直接用项目建模依赖运行）。模型文件与 `spatial_preprocessing.json` 位于本目录，另有 `metrics.csv`、`validation_predictions.csv`、`manifest.json`。
