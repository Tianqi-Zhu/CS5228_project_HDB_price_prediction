# Prediction experiments

| 目录 | 主要比较 | 验证口径 |
| --- | --- | --- |
| `model_baseline/` | 均值与 CatBoost A/B/C | 2024-04 至 2025-03 |
| `model_families/` | Ridge、RF、CatBoost、LightGBM | 同上 |
| `auxiliary_ablation/` | 主表、楼栋、设施、宏观逐组消融 | 同上 |
| `economic_feature_ablation/` | 只使用先前月份的经济特征 | 多个前向折 |
| `deep_learning/` | 深度模型、辅助树、历史先验和融合 | 2024-04 至 2025-03 |
| `spatiotemporal_hybrid/` | 固定预测原点的历史空间模型 | 三个扩展窗口折 |
| `neural_tabm/` | TabM 风格网络及历史树融合 | 同上 |
| `relu_mlp/` | ReLU MLP 嵌套 epoch 选择 | 2025-01 至 2025-03 |
| `spatial_benchmark/` | 旧空间分支的树模型与 KNN 对照 | 2024-04 至 2025-03 |

同一目录内的模型通常可以直接比较；跨目录比较前必须同时核对验证月份、特征、模型、
异常值处理和预测原点。
