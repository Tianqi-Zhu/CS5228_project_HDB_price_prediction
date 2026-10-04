# Historical spatial benchmark

这是 `origin/ziyuan——spatialfeatures` 的精简结果，保留 LightGBM、CatBoost 和抽样 KNN
的指标，不复制模型二进制和逐行验证预测。

验证期为 2024-04 至 2025-03。LightGBM 加入 17 个楼栋／空间特征后，RMSE 从
484.49 降至 477.20；CatBoost 从 487.05 降至 480.03。KNN 只使用 12,000 条训练样本，
不能和全量树模型作严格排名。

它与 `../auxiliary_ablation/` 高度相似，但不完全重复：两者的基础时间编码、类别列和
空间特征集合不同，因此 477.20 与 476.97 的小差异不是结果冲突。
