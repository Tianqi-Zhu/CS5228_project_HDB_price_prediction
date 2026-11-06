# 空间 EDA：主线选取与来源核验

| 主线证据 | 内容 | 使用边界 |
|---|---|---|
| [原始与调整后关联](raw_adjusted_comparison.csv) | CBD、MRT、商场、小学距离及楼层、楼龄，6 项 | 同时展示 Pearson 与控制月份、户型、面积后的 Pearson；不能把相关当预测增益 |
| [设施距离分箱对照](facility_bin_controls.csv) | `joint_cbd` 三类设施、18 个分箱 | 149,841 条完整记录，排除 159 条；9,395 栋楼聚类，点态 95% 区间 |
| [来源与输入核验](../evidence-audit.json) | 原来源文件逐字节哈希、当前原始数据内容身份 | 此处复核来源和统计口径，没有独立重估原回归 |

来源：`ziyuan——spatialfeatures @7caf2cb77095f3b99cd242db0074189a355c0aae`。逐文件原路径和 Git blob 哈希见 [imported-evidence.json](../imported-evidence.json)。`source/` 为未改写的历史来源快照，原报告中的旧相对链接和 `OPEN_MRT` 命名保留；当前采用[所给设施快照场景](../../../../docs/methods/prediction-scenario.md)。原始 CSV 的 CRLF 与 Git 源 LF 只在内容身份核验时规范化，不修改训练输入字节。

两组调整口径分别记录：关联表的调整为月份、户型、面积；分箱模型另有月份×户型固定效应、town 固定效应、面积及楼龄二次项、楼层和 CBD 距离。分箱参照组为 0–0.3 km。不能将这两种调整合称同一模型，也不能据区间或关联推断设施的因果作用。

主线仅使用以上两张选表解释空间变量的定义与混杂；完整探索仍服务 appendix X01/X08。来源 26 列方案与主线 static19、common27、adjusted62 的差别见 [特征清单](../../M03/feature_schemas/README.md)。
