# 比赛数据说明

此目录保留比赛提供的原始CSV。以下行数为当前本地版本的数据行数，不含表头。
项目预测目标为每月租金 `MONTHLY_RENT`，单位为新加坡元（SGD）。

| 文件 | 行数 | 用途 |
| --- | ---: | --- |
| [train.csv](train.csv) | 150,000 | 带租金标签的样本；按时间划分拟合、选择与本地验证窗口 |
| [test.csv](test.csv) | 50,000 | Kaggle查询输入；不包含 `MONTHLY_RENT` |
| [example-submission.csv](example-submission.csv) | 50,000 | 输出模板，列为 `Id,Predicted`；ID及行顺序按模板保留 |
| [auxiliary/sg-hdb-block.csv](auxiliary/sg-hdb-block.csv) | 9,464 | 楼栋坐标、建成年份、楼层及区域信息 |
| [auxiliary/sg-mrt-stations.csv](auxiliary/sg-mrt-stations.csv) | 243 | 地铁站位置与状态，用于交通可达性特征 |
| [auxiliary/sg-schools.csv](auxiliary/sg-schools.csv) | 337 | 学校类别和位置，用于周边设施特征 |
| [auxiliary/sg-shopping-malls.csv](auxiliary/sg-shopping-malls.csv) | 113 | 商场位置，用于距离等空间特征 |
| [auxiliary/sg-coe-prices.csv](auxiliary/sg-coe-prices.csv) | 539 | 月份、车辆类别及COE价格等所给宏观信息 |
| [auxiliary/sg-stock-prices.csv](auxiliary/sg-stock-prices.csv) | 91,642 | 日期、证券标识及价格等所给宏观信息 |

`train.csv` 比 `test.csv` 多一列 `MONTHLY_RENT`；其余原始列为日期、Town、楼栋、街道、户型、
房型、面积、家具情况、租契起始年份及费用。实际输入是否使用某列，由各轮配置决定。
字段清洗和基础特征定义见[特征说明](../docs/methods/baseline_features.md)，
最终方法见[最终方案](../experiment/01_final/README.md)。

## 使用边界

- 本地验证从 `train.csv` 的历史时间段划分，历史租金统计按各实验的信息截止日构造。
- `test.csv` 用作最终查询和提交预测，不能提供训练或选择所需的真实租金标签。
- 纯历史方案与使用比赛所给查询月份宏观信息的方案分别报告；给定未来月份宏观表不等于当时可实时观测的预测条件。
- 最终三份提交不依赖 `.cache/public_rents/` 中独立来源审计的数据，也不使用公开租金匹配结果。

清洗、派生特征、缓存和预测写到对应产物目录，不覆盖此处原始CSV。
已有实验的 `source_sha256` 和提交组件清单记录输入版本；更换CSV后须重新核验，不能直接沿用旧指标。
环境与运行命令见[运行指南](../docs/guides/running_experiments.md)。
