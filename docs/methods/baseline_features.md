# 推荐特征工程：19 个静态特征 + 8 个历史特征

每列构造原因与时间可用性已整理为 [27列字典](../../experiment/appendix/X01/feature_evidence/feature_dictionary.csv)。
逐列/成组删除后的实际RMSE增量见 [新消融证据](../../experiment/appendix/X01/feature_evidence/FEATURE_ABLATION.md)，
主结论和报告写作入口见 [当前项目报告](../../experiment/appendix/X01/feature_evidence/BASELINE_27_FEATURES.md)。删除影响是条件预测增量，不是因果贡献。

当前推荐使用 `analysis.pipeline.features.RentalFeatureEngineer()`，默认策略为
`compact_history_27`。确定性处理、类别词表和租金历史都由这个入口完成。首轮七组比较
选出31列，随后深入审计删除4个重复统计，四窗分数保持相同。完整依据见
[深入特征实验](history_feature_optimization.md)，不代表全局最优。

## 为什么需要收敛

原项目的特征列表分散在基础模型、辅助消融、深度学习和时空实验中。相同距离存在多个
别名，多个年份和楼龄字段高度冗余；不同实验又混用了三个月、十二个月、多折验证，
以及每月更新和一次预测未来多月的场景。新的建模工作应先使用一个明确的默认策略。

训练数据为 2021-01 至 2025-03，测试数据为 2025-03 至 2026-07。推荐策略采用
固定起点预测：历史信息在拟合截止时冻结，不依赖未来月份的经济观测或真实租金。

## 默认清单

| 分组 | 字段 | 数量 |
| --- | --- | ---: |
| 房屋属性 | `FLAT_TYPE_CLEAN`, `FLAT_MODEL_CLEAN`, `FLOOR_AREA_SQM`, `FLAT_AGE_YEARS`, `MAX_FLOOR` | 5 |
| 时间 | `APPROVAL_MONTH_INDEX`, `APPROVAL_MONTH_SIN`, `APPROVAL_MONTH_COS` | 3 |
| 区位 | `TOWN_CLEAN`, `PLANNING_AREA_CLEAN`, `LATITUDE`, `LONGITUDE`, `CBD_DIST_KM` | 5 |
| 设施 | MRT、商场、小学各自的最近距离和 1km 内数量 | 6 |
| 过去租金 | 两种分组各四个统计量，详见下文 | 8 |
| 合计 | 4 个类别特征、23 个数值特征 | 27 |

`MAX_FLOOR` 是楼栋最高楼层。`FLAT_AGE_YEARS` 是审批年份减租约开始年份。
设施距离是 Haversine 球面距离，设施数量按去重后的坐标计算。

四个类别字段为 `FLAT_TYPE_CLEAN`、`FLAT_MODEL_CLEAN`、`TOWN_CLEAN`、
`PLANNING_AREA_CLEAN`。词表只从拟合集学习，预测中的新类别置为缺失。
LightGBM 直接使用 pandas 类别类型及数值缺失值，这套默认方案不需要 One-hot 或
StandardScaler，也不从全量数据学习填补值。

保留原始行数、顺序、索引与重复记录，不对租金做 IQR 删行。`FURNISHED`、`FEE` 为
常量，不作为输入；每个物理距离只保留一个字段名。省去原始地址、邮编等高基数类别，
地址仅作为历史分组键。省去竣工年份和另一套楼龄、多个设施半径、手工交互与经济信号。
这些取舍由整组候选比较支持，不应解读成逐一证明了每个被省略字段没有价值。

## 历史租金的计算规则

两种分组分别是：

1. 同楼栋、同户型。
2. 同规划区、同户型。

原方案的“同一房屋属性组合”历史在当前数据上与第一组完全重复，已删除。

每组输出 `MEAN`、`COUNT`、`LAST_MEAN`、`RECENCY_MONTHS`：过去所有记录的均价、
记录数、最近有记录月份的均价、距该月份的月数。

每条记录只查询严格早于其审批月份的租金；同月的所有记录使用同一份过去历史。
验证和测试的统计只来自拟合集。未见房屋的细粒度均价为缺失、计数为零，模型还可以
使用规划区历史和静态特征。计数让模型识别统计的支持程度。

测试集有 380 条记录位于训练集也出现的 2025-03。预测这些记录时，历史统计只使用
2025-02 及以前的标签。预测 2025-04 及以后时，历史均价固定为截至 2025-03 的
拟合统计，距最近记录的月数随预测月份增加。这比一般的留一目标编码有更严格的时间
约束；留一编码仍可能使用该行之后月份的标签。

## 比较方法与结果

固定使用 LightGBM：400 棵树、学习率 0.05、31 个叶节点、最小叶样本数 30、L2 为 3、
随机种子 42。每组使用相同的拟合行与验证行，不做早停或调参。

三个排名窗口均为十二个月，互不重叠：2022-04 至 2023-03、2023-04 至 2024-03、
2024-04 至 2025-03。各折只用起点之前的数据拟合，验证标签不反馈给历史。
另单独报告 2023-12 至 2025-03 的十六个月压力测试，其月份与排名窗口重叠，不计入汇总。

| 特征方案 | 特征数 | 2022-04 起 RMSE | 2023-04 起 RMSE | 2024-04 起 RMSE | 三窗汇总 RMSE | 16个月 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原基础 C | 7 | 722.46 | 595.68 | 485.09 | 605.20 | 502.21 |
| 原空间方案的规范字段版本 | 29 | 719.97 | 591.90 | 476.97 | 600.77 | 495.21 |
| 精简静态 | 19 | 719.59 | 589.76 | 477.02 | 599.90 | 493.89 |
| 丰富静态 | 40 | 723.46 | 606.72 | 479.77 | 607.93 | 503.04 |
| 精简静态 + 历史（首轮方案） | 31 | 711.08 | 582.59 | 468.49 | 591.96 | 491.22 |
| 深入审计后的当前默认 | 27 | 711.08 | 582.59 | 468.49 | 591.96 | 491.22 |
| 原空间 + 历史 | 41 | 712.74 | 587.68 | 470.03 | 594.75 | 491.13 |
| 丰富静态 + 历史 | 52 | 721.03 | 610.34 | 485.28 | 609.77 | 512.10 |

31 个特征在三段年度验证中均优于 29 特征方案，三窗汇总 RMSE 降低约 1.47%。
十六个月窗口中，41 特征方案比它低约 0.09 SGD；结合年度稳定性与特征数，仍选择 31 个
特征作为首轮方案。深入审计后进一步简化为分数相同的27列。没有证据将这0.09 SGD
的差异解释为可靠的长期预测优势。

完整七组、多折及十六个月结果在
`output/experiment_runs/features_selection`：
`fold_metrics.csv` 为分折分数，`pooled_metrics.csv` 为三段窗口加权汇总，
`monthly_errors.csv` 为月度分数，`results.json` 记录字段、模型参数、版本和源文件哈希。

作为不纳入经济信号的已有依据，原经济消融的 COE+股票组合在四个前向窗口均差于
空间基线，且逐月更新假设不适合本次固定起点预测。见
`output/experiment_runs/features_economic_ablation/forward_scores.csv`。

## 复用与复跑

复跑基础和当前注册特征策略比较：

```powershell
python -m analysis.main feature-select
```

推荐方案可放入 sklearn 管线。先用时间划分后的拟合集 `fit`，再对验证集 `predict`；
最终配置确定后再对全部训练集拟合。下面演示最终拟合，不写提交文件：

```python
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from analysis.pipeline.config import FAMILY_SETTINGS
from analysis.pipeline.features import RentalFeatureEngineer

train = pd.read_csv("data/train.csv", dtype={"BLOCK": "string"})
test = pd.read_csv("data/test.csv", dtype={"BLOCK": "string"})
model = Pipeline([
    ("features", RentalFeatureEngineer()),
    ("model", LGBMRegressor(**FAMILY_SETTINGS["lightgbm"])),
])
model.fit(train.drop(columns="MONTHLY_RENT"), train["MONTHLY_RENT"])
prediction = model.predict(test)
```

整条管线可以通过 joblib 保存，确保词表和租金历史与模型一起复用。
基础 `validate --features A/B/C` 仍是历史基线入口，默认新方案由上述类提供。

运行检查见 `output/experiment_runs/features_history_optimization/runtime_validation.json`：当前27列的
sklearn管线复现最新验证 RMSE 468.486718，并完成150,000条全量拟合及50,000条测试记录
的内存预测；测试顺序保留、预测全部为有限值。源 CSV 的哈希校验通过。

## 结论边界

所有带标签月份在以往实验中已被查看过，本轮是方案选择评估，没有新的独立测试标签。
三个年度窗口改善及重复字段审计支持将 27 个特征作为默认起点；它不能证明未来 Kaggle 得分或任何模型
下的最优性。早期窗口仍明显低估，市场趋势变化没有仅靠这些特征解决。
设施表为静态快照，缺少历史开通日期；当前实验检验的是所给比赛数据条件下的空间代理。
