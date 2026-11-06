# 第二轮租金预测探索

本轮筛选 **88个配置**：31组误差校正、14组TabICL、24组市场归一化、11组同期特征、8组精简/交互。使用较早期预测选择方案和融合权重。

**有用的新信息是同楼栋、同户型的历史同期租金。400–430目标尚未达到。**

| 验证窗口 | 行数 | 上轮融合 RMSE | 本轮主融合 RMSE | 本轮 MAE |
| --- | ---: | ---: | ---: | ---: |
| 2024-04 至 2025-03 | 35,158 | 463.7589 | **452.9386** | 328.4365 |
| 2024-06 至 2025-03 | 28,995 | 456.8220 | **443.8630** | 322.9187 |
| 2023-12 至 2025-03，16个月 | 47,171 | 487.5770（原XGBoost） | **479.6351** | 349.2609 |

此前TabPFN分支在相同六月起点/28,995行上的最低开发记录为455.0710。本轮443.8630低约 **11.2080 SGD**。TabPFN为此前分支保存的参考分数，本次未重跑它。

## 为什么这轮有效

旧历史特征把同一楼栋/户型的过去租约混在一起。本轮另查目标月份之前12/24/36个月、精确同月和前后一个月的同期记录，生成样本数、均价、相对当时同户型市场的租金比例、比例波动和低租金比例。
它可能保留了周期性重复租赁的水平，但没有租客或单元ID，不能称为已识别续租；也不能据此作因果解释。
验证期查询只能匹配拟合起点前已观测月份；长跨度下不存在的记录保持缺失/计数0，不回填未来租金。

| 对照 | 原47列 LightGBM | 加同期统计62列 LightGBM | 差值 |
| --- | ---: | ---: | ---: |
| outer | 466.6302 | 457.8450 | -8.7852 |
| june | 458.3339 | 447.1701 | -11.1639 |

两组使用相同训练/验证行、基础树设置和较早时间早停规则；轮数各自由早期窗口选择。差异是该特征组在此管线中的增量，不能解释为单列或真实续租的因果效应。

## 本轮主融合

| 成员 | 权重 |
| --- | ---: |
| normalized_market/compact_ratio15_market_last | 13.094% |
| anniversary/broad_lgb15 | 9.659% |
| anniversary/broad_lgb63 | 10.352% |
| anniversary/exact_lgb15 | 17.311% |
| anniversary/exact_lgb63 | 0.658% |
| anniversary/window_cat6 | 41.380% |
| anniversary/window_lgb63 | 7.546% |

权重非负、和为1、单成员最高0.6，仅以2023-10至2024-03的17,977条较早预测优化。年度/六月模型重新拟合至各自起点，保持较早选定的轮数与权重。

## 其他路线与负结果

- 第二层偏差校正：31组配置，年度最低463.6825；对六月窗口未显示改善。
- 市场水平与房屋相对价格分解：24组配置。单独并未胜过同期模型，但其中一个轻量模型在主融合中提供互补。
- TabICLv2：14种上下文/目标/特征配置；早期抽样为每月200行。最初完整确认的相对历史残差版本在年度为488.1767，与旧融合按早期选权融合约463.7127。它不是本轮赢家。
- 加同期特征后的TabICL在相同1,200条抽样记录上为534.9542，同期CatBoost为530.1784、主融合为526.8643。抽样分数不与完整窗口混排。
- 精简36月字段、地址×户型×月份类别与归一化目标再试8组。较早选择的精简比例模型进一步混入后，年度453.7023、六月444.7842，均不如主融合，故不替换主结果。

## 长跨度复核与边界

16个月复核另用2023-06至2023-11选轮数和权重，再从2023-12固定起点预测至2025-03；不沿用包含2024年标签的选轮数或权重。模型/特征结构仍来自当前开发。
超过12个月时，12月前的同期记录逐渐不再属于已观测历史，24/36月统计继续可用。因此长跨度表现必须单独检查。
该窗口与一年期/六月窗口重叠，不计算跨窗口混合排名。所有时期此前都被项目实验查看过，仍为开发证据，不是独立最终测试或Kaggle成绩。

## 验证与复现

已独立重算 106 份预测结果，检查源标签、月份、行ID、有限值和源文件哈希；主融合14个模型重载后预测一致。时间边界测试见 tests/test_second_wave.py。

按三个月连续区块重采样得到的差值区间只描述固定预测在当前月份上的敏感性，不覆盖调参、随机种子或未来市场变化。见 monthly_comparison_*.csv 和 summary.json。

```powershell
python -m unittest tests.test_second_wave -v
python -m analysis.experiments.second_wave_anniversary screen
python -m analysis.experiments.second_wave_normalized screen
python -m analysis.experiments.second_wave_ensemble select
python -m analysis.experiments.second_wave_ensemble outer
python -m analysis.experiments.second_wave_ensemble june
python -m analysis.experiments.second_wave_stress
python -m analysis.experiments.second_wave_report
```

模型/逐行预测：`output/models/round_02_anniversary_history/`。本轮主方法只需已有Conda建模环境；TabICL额外包安装在项目 `.cache/foundation-deps/`，权重在 `.cache/foundation-models/`。
TabICL来源：[作者实现](https://github.com/soda-inria/tabicl)、[作者权重](https://huggingface.co/jingang/TabICL)。所有模型推理在本地进行。
