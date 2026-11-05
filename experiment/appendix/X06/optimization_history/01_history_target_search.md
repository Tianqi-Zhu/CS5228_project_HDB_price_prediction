# 新一轮租金预测探索

本轮筛选 **44 个新配置**，并按较早窗口的选择对 7 个配置做年度重训。
原本地融合 RMSE **465.6623**；本轮预先锁定融合 **463.7589**。
差值（新−旧）为 **-1.9034 SGD**。目标 400–430：**未达到**。

## 验证协议

- 筛选：2023-10 至 2024-03，17,977 行，信息截至 2023-09；用于参数、轮数、特征方案和融合权重选择。
- 年度验证：2024-04 至 2025-03，35,158 行，信息截至 2024-03；所有验证行均保留。
- 新模型外层训练使用内部选定的固定轮数，不在年度窗口早停；融合权重在本轮年度评分之前锁定。
- 所有历史标签只来自查询信息截止点之前；跨度模拟使用 0/5/11 个月延迟。
- 全分布模型有六个月历史预热；受控27列实验保留全部拟合行。因此跨两类的变化不能全归因于架构或单个特征。
- 原项目此前已反复查看年度验证期；本轮仍属于开发证据，不是新独立测试，也不是 Kaggle 分数。

## 探索的模型与特征

| 路线 | 内容 |
| --- | --- |
| 分布历史 | 楼栋/街道/规划区/Town/房型层次的均价、标准差、相对同月同户型均价、低/高租金比例、最近均价、6/12月统计 |
| 市场目标分解 | 直接租金、相对历史市场基准的残差、相对市场基准比例；比例模型按基准平方加权，保持SGD平方损失目标 |
| 跨度模拟 | 同一训练行生成0/5/11个月信息延迟，匹配冻结起点的多月预测场景 |
| 空间可比 | 同户型、24个月内的近邻，按位置/面积/租约年份与时间衰减得到16/48/128近邻统计 |
| 分层预测 | 先预测低/普通/高租金概率，再合并各层条件均值；查询租金不参与推理 |
| 稳定性 | 6/12/24月半衰期加权；仅相对价格特征；去掉高基数地址类别；15/31/63叶树 |
| 受控特征添加 | 固定原27列，只加楼栋分布、各层分布、相对历史或可比房源统计 |

## 年度验证结果

| 配置 | 特征数 | RMSE | MAE |
| --- | ---: | ---: | ---: |
| distribution_cat6 | 122 | 466.3714 | 341.2348 |
| compact_plus_relative_history_15 | 47 | 466.6302 | 341.6711 |
| compact_normal | 118 | 467.0800 | 346.0002 |
| compact_plus_relative_history_63 | 47 | 467.5661 | 342.5580 |
| distribution_residual | 122 | 468.8795 | 346.0106 |
| distribution_ratio | 122 | 469.8058 | 347.1550 |
| mixture_three | 122 | 472.1591 | 348.9446 |
| 本轮锁定融合 | 多模型 | 463.7589 | 338.5783 |
| 原本地融合 | 多模型 | 465.6623 | 336.9480 |

## 融合与结果解释

融合权重只使用较早的17,977行预测；非负、和为1、单成员最高0.6，不增加常数偏移。完整候选与权重在 `ensemble_selection.json`。

| 成员 | 权重 |
| --- | ---: |
| old_catboost | 36.1128% |
| old_ft_transformer_default | 5.0438% |
| old_hist_gradient_boosting | 4.4362% |
| old_tabnet | 5.3526% |
| old_xgboost | 6.4580% |
| compact_normal | 24.6252% |
| compact_plus_relative_history_15 | 4.7052% |
| compact_plus_relative_history_63 | 0.3056% |
| distribution_cat6 | 4.7445% |
| distribution_ratio | 3.4128% |
| distribution_residual | 0.1884% |
| mixture_three | 4.6149% |

对固定预测做三个月连续区块重采样5,000次，新−旧RMSE差值的描述性95%区间为 [-2.629, -1.600]。它仅反映这12个月的敏感性。

本轮没有发现支持400–430目标的结果。低租金分层、更多历史分布或跨度模拟本身都不保证改善；小幅开发分数差异不能保证未来收益。
原始街道大小写的训练期审计也未出现明显分层信号，未将格式当作新主线。

## 文件与复现

- `metrics.csv`：全部筛选与年度配置；`summary.json`：主要结论。
- `verification.json`：独立重算、标签/行序/源文件哈希、模型重载和查询标签变更检查。
- `.cache/round_01_history_target_search/`：可再生特征；`output/models/round_01_history_target_search/`：本地模型与逐行预测。
- 原始CSV保持不变，所有本轮新特征均未读取公开源的目标月份租金；本轮未生成或上传比赛提交。

```powershell
python -m unittest tests.test_frontier_search -v
python -m analysis.experiments.frontier_search screen
python -m analysis.experiments.frontier_refinement screen
python -m analysis.experiments.frontier_mixture screen
python -m analysis.experiments.frontier_compact screen
python -m analysis.experiments.frontier_ensemble select
python -m analysis.experiments.frontier_ensemble evaluate
python -m analysis.experiments.frontier_report
```

环境使用已有 cs5228 Conda 环境，不需要新增依赖。

## 仅旧模型的融合对照

按相同的较早窗口优化、相同非负/权重上限规则，仅使用旧模型得到年度RMSE **465.1374**。加入新模型的融合为 **463.7589**，进一步降低 **1.3785 SGD**。
这控制了将原来三个成员扩展为更多旧模型的影响，但不能将收益归因于某一个新特征。该补充对照在主年度结果之后添加，权重优化仍只读取较早窗口标签。

## 与此前TabPFN最低分的同起点对照

额外按2024-06起点重训全部12个成员，在相同28,995行上的融合RMSE为 **456.8220**；此前TabPFN最低记录为 **455.0710**。本轮没有刷新该记录。
两组只用六月以前标签重新选权的融合也未改善；完整结果、成员和复现命令见 [六月对照](../../output/experiment_runs/round_01_history_target_search/matched_june/README.md)。
本轮一年期的改善不能被写作整体最佳模型刷新；两个验证窗口必须分别报告。
