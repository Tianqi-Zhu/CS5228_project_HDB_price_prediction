# MLP 集成预测与结果说明

本分支交付 MLP、TabM、融合预测、训练数据清洗对照及预测区间。原始 CSV 保持不变；新增代码、精简实验记录和可直接查看的预测文件一起保存。

**同一套独立区间校准协议下，未清洗融合模型 RMSE 为 467.03，清洗后的纯 MLP 为 471.26。** 历史最低开发验证 RMSE 464.21 来自另一套训练截至 2024-03 的神经网络融合，不能把它与本次截至 2023-12 的清洗对照直接归因为清洗收益。所有分数均为逐条租金、SGD 单位的本地验证结果，不是 Kaggle 隐藏测试成绩。

## 输入与网络

主要输入包括户型、房屋型号、Town、街道和楼栋 ID，面积、时间、房龄、位置、楼栋属性、设施距离/数量，以及严格来自过去的租金统计。街道和楼栋使用可学习的类别嵌入，少于四条的类别归入未知/稀有桶；数值缺失填补与分位数变换只在当前拟合部分学习。

当前独立 MLP 将类别嵌入和数值特征拼接，经过宽度 128 的输入投影及两个 128 单元全连接块，使用 SiLU、BatchNorm/LayerNorm 与 Dropout。它预测市场调整后的历史基准与真实租金之间的残差，再加回基准得到 SGD 租金；训练时另有价格档辅助分类头，预测时不读取价格档真实标签。AdamW 学习率 0.0005，batch 1,024。

TabM 使用官方 `tabm==0.0.3`，配合类别嵌入和分段线性数值嵌入，宽度 128、两个块、每个网络 16 个共享参数成员，AdamW 学习率 0.001、batch 512。它预测标准化租金，再还原为 SGD。

Deep Ensemble 分别用 seed 42、43、44 训练三个独立 MLP；TabM 也进行三个独立随机种子重训。先对每个 TabM 的内部成员求均值，再平均三个随机种子。融合为 MLP 集成与 TabM 集成各 50%。共享参数成员不视为 48 个完全独立网络。

此处的主要集成没有加入 RNN/LSTM latent vector。早期加深和 latent 实验作为历史对照保留，不能把所有尝试都解释为收益。

## 时间划分

| 阶段 | 数据 | 用途 |
| --- | --- | --- |
| 选择轮数 | 2021-01～2023-09 拟合，2023-10～2023-12 验证 | 选定 MLP 3 个 epoch、TabM 4 个 epoch |
| 重训独立网络 | 2021-01～2023-12，共 105,837 条 | 三个随机种子，所有训练期预处理重新拟合 |
| 区间校准 | 2024-01～2024-03，共 9,005 条 | 只校准区间，不再选择网络或融合权重 |
| 完整验证 | 2024-04～2025-03，共 35,158 条 | 比较 RMSE、MAE、偏差与覆盖率 |
| 测试预测 | `test.csv` 全部 50,000 条 | 保持样本提交的 ID 顺序，没有真实标签可评分 |

为保持校准流程一致，交付 CSV 使用截至 2023-12 的冻结模型，没有在区间校准后用全部 150,000 条标签重训。验证跨度为拟合后的第 4～15 个月，测试跨度为第 15～31 个月；更远月份的预测与区间属于超出已验证跨度的外推。

## 预测结果

| 方法 | RMSE | MAE | 平均偏差，预测减真实 |
| --- | ---: | ---: | ---: |
| 未清洗 MLP 集成 | 471.80 | 351.01 | -34.52 |
| 未清洗 TabM 集成 | 469.74 | 342.92 | +1.99 |
| 未清洗融合 | 467.03 | 343.66 | -16.26 |
| 融合加历史风险修正候选 | 465.34 | 344.72 | -31.25 |
| 明显低价清洗后的 MLP | 471.26 | 345.46 | -7.61 |
| 明显低价清洗后的融合 | 468.79 | 337.00 | +40.44 |
| 严格高价清洗后的融合 | 476.25 | 339.09 | +78.28 |

历史风险修正用更早的预测误差学习小幅调整。它降低了 RMSE，但 MAE 上升，未按预先设定的“双指标”条件替换对照。历史低价均值回缩等方案也已保存失败结果。

清洗对照的完整九行模型结果见[清洗指标](../reports/experiments/mlp_summary/cleaning/comparison_metrics.csv)。清洗提高预测价格，改善高租金组，却加剧低租金组误差：[分组结果](../reports/experiments/mlp_summary/cleaning/rent_band_metrics.csv)。

历史搜索中最好的点预测融合为 **RMSE 464.21、MAE 336.94**，使用 114,842 条 2021-01～2024-03 拟合记录，外层验证仍为上述 35,158 条。它同时改变了训练量、特征和模型组合，且外层月份已重复查看；没有证据将全部提升单独归因于加深网络。本次打包提供其[逐条验证预测](../reports/experiments/mlp_summary/validation/best_point_blend_outer.csv)，未提供与其对应的测试区间文件。

普通独热编码 MLP 的 6/8 层无 latent 结果为 488.35/484.79，加入 RNN/LSTM 并未稳定改善。可学习嵌入、历史租金、残差目标、数值表示与模型融合构成后续实验方向。完整正负结果见[历史实验目录](../reports/experiments/mlp_summary/historical_catalog.csv)，其中早期数值嵌入失效的实验已单独标记，不能作为正确分段嵌入的比较结果。

## Confidence 如何解释

本分支输出个人交易租金的预测区间，不给出“这条预测正确的概率”。独立网络的预测标准差只描述模型分歧，不等于全部预测误差，也没有除以随机种子数量的平方根。

校准分数为 `abs(真实租金 - 预测租金) / max(模型分歧标准差, 100 SGD)`。在独立校准集上，以有限样本修正后的秩分位数计算 90% 和 95% 区间。另保留固定宽度绝对误差校准作为对照，未按外层表现挑选区间方法。方法背景见 [Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) 与 [Conformal Prediction](https://arxiv.org/abs/2107.07511)。

| 未清洗模型 | 90%区间实际覆盖率 | 95%区间实际覆盖率 | 90%区间平均宽度 |
| --- | ---: | ---: | ---: |
| MLP | 92.56% | 96.28% | S$1,655 |
| TabM | 92.73% | 96.21% | S$1,666 |
| 融合 | 92.79% | 96.31% | S$1,670 |

这些是开发验证期的经验覆盖率。租金存在时间变化、同楼栋交易相关性及样本构成变化，不满足简单独立同分布假设，不能保证未来月份或每个房型都达到名义覆盖率。低租金组的融合 90%覆盖率只有 29.4%，即使整体覆盖率超过90%，特殊交易仍可能落在区间外。

## 剩余误差集中在哪里

最新风险修正候选中，真实租金≤1,900 的记录只占 3.7%，却贡献25.7%的平方误差，平均高估约 S$1,112。租金>3,800 的记录平均低估约 S$427。最差5%记录贡献40.9%的平方误差；Executive、Maisonette、大面积房屋和 Jurong West 更容易进入大误差组。这些分组重叠，是排查方向，不是因果解释。

大误差并不主要来自完全没有历史记录的房屋：最差5%的历史样本量中位数为9，全体为10。当前模型分歧也未可靠识别困难样本：1,422条绝对误差超过1,000的记录中，1,182条模型分歧标准差仍小于100。

相同输入对应多个标签会形成局部限制。例如验证期同一组输入的真实租金为4,000/5,000/6,000，统一预测5,000时组内RMSE最低，仍为816.50；当前未清洗融合预测5,028.49，组内RMSE816.99，已经接近这个局部最优。这只证明该组三条的限制，不能证明整体RMSE无法降至450。[案例与计算](../reports/experiments/mlp_summary/uncertainty_diagnostics/indistinguishable_inputs_proof.json)

## 可使用的预测文件

下列每份均为50,000条、按样本提交ID排序。清洗版本和风险修正版本保留作候选对照，未覆盖原预测。

| 文件 | 对应模型与用途 |
| --- | --- |
| [未清洗 MLP](../reports/experiments/mlp_summary/predictions/uncleaned_mlp_test_intervals.csv) | 独立 MLP 集成对照，验证 RMSE 471.80 |
| [未清洗融合](../reports/experiments/mlp_summary/predictions/uncleaned_hybrid_test_intervals.csv) | 主要融合对照，验证 RMSE 467.03 |
| [明显低价清洗 MLP](../reports/experiments/mlp_summary/predictions/obvious_low_mlp_test_intervals.csv) | 本轮清洗中纯 MLP 数值最好，验证 RMSE 471.26 |
| [明显低价清洗融合](../reports/experiments/mlp_summary/predictions/obvious_low_hybrid_test_intervals.csv) | MAE 337.00，验证 RMSE 468.79 |
| [严格高价清洗融合](../reports/experiments/mlp_summary/predictions/highest_only_hybrid_test_intervals.csv) | 失败对照，验证 RMSE 476.25 |
| [历史风险修正融合](../reports/experiments/mlp_summary/predictions/risk_corrected_hybrid_test_intervals.csv) | RMSE 465.34，但 MAE 略高的候选 |

`Id` 对应比赛样本ID；`month` 是审批月份；`prediction_sgd` 为预测月租；`ensemble_std_sgd` 为模型分歧；`pi90_lower_sgd/upper_sgd` 和 `pi95_lower_sgd/upper_sgd` 是相应预测区间。测试集没有真实标签，因此没有测试 RMSE。

如需比赛的两列格式，可从任一预测文件取出 `Id` 和 `prediction_sgd`，将后者改名为 `Predicted`。区间文件本身包含额外列，不直接作为两列提交文件。

## 复现与核验

主要实验在 Python 3.12、CPU、PyTorch 2.14.1、TabM 0.0.3、rtdl-num-embeddings 0.0.12 上运行。网络训练使用两个 Torch 线程，不依赖 CUDA。新建输出目录，避免覆盖已完成实验。

发布前的37项相关测试全部通过；归档核验另外复算5份逐条验证预测的RMSE，并检查6份测试预测各50,000条记录的ID顺序、月份和区间。数值检查不等于独立未来测试。

```bash
python -m pip install -r requirements-mlp-search.txt
python -m analysis.experiments.verify_mlp_delivery
python -m unittest tests.test_cleaned_neural_ensemble tests.test_neural_ensemble_uncertainty tests.test_historical_risk_correction tests.test_adjusted_rent_history tests.test_tabm_numeric_gradients

# 重新生成未清洗基线、校准与测试预测
python -m analysis.experiments.neural_ensemble_uncertainty --output-dir reports/local_runs/mlp_uncertainty --predict-test

# 使用刚生成的基线设置，重新进行两套清洗实验
python -m analysis.experiments.cleaned_neural_ensemble --baseline-dir reports/local_runs/mlp_uncertainty --output-dir reports/local_runs/mlp_cleaning
```

macOS 若遇到 OpenMP 动态库问题，可在相应命令前设置 `DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib`；无图形界面环境可设置 `MPLBACKEND=Agg`、`MPLCONFIGDIR=/tmp/hdb-mpl`。数值库和硬件差异可能产生小幅结果差异。只有设置、数据及代码哈希一致时，训练脚本才接受 `--resume`。

[交付清单](../reports/experiments/mlp_summary/delivery.json)区分点预测历史结果与测试文件；[文件清单](../reports/experiments/mlp_summary/manifest.json)保存精简证据的SHA-256。验证命令复算归档验证集RMSE并核对测试ID、月份与区间顺序，不需要下载模型。模型权重和预处理缓存留在本地，不随本次分支提交；重新训练命令会生成它们。
