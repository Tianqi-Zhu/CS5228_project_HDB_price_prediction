# MLP 集成预测与结果说明

本分支交付 MLP、TabM、融合预测、训练数据清洗对照、预测区间，以及MLP全量重训后的Kaggle提交。原始 CSV 保持不变；新增代码、精简实验记录和可直接查看的预测文件一起保存。

**最新Kaggle截图成绩：全量清洗MLP为437.34455，全量未清洗MLP为439.38193。** 这是用户在2026-11-05提供的提交结果；截图未标明公榜／私榜，未通过API独立核验。当前推荐的MLP候选为全量清洗版，仅表示下列四份MLP提交中显示分数最低。

原独立区间校准协议下，未清洗融合模型的本地验证RMSE为467.03，清洗后的纯MLP为471.26。历史最低开发验证464.21来自另一套训练截至2024-03的神经网络融合。以下将Kaggle显示成绩、旧模型的时间验证结果和全量训练结果分别注明，避免混算。

## Kaggle提交与全量重训（2026-11-05）

| 提交版本 | 训练截止 | 实际训练行数 | Kaggle显示分数 |
| --- | --- | ---: | ---: |
| 冻结模型，未清洗MLP | 2023-12 | 105,837 | 461.64312 |
| 冻结模型，明显低价清洗MLP | 2023-12 | 104,505 | 458.91884 |
| 全量重训，未清洗MLP | 2025-03 | 150,000 | 439.38193 |
| **全量重训，明显低价清洗MLP** | **2025-03** | **147,983** | **437.34455** |

全量重训后，未清洗版分数降低22.26119，清洗版降低21.57429（约4.70%）；全量清洗版比全量未清洗版低2.03738。原结构、AdamW配置、seed 42/43/44及每个网络3个epoch均保持不变。拟合集、过去租金统计和预处理随重训更新，训练批次数也增加，因此这是全量重训管线的增量，不能把所有变化单独归因于样本量。时间序号仍使用原分位数编码，本轮没有加入新的时间表示、同期历史或其他架构。

原本用于2024-01至2024-03区间校准、2024-04至2025-03本地验证的44,163条已知标签，现在进入最终拟合集。清洗规则重新应用于全部150,000条记录，共删除2,017条、保留147,983条，其中43,478条来自2024-01至2025-03。全部50,000条测试记录均保留；训练、预测均未读取测试真实租金。

测试集中有380条2025-03记录与最后训练月重叠。全量监督拟合包括该月已知训练标签，但这些查询的历史统计和历史市场基准只读取2025-03之前的数据。因此该重叠子集不能称为完全事前拟合的评估。6个MLP保存重载后的预测最大差为0；每份CSV均核对50,000行、`Id,Predicted`列名、ID顺序、有限预测值和三种子均值。

全量拟合后没有保留新的独立本地测试集，不能在原验证月份上重新报告独立验证RMSE。上表外部成绩来自用户截图；全量运行摘要中的`test_rmse_sgd: null`表示本地没有隐藏标签，截图分数另存于[kaggle_results.json](../reproduction/mlp/artifacts/fullfit/kaggle_results.json)。

- [全量清洗MLP提交CSV](../reproduction/mlp/artifacts/fullfit/submission_mlp_fullfit_cleaned.csv)
- [全量未清洗MLP提交CSV](../reproduction/mlp/artifacts/fullfit/submission_mlp_fullfit_uncleaned.csv)
- [训练摘要](../reproduction/mlp/artifacts/fullfit/summary.json)、[逐行检查](../reproduction/mlp/artifacts/fullfit/verification.json)、[删除清单](../reproduction/mlp/artifacts/fullfit/obvious_low/removed_training_rows.csv)
- [全量成绩截图](../reproduction/mlp/artifacts/fullfit/evidence/kaggle_fullfit_2026-11-05.png)、[原冻结模型成绩截图](../reproduction/mlp/artifacts/fullfit/evidence/kaggle_frozen_2026-11-05.png)

这些全量CSV为点预测，没有沿用旧模型的90%／95%区间。旧模型的区间覆盖率也不能转移到本次全量模型。

## 时间衰减权重：本地对照

新增可选参数`--half-life-months`。训练行的原始权重为`2 ** (-距最后拟合月的月数 / 半衰期)`，再归一化到训练行平均权重为1；回归与辅助价格档分类损失使用同一权重。该参数只改变训练损失，房屋输入、历史统计定义与原分位数预处理保持不变。省略参数时保持原来的均匀权重。

本轮沿用明显低价清洗、3轮训练和seed 42/43/44，比较不加权及6／12／24／36个月半衰期。先在2023-04至2023-09、2023-10至2024-03两个不重叠的较早窗口比较，评估指标均不加权且保留全部验证行。合并RMSE为577.6236／578.5251／578.1931／577.8959／577.7108：总体上不加权最低，正半衰期候选中36个月最低。固定该加权候选后再评估后续年度窗口。

| 年度验证：2024-04至2025-03，35,158行 | RMSE | MAE |
| --- | ---: | ---: |
| 不加权 | 470.2343 | 341.4548 |
| 36个月半衰期 | 469.7138 | 341.4755 |

两版均在截至2024-03的113,324条清洗后记录上拟合。年度RMSE降低0.5205，但MAE几乎不变、早期合并结果也未获胜，当前不足以称为稳定提升。固定3轮有利于隔离权重变化，尚未为每个半衰期重新选择训练轮数。全部本地窗口已在历次研究中查看，属于开发对照。

36个月半衰期下，相对最新记录，一年前、两年前、三年前的原始权重约为0.794／0.630／0.500。已另行训练全量加权候选供Kaggle对照；当前437.34455仍属于原不加权的全量清洗版，不移记为加权成绩。全量重训与参数搜索入口分别为：

```bash
python -m analysis.main mlp fullfit --half-life-months 36 --output-dir reports/local_runs/mlp_weighted_h36
python -m analysis.main mlp recency-search --output-dir reports/local_runs/mlp_recency_search
```

## 输入与网络

主要输入包括户型、房屋型号、Town、街道和楼栋 ID，面积、时间、房龄、位置、楼栋属性、设施距离/数量，以及严格来自过去的租金统计。街道和楼栋使用可学习的类别嵌入，少于四条的类别归入未知/稀有桶；数值缺失填补与分位数变换只在当前拟合部分学习。

当前独立 MLP 将类别嵌入和数值特征拼接，经过宽度 128 的输入投影及两个 128 单元全连接块，使用 SiLU、BatchNorm/LayerNorm 与 Dropout。它预测市场调整后的历史基准与真实租金之间的残差，再加回基准得到 SGD 租金；训练时另有价格档辅助分类头，预测时不读取价格档真实标签。AdamW 学习率 0.0005，batch 1,024。

TabM 使用官方 `tabm==0.0.3`，配合类别嵌入和分段线性数值嵌入，宽度 128、两个块、每个网络 16 个共享参数成员，AdamW 学习率 0.001、batch 512。它预测标准化租金，再还原为 SGD。

Deep Ensemble 分别用 seed 42、43、44 训练三个独立 MLP；TabM 也进行三个独立随机种子重训。先对每个 TabM 的内部成员求均值，再平均三个随机种子。融合为 MLP 集成与 TabM 集成各 50%。共享参数成员不视为 48 个完全独立网络。

此处的主要集成没有加入 RNN/LSTM latent vector。早期加深和 latent 实验作为历史对照保留，不能把所有尝试都解释为收益。

## 原区间实验的时间划分

| 阶段 | 数据 | 用途 |
| --- | --- | --- |
| 选择轮数 | 2021-01～2023-09 拟合，2023-10～2023-12 验证 | 选定 MLP 3 个 epoch、TabM 4 个 epoch |
| 重训独立网络 | 2021-01～2023-12，共 105,837 条 | 三个随机种子，所有训练期预处理重新拟合 |
| 区间校准 | 2024-01～2024-03，共 9,005 条 | 只校准区间，不再选择网络或融合权重 |
| 完整验证 | 2024-04～2025-03，共 35,158 条 | 比较 RMSE、MAE、偏差与覆盖率 |
| 测试预测 | `test.csv` 全部 50,000 条 | 保持样本提交的 ID 顺序，没有真实标签可评分 |

为保持原校准流程一致，原六份带区间CSV使用截至2023-12的冻结模型。其验证跨度为拟合后的第4～15个月，测试跨度为第15～31个月；更远月份的预测与区间属于超出已验证跨度的外推。上节新增的两份全量提交另行保存，训练截止已更新到2025-03。

## 原冻结模型的本地验证结果

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

清洗对照的完整九行模型结果见[清洗指标](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/comparison_metrics.csv)。清洗提高预测价格，改善高租金组，却加剧低租金组误差：[分组结果](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/cleaning/rent_band_metrics.csv)。

历史搜索中最好的点预测融合为 **RMSE 464.21、MAE 336.94**，使用 114,842 条 2021-01～2024-03 拟合记录，外层验证仍为上述 35,158 条。它同时改变了训练量、特征和模型组合，且外层月份已重复查看；没有证据将全部提升单独归因于加深网络。本次打包提供其[逐条验证预测](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/validation/best_point_blend_outer.csv)，未提供与其对应的测试区间文件。

普通独热编码 MLP 的 6/8 层无 latent 结果为 488.35/484.79，加入 RNN/LSTM 并未稳定改善。可学习嵌入、历史租金、残差目标、数值表示与模型融合构成后续实验方向。完整正负结果见[历史实验目录](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/historical_catalog.csv)，其中早期数值嵌入失效的实验已单独标记，不能作为正确分段嵌入的比较结果。

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

相同输入对应多个标签会形成局部限制。例如验证期同一组输入的真实租金为4,000/5,000/6,000，统一预测5,000时组内RMSE最低，仍为816.50；当前未清洗融合预测5,028.49，组内RMSE816.99，已经接近这个局部最优。这只证明该组三条的限制，不能证明整体RMSE无法降至450。[案例与计算](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/uncertainty_diagnostics/indistinguishable_inputs_proof.json)

## 原区间预测文件

下列每份均为50,000条、按样本提交ID排序。清洗版本和风险修正版本保留作候选对照，未覆盖原预测。

| 文件 | 对应模型与用途 |
| --- | --- |
| [未清洗 MLP](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/uncleaned_mlp_test_intervals.csv) | 独立 MLP 集成对照，验证 RMSE 471.80 |
| [未清洗融合](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/uncleaned_hybrid_test_intervals.csv) | 主要融合对照，验证 RMSE 467.03 |
| [明显低价清洗 MLP](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/obvious_low_mlp_test_intervals.csv) | 本轮清洗中纯 MLP 数值最好，验证 RMSE 471.26 |
| [明显低价清洗融合](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/obvious_low_hybrid_test_intervals.csv) | MAE 337.00，验证 RMSE 468.79 |
| [严格高价清洗融合](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/highest_only_hybrid_test_intervals.csv) | 失败对照，验证 RMSE 476.25 |
| [历史风险修正融合](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/predictions/risk_corrected_hybrid_test_intervals.csv) | RMSE 465.34，但 MAE 略高的候选 |

`Id` 对应比赛样本ID；`month` 是审批月份；`prediction_sgd` 为预测月租；`ensemble_std_sgd` 为模型分歧；`pi90_lower_sgd/upper_sgd` 和 `pi95_lower_sgd/upper_sgd` 是相应预测区间。测试集没有本地可用的真实标签，因此本地不能计算测试RMSE；已有Kaggle截图成绩见首节。

如需比赛的两列格式，可从任一预测文件取出 `Id` 和 `prediction_sgd`，将后者改名为 `Predicted`。区间文件本身包含额外列，不直接作为两列提交文件。

## 复现与核验

主要实验在 Python 3.12、CPU、PyTorch 2.14.1、TabM 0.0.3、rtdl-num-embeddings 0.0.12 上运行。网络训练使用两个 Torch 线程，不依赖 CUDA。新建输出目录，避免覆盖已完成实验。

发布前的37项相关测试全部通过；归档核验另外复算5份逐条验证预测的RMSE，并检查6份测试预测各50,000条记录的ID顺序、月份和区间。数值检查不等于独立未来测试。

```bash
python -m pip install -r requirements/mlp.txt
python -m analysis.main mlp verify
python -m analysis.main mlp test

# 重新生成未清洗基线、校准与测试预测
python -m analysis.main mlp intervals --output-dir reports/local_runs/mlp_uncertainty --predict-test

# 使用刚生成的基线设置，重新进行两套清洗实验
python -m analysis.main mlp cleaning --baseline-dir reports/local_runs/mlp_uncertainty --output-dir reports/local_runs/mlp_cleaning

# 使用归档的原模型配置与轮数，重训全量MLP并输出两份Kaggle格式CSV
python -m analysis.main mlp fullfit --baseline-dir reproduction/mlp/config/interval_baseline --output-dir reports/local_runs/mlp_fullfit
```

macOS 若遇到 OpenMP 动态库问题，可在相应命令前设置 `DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib`；无图形界面环境可设置 `MPLBACKEND=Agg`、`MPLCONFIGDIR=/tmp/hdb-mpl`。数值库和硬件差异可能产生小幅结果差异。只有设置、数据及代码哈希一致时，训练脚本才接受 `--resume`。

[交付清单](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/delivery.json)区分点预测历史结果与测试文件；[文件清单](https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/322fa4c20d83f20e19328823203873a0f81e0f4a/reports/experiments/mlp_summary/manifest.json)保存精简证据的SHA-256。验证命令复算归档验证集RMSE并核对测试ID、月份与区间顺序，不需要下载模型。模型权重和预处理缓存留在本地，不随本次分支提交；重新训练命令会生成它们。
