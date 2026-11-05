# 第十轮：模型结构实际训练对照

本轮探索数值token、显式交叉和条件专家三条神经结构路线；含原始线性token控制组，共9个较早窗口配置。另并行探索2个非对称CatBoost树结构。神经网络从较早窗口选每条路线代表和轮数，后续固定轮数训练；树模型固定原对称树选出的833轮。神经网络统一使用去掉五列36月同期统计的62个候选输入，并学习相对24月历史基准的残差。树模型沿用原对称树的62列window历史输入、直接预测租金；两组62列的内容不同。

## 结构与实现来源

- FT-Transformer：使用[作者RTDL骨干](https://arxiv.org/abs/2106.11959)，比较线性、分段线性PLE和周期PLR数值token；嵌入来自[作者实现](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md)，方法对应[数值嵌入论文](https://arxiv.org/abs/2203.05556)。96宽2层与128宽3层为两个容量配置。新组合由项目适配，不宣称逐位复现论文成绩。
- Deep & Cross：按[DCN-V2](https://arxiv.org/abs/2008.13535)的全矩阵交叉公式实现显式交互分支，并联两层深度分支。比较线性嵌入＋2个交叉层、PLR嵌入＋3个交叉层；为项目实现，未宣称作者代码原样复现。
- 条件专家：项目实现，共享PLR/类别嵌入和128维主干，4或8个回归专家，软门控仅使用输入特征；MSE加0.001负载均衡项，没有根据查询真实租金指定专家。

- 非对称树：按[CatBoost官方参数](https://catboost.ai/docs/en/references/training-parameters/common#grow_policy)比较Depthwise深度6和Lossguide深度8/最多64叶；最小叶样本20。与原CPU对称树保持输入、目标、833轮、学习率0.045、L2=15和seed42一致，其余结构容量限制单独记录，不能将效果归因于一个参数。

## 完整窗口单模型结果

| 代表模型 | 年度RMSE | 近期RMSE |
|---|---:|---:|
| cat_symmetric_saved | 456.4077 | 446.3793 |
| cat_depthwise6 | 460.1803 | 448.0859 |
| ft_ple_128_3 | 458.2931 | 449.8158 |
| ft_plr_128_3 | 462.0714 | 451.0982 |
| dcn_plr_3 | 461.5278 | 452.1061 |
| ft_linear_96_2 | 461.9852 | 458.9254 |
| moe_plr_8 | 458.6529 | 478.9720 |

单模型与融合的指标分开比较。它们是有限预算、单随机种子的结构探索，不能据此判定某架构普遍优劣。

## 固定融合规则下的增量

保持第九轮24/36月覆盖分组、月度中心化误差、全局及组内正则0.003不变，仅改变是否加入新模型预测。每个原点重新拟合融合权重，但不再调融合超参数。baseline完整复现441.9591/450.8650，避免将融合规则改变算成结构收益。

| 新增路线 | 年度RMSE | 近期RMSE |
|---|---:|---:|
| only_ft_plr | 451.0295 | 441.8345 |
| baseline | 450.8650 | 441.9591 |
| only_tree | 450.8034 | 441.9752 |
| only_ft_ple | 450.7508 | 441.9845 |
| only_dcn | 450.8395 | 442.1377 |
| only_ft_linear | 450.7687 | 442.2048 |
| all_new | 450.8987 | 442.5100 |
| all_new_plus_tree | 450.8121 | 442.5193 |
| only_moe | 450.8200 | 442.8225 |

本轮最低候选 `only_ft_plr`：近期 **441.8345**，年度 **451.0295**。这是后期开发集比较后的选择，不能当作预先锁定的独立测试最佳。
相对第九轮，近期降低0.1245，10个月中7个月改善；年度反而升高0.1645。PLR单模型不是本轮最低单模型，但在固定融合规则下有互补性。相关波动见[月度对照](../../output/experiment_runs/round_10_plr_transformer_final/monthly_comparison.json)，不能据此直接推断Kaggle分数会下降。
已全量重训所需新模型并导出[50,000行新CSV](../../output/submissions/candidates/kaggle_structure_blend.csv)，用户已提交，截图显示Kaggle **415.37529**；在历史版415.86912、宏观版415.78232和本方案三份提交中选择本方案作为final。

## 验证口径

本轮共23次基础模型拟合（19次神经网络、4次树模型），另有9种纳入路线×3窗口的融合权重流程。基础训练只访问拟合标签；后期不用查询标签早停。融合年度标签截止2024-03，六月截止2024-05；全量融合截止2025-03。较早窗口本身已用于模型、epoch和历轮特征选择，因此所有分数仍是反复查看的开发证据，年度和近期窗口部分重叠。
早期内部选择的神经网络融合纳入路线为 `all_new`；加入树模型的额外对照另行记录。所有23份基础模型及27份融合权重均重新加载复现，baseline匹配此前结果，9项相关结构/信息边界测试通过。单模型表包含不同输入组，不能跨组作架构公平排名。

[单模型全部记录](../../output/experiment_runs/round_10_plr_transformer_final/model_metrics.csv)、[融合全部记录](../../output/experiment_runs/round_10_plr_transformer_final/ensemble_metrics.csv)、[选择记录](../../output/experiment_runs/round_10_plr_transformer_final/selection.json)、[复现核查](../../output/experiment_runs/round_10_plr_transformer_final/verification.json)、[总结](../../output/experiment_runs/round_10_plr_transformer_final/summary.json)。

最终报告选择：[FINAL_MODEL.md](../01_final/README.md)。
