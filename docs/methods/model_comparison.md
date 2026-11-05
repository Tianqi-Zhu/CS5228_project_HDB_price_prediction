# 早期模型搜索：协议、配置与复现

本页保留各模型可选择自身特征/残差表示的完整管线搜索口径。
新增的 [同27列模型对照](../../experiment/03_model_comparison/PROTOCOL.md) 固定输入和拟合行，
更适合讨论架构差异；当前整体叙述见 [项目报告](../../experiment/02_feature_engineering/BASELINE_27_FEATURES.md)。不把两张表拼成同一架构排名。

本轮比较单条 HDB 月租金，所有最终误差均反变换为 SGD 后计算 RMSE。
不将区域均价、标准化目标误差或对数误差当作该指标。

该阶段结果在 [`output/experiment_runs/comparison_model_search/model_metrics.csv`](../../output/experiment_runs/comparison_model_search/model_metrics.csv)，
汇总与目标是否达到的状态在 `summary.json`，图表为 `model_comparison.png`。
每个模型目录保留选中的参数、特征版本、内部验证分数、固定重训轮数和外层分数。

该阶段完成30组内部候选配置及10类模型的外层训练。分数以对应结果表为准，本页保留协议和复现配置。

## 时间验证协议

主要外层验证为 **2024-04 至 2025-03 的全部 35,158 条记录**，拟合数据截止 2024-03。
其中 2023-10 至 2024-03 的 17,977 行只用于内部选择特征版本、参数和训练轮数；
确定选择后，在外层起点之前的全部拟合数据上按固定轮数重训。
外层验证不参与早停，也不选择融合权重。

2023-04、2022-04 起的两个年度窗口及2023-12起十六个月窗口用于复核部分树模型，分别在各自起点之前的六个月
完成内部选择。不同年度的分数单独保留，不能把较容易的窗口当作整体表现。

外层月份在以往项目实验中已被查看过，本轮仍是方案选择证据，不是新独立测试成绩。
这种嵌套规则减少本轮的直接调参偏差，并不能消除过去的重复选择。

## 模型与特征适配

| 模型 | 主要比较配置 |
| --- | --- |
| LightGBM | 31/63叶节点、精简历史/历史溢价/市场残差；内部早停 |
| CatBoost | 深度6/8、原生类别、地址类别、市场残差；CPU训练，内部早停 |
| XGBoost | 深度4/6、原生类别、精简/溢价/市场残差；CPU hist，内部早停 |
| HistGradientBoosting | 400轮、精简或市场残差；类别限制在低基数字段 |
| ExtraTrees | 300棵树、叶节点最小样本数5/10；类别One-hot，数值训练集填补 |
| Embedding MLP | 类别Embedding + 3层256维MLP，比较精简、地址、市场残差输入 |
| FT-Transformer | 官方实现，3层96维；类别和数值token，比较相同特征版本 |
| TabNet | 官方PyTorch实现，24/32维决策与注意力、4步、类别Embedding、entmax |
| Embedding ResNet | 类别Embedding + 3个256维残差块；地址类别尝试低频合并 |
| FT-Transformer 192 | 官方192维规格，学习率1e-4、weight decay 1e-5、较小批量 |

FT-Transformer 使用 [作者实现](https://github.com/yandex-research/rtdl-revisiting-models)，
TabNet 使用 [DreamQuark实现](https://github.com/dreamquark-ai/tabnet)。
ResNet为本项目实现，神经网络训练与预处理由本项目统一管理。

所有候选及参数清单见 `analysis/experiments/model_comparison/model_search.py`，神经网络接口与预处理见
`analysis/pipeline/neural_models.py`。

### 特征版本

- `compact`：前一轮选定的27列。
- `premium`：历史均价加历史同户型月度溢价的研究版本。
- `entities`：27列再加入街道、地址键、Subzone三项类别，供Embedding和原生类别模型选择。
- `entities_residual`：地址版本再加入过去同户型市场背景，预测相对最近市场基准的残差。
- `compact_residual`：低基数精简版本的市场残差，适配 sklearn 的类别数量限制。

神经网络的类别词表、数值中位数、标准化或分位数变换均仅在相应拟合集拟合。
历史计数先做 log1p，拟合时有缺失的数值列另附缺失指示。连续时间序号保留线性标准化，
避免分位数变换将未来时间直接截成训练期最大值。
未知类别编码为0；部分新增ResNet/大规格FT候选将训练中少于10次的类别合并到0。

市场残差版本丢弃最初六个拟合月份以提供历史预热；这些已知标签仍可用于构建后续
月份的历史。每个模型实际使用的拟合行数已单独记录，验证行始终全部保留。
因此表格比较的是各模型通过内部验证选择的完整管线，不能将全部差异归因于架构本身。

## 融合与误差诊断

`blend` 从内部验证分数最好的三个已完成模型中，在内部预测上选择非负、和为1的权重。
之后直接对外层预测应用这些权重。不会依据外层分数反复寻找最优权重。

`report` 汇总该阶段分数并生成图表，并输出最佳观察结果按区域、户型、租金段的
误差及最大误差记录。逐行预测和二进制模型保留在本地，不进入Git。

`data_diagnostics.json` 是单独的数据可区分性检查：

- 最新外层中，相同原始输入的记录可有不同租金；即使使用这些验证标签计算各组最优
  均值，样本内 RMSE 仍为150.86。该数值只是诊断，绝不作为模型验证成绩。
- 重复输入组的合并组内标准差约459.96。它提示未观察差异较大，但只是重复样本的描述性
  估计，不能据此宣称全体样本的理论误差下界为460，或严格证明某个误差目标不可能。

这些记录可能代表同楼栋内不同租约；给定字段没有出租单元楼层、室内状况等具体差异。
因此应同时检查信息不足、时间变化和模型欠拟合，不能只通过增大网络来推断某个误差目标可达。

## 环境与复跑

核心包统一由`environment.yml`声明，包含RTDL、TabNet和XGBoost。
既有`.cache/model-deps/`仍可兼容读取，新环境从主YAML安装即可。

```powershell
# 新建核心环境，已有环境无需重复创建
conda env create -f environment.yml
conda activate cs5228

# 准备严格分开的内部/外层特征缓存
python -m analysis.main model-search prepare --fold 2024-04

# 训练全部模型，或通过 --families 选择子集；已完成的任务自动续用
python -m analysis.main model-search run --fold 2024-04
python -m analysis.main model-search run --families embedding_mlp ft_transformer tabnet

# 内部验证选权融合及最终报告
python -m analysis.main model-search blend --fold 2024-04
python -m analysis.main model-search diagnose
python -m analysis.main model-search report
```

更换搜索配置或增加训练预算时，应使用新的 `--output-dir` 并先准备对应缓存，避免将
已经完成的任务误认为新配置的结果。单个候选与单个模型完成后即写入结果，可从中断处续跑。
