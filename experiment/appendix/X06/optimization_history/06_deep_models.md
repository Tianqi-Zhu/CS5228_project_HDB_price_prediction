# 第六轮：更强深度表格模型的实测

本轮完成4类深度模型、19次较早窗口训练（15种基础配置＋4次补充种子），连同两个后续窗口共39次训练，覆盖数值嵌入、参数共享集成、优化过的MLP与学习式检索。使用同一train＋静态auxiliary历史特征，不使用公开租金目标，不接收验证期租金更新。

## 模型与实现

- [TabM作者实现](https://github.com/yandex-research/tabm)：`tabm==0.0.3`，32个参数共享成员、3×256结构。比较普通输入/分段线性数值嵌入、原始/残差目标及两种学习率。训练分别计算每个成员的MSE，预测时平均。
- [数值嵌入作者实现](https://github.com/yandex-research/rtdl-num-embeddings)与[RTDL MLP](https://github.com/yandex-research/rtdl-revisiting-models)组合为MLP-PLR；周期数值嵌入、类别Embedding和3×256 MLP。
- [RealMLP作者发布的独立PyTorch实现](https://github.com/dholzmueller/pytabkit/tree/main/pytabkit/models/realmlp)，来自`pytabkit==1.8.0`。原生缺失数值处理、稳健缩放、平滑截断、PBLD嵌入、参数学习率因子与调度；测试1/4个成员。该独立版是作者标注的实验实现，与原Lightning版本存在初始化和权重衰减差异，不能宣称逐位等价。
- [ModernNCA作者源码](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py)：固定Git提交，源码未修改，另写训练适配器。学习数值/类别表示及邻域权重，从拟合期租金检索；比较普通/周期嵌入，以及作者默认结构。训练候选池限制4096/8192行，排除当前训练批次；推理使用全部拟合记录，缓存编码分块计算。

所有模型均从随机初始化学习，不额外引入新的租金数据或预训练目标。此前TabPFN/TabICL实验保留作另外的预训练路线证据。依赖安装在`.cache/deep-deps`，未改动现有Conda环境。

## 输入和时间协议

候选输入为62列强历史特征＋5列过去市场/同期历史收缩特征。只根据拟合集删除常量/全缺失列；因此不同起点实际列数可以不同。类别词表、数值变换、PLE分箱以及RealMLP稳健统计都仅由相应拟合集学习。残差预测在输出时只加回由过去租金构造的基准。

2023-10—2024-03的17,977行用于选择配置、停止epoch与融合权重；2024-04—2025-03年度窗口35,158行，以及2024-06—2025-03近期窗口28,995行，用各起点前全部标签按早期选定epoch重训。后续标签不用于早停。训练允许检索全部拟合期标签；检索上下文不含验证目标，当前训练查询不能从候选池检索自己的标签。

TabM和MLP-PLR最多60个epoch；RealMLP最多128个epoch，并保留128-epoch学习率调度用于固定epoch重训；ModernNCA最多50个epoch。早停耐心和有限上下文预算见`plan.json`。这是有预算的实测，未穷尽所有调参可能。

## 同窗口模型对比

每条路线的代表配置仅根据较早窗口选择；三种子均值使用预先固定42/43/44，所有种子均参与平均。未在后续窗口挑选最低分种子。

| 路线 | 年度RMSE | 近期RMSE | 较早窗口选定配置 |
|---|---:|---:|---|
| 官方TabM | 456.1612 | 461.0924 | `seedmean__tabm_ple_residual_lr0.0005` |
| 作者RealMLP独立版 | 458.0752 | 448.4527 | `seedmean__realmlp_residual_ens4` |
| MLP-PLR | 475.9378 | 448.0631 | `mlp_plr_residual` |
| 作者ModernNCA | 461.3778 | 463.9001 | `modernnca_residual_plr` |

作为既有神经路线参照，第三轮强历史输入的FT-Transformer年度/近期RMSE为471.5338/458.6012，逐行预测已重新核对。它同样使用67个候选输入与残差目标，但旧版预处理、网络和训练轮数不同；因此该对照说明整条训练方案的进展，不是仅更换架构的严格消融。

### 后续窗口全部重训配置

以下保留全部后续结果，包括可能优于早期代表配置的单次运行；按近期分数展示只用于完整记录，不据此重选融合成员或权重。

| 配置 | 年度RMSE | 近期RMSE |
|---|---:|---:|
| `realmlp_residual_ens4_seed43` | 456.6087 | 447.8440 |
| `mlp_plr_residual` | 475.9378 | 448.0631 |
| `seedmean__realmlp_residual_ens4` | 458.0752 | 448.4527 |
| `realmlp_residual_ens4` | 459.8459 | 449.5217 |
| `realmlp_residual_ens4_seed44` | 459.9596 | 450.6923 |
| `tabm_ple_residual_lr0.0005` | 456.4600 | 460.4710 |
| `tabm_ple_residual_lr0.0005_seed43` | 457.4200 | 460.4771 |
| `seedmean__tabm_ple_residual_lr0.0005` | 456.1612 | 461.0924 |
| `modernnca_residual_plr` | 461.3778 | 463.9001 |
| `tabm_ple_residual_lr0.0005_seed44` | 456.4843 | 463.9748 |
| `tabm_ple_residual_lr0.002` | 453.6567 | 465.1498 |
| `tabm_ple_raw_lr0.002` | 455.2046 | 486.1951 |

### 数值表示的较早窗口对照

TabM在相同候选输入、目标和学习率下比较普通数值输入与PLE；各配置独立早停，嵌入版本参数量也增加，因此这是训练方案对照。以下只作为筛选证据，未对全部普通输入版本重跑后续窗口，不能声称其增益已经跨窗口验证。

| 目标 / 学习率 | 普通输入RMSE | PLE输入RMSE | 差值 |
|---|---:|---:|---:|
| raw / 0.002 | 527.3965 | 523.9942 | -3.4023 |
| residual / 0.002 | 526.6436 | 523.9029 | -2.7406 |
| residual / 0.0005 | 526.8906 | 523.3821 | -3.5085 |

## 与原最优融合的比较

| 窗口 | 原方案RMSE | 新深度模型融合RMSE | 差值 | 新MAE | 改善月份 |
|---|---:|---:|---:|---:|---:|
| outer | 451.1810 | 450.4924 | -0.6886 | 329.9547 | 12/12 |
| june | 442.9025 | 444.6888 | +1.7863 | 328.8985 | 3/10 |

新融合没有降低近期窗口RMSE，因此保留此前442.9025的历史主方案；已有可提交CSV不被覆盖。年度窗口的改善单独报告，不宣称所有时间范围均提升。


三个月连续月份块重采样的差值区间仅作开发期误差变化描述，不能修正多轮搜索和模型选择的偏差：

| 窗口 | 新减旧RMSE的描述性95%区间 |
|---|---:|
| outer | [-0.9041, -0.5015] |
| june | [-0.1952, +3.4086] |

三种子结果（单次RMSE的标准差与平均预测的RMSE含义不同）：

| 窗口 | 配置 | 单次RMSE均值 ± 标准差 | 三次预测平均RMSE |
|---|---|---:|---:|
| screen | `tabm_ple_residual_lr0.0005` | 523.5183 ± 0.2076 | 523.1748 |
| outer | `tabm_ple_residual_lr0.0005` | 456.7881 ± 0.5473 | 456.1612 |
| june | `tabm_ple_residual_lr0.0005` | 461.6410 ± 2.0211 | 461.0924 |
| screen | `realmlp_residual_ens4` | 524.0615 ± 0.4386 | 523.3840 |
| outer | `realmlp_residual_ens4` | 458.8047 ± 1.9026 | 458.0752 |
| june | `realmlp_residual_ens4` | 449.3527 ± 1.4316 | 448.4527 |

融合采用较早窗口上非负、和为1的权重，每个新候选最高0.5，并以固定0.001系数向原融合收缩；没有用后续窗口重新调权。单模型和融合的收益分别呈现，不能把不同特征/目标版本解释成架构的纯因果贡献。

候选池同时包含种子42的原配置与三种子均值，因此最终融合的各个种子有效权重可以不同；它们仅由较早窗口选定。`effective_blend_weights.json`展开所有重复成员，避免把整个最终融合误称为严格等权的三种子平均。

430目标尚未达到。所有后续窗口已被前几轮研究查看，当前分数仍是开发证据；三种子标准差只描述初始化敏感性，不代表未来市场误差的不确定性。两个后续窗口重叠，不合并样本量。

## 核验与文件

独立核对45份预测的ID、月份、标签、样本数和重算指标，重载20个后续模型复现预测；三种子均值和最终融合重新组合一致。6项专门测试通过，覆盖作者接口、拟合集统计、未知类别、成员独立损失、训练查询排除及缓存检索等价性。源数据和作者源码哈希已记录。

- `selected_model_comparison.csv`：各架构的年度/近期对照。
- `verified_metrics.csv`：全部配置、epoch、时间和PyTorch峰值显存（不是物理驻留显存）。
- `seed_variation.csv`：种子变化及等权平均结果。
- `ensemble_selection.json`：在较早窗口锁定的权重与代表模型。
- `author_source.json` / `verification.json`：源码来源、版本和完整核验。

## 复现

需要现有CUDA PyTorch环境，以及`.cache/model-deps`中的RTDL依赖。新增作者包单独安装，不覆盖现有环境。源码下载器固定本轮提交，并记录哈希；具体提交见`author_source.json`。

```powershell
python -m pip install --no-deps --target .cache/deep-deps -r requirements/optional_deep_models.txt
python -m analysis.experiments.sixth_wave_sources
python -m analysis.experiments.sixth_wave_deep screen
python -m analysis.experiments.sixth_wave_ensemble select
python -m analysis.experiments.sixth_wave_ensemble outer
python -m analysis.experiments.sixth_wave_ensemble june
python -m unittest tests.test_sixth_wave
python -m analysis.experiments.sixth_wave_report
```

融合使用此前实验保存的逐行预测，因此需保留第二至四轮输出。已有结果默认跳过；后续评估仅使用较早窗口保存的轮数和权重。
