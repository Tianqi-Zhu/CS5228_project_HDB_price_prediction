# main 新增要求卡执行状态

初轮完成原“新增实验要求卡”中 main 的离线审计和诊断，新增训练数为 0。本次进一步补齐[五项材料](main-five-item-completion.md)，新增 F1 Y/J 各 15 个固定配方基础模型，全部保存及重载核验；没有重新选择模型、权重或超参数。`planned` 本身仍只审计，不训练。科学验收以实际输入、对照与结果为准，未满足的要求继续登记。

| 卡片 | 本轮执行 | 当前状态 | 证据 |
|---|---|---|---|
| M00 | 旧 26 组预测审计；F1 固定生成链；F2 近期行 ID/标签和候选组件身份 | F1 已完成；F2 近期数值配对可执行，旧档案原生成提交未全恢复 | [协议注册表](../../experiment/main/M00/evaluation/protocol_registry.csv)、[新 F1 生成链](../../reproduction/f1/validation_evidence/audit.json) |
| M01 | 原始行数与数据政策；补入空间关联/受控分箱、清洗匹配对照和显式场景声明 | 已完成所给设施快照情景；原回归未独立重估，历史设施仍为 X08 | [数据政策](../../experiment/main/M01/evaluation/data_policy_audit.json)、[M01 新材料](../../experiment/main/M01/README.md)、[预测场景](../methods/prediction-scenario.md) |
| M02 | 复用已经核验和重训的 C7 传统模型表 | 既有实验可用；本轮不重复训练 | [C7 对照](../../experiment/main/M02/README.md) |
| M03 | 固定阶梯 16 组预测重算；统一有序 schema 与组件输入，纠正 PLE/PLR 旧元数据遗漏 | 已完成；两套 62 列输入单独命名 | [增量表](../../experiment/main/M03/evaluation/feature_ladder_deltas.csv)、[特征清单](../../experiment/main/M03/feature_schemas/README.md) |
| M04 | F1、62 列 LightGBM/CatBoost 的月份、户型、24/36 月支持组、拟合集 q10/q90 租金组；144 行分组记录；输入相同组诊断 | 已完成两窗匹配诊断 | [误差组](../../experiment/main/M04/evaluation/error_groups.csv)、[支持审计](../../experiment/main/M04/evaluation/feature_support_audit.json)、[相同输入](../../experiment/main/M04/evaluation/identical_input_diagnostics.csv) |
| M05 | PLE/PLR 单模型与固定融合规则下不新增/新增 PLE/新增 PLR 的配对比较；保留预算差异及负结果 | 已补齐可重算证据；共同 27 列块独立保留 | [结构增量](../../experiment/main/M05/evaluation/controlled_structure_deltas.csv)、[选择审计](../../experiment/main/M05/evaluation/selection_audit.json) |
| M06 | 复用三份历史提交的身份与选择依据 | 既有实验可用；历史/宏观替代方案留 appendix | [F1 选择](../../experiment/main/M06/f1_selection.json) |
| M07 | 核验七树＋TabPFN87 组成、固定等权公式、来源哈希和 50,000 行候选 | 候选组件重组通过；完整有序字典、截图关联待补 | [F2 清单](../../experiment/main/M07/evaluation/f2_manifest.json) |
| M08 | 相同 28,995 行近期 F1/F2 数值比较及月/房型/覆盖分组 | J 已完成；Y 与完整信息政策等价审计待补 | [协议检查](../../experiment/main/M08/evaluation/protocol_checks.json)、[配对指标](../../experiment/main/M08/evaluation/final_pair_metrics.csv) |
| M09 | F1 verify/CPU 重组/固定生成链；F2 候选组件重组 | F1 在所列范围通过；F2 组件重组通过，未将其当成新 GPU 推理 | [验收](../../experiment/main/M09/evaluation/verification.json)、[F2 重组](../../experiment/main/M09/evaluation/f2_reconstruction.json) |

## 本轮结果的读取范围

- Y：2024-04—2025-03，35,158 行；J：2024-06—2025-03，28,995 行。两个窗口重叠，不能合并成独立测试。
- 加入 PLR 相对同规则、无新增网络的融合对照，Y 的 RMSE **+0.164455** SGD，J **−0.124548** SGD。保留“单模型、融合、窗口可能给出不同结论”的实际结果。
- F1 在 Y 中低于拟合集 q10 的 834 条记录占总 SSE 约 **21.03%**；J 中相应 640 条记录占约 **19.40%**。这些组使用验证真实租金，仅供事后诊断，不能作为预测期特征或删除规则。
- “相同输入”只针对两种 62 列控制模型的实际有序输入；组内标签差异是本地描述性诊断，不是全局不可约误差下限，也不代表 F1 的完整异质成员输入。
- 原始缓存的预测及摘要有哈希，分支提交仅作来源背景；没有将来源分支名冒充每份缓存的确切生成提交。

## 执行与复现

```powershell
python -m analysis.main planned
python -m analysis.main final-pair
python -m analysis.validation.repository_audit
```

`planned` 使用 [5.8 MB 冻结预测包](../../experiment/main/M00/inputs/validation.npz)，不需要历史模型权重或原工作区缓存；独立重建支持分组时只读取原点前的训练记录，并在构建查询特征前移除租金。结果写入 `output/planning_validation/main/<内容摘要>/`，已存在的 run 不覆盖。可以用 `--output-dir output/planning_validation/check-2` 另建运行。

归档结果分别放入各卡片的 `evaluation/`；[本次摘要](../../experiment/main/evaluation-summary.json)记录执行状态，图表在 [M04](../../experiment/main/M04/README.md)。冻结输入的 [metadata.json](../../experiment/main/M00/inputs/metadata.json)包含原预测/配置文件哈希、schema、学习器设置与历史选参记录。重新整理原始缓存可用 `python -m analysis.validation.prepare_inputs --destination output/planning_validation/new-inputs`，该准备步骤需要所列本地历史产物，普通复算不需要。

## 必须继续补齐的材料

1. F2 的完整导出有序 87 列字典，以及现有候选与历史 413.19635 截图分数的确切关联；成员、等权公式、候选和推理配置现已可查。
2. F2 的年度 Y 原点前拟合预测，以及两管线完整输入/信息政策等价审计；近期 J 数值配对已完成。不能用全量 150,000 行模型回测已参与拟合的记录。
3. 旧缓存的原始生成代码树仍有来源缺口；新 F1 Y/J 已由固定提交重新生成并核验。严格历史设施情景若作为主张，仍需要 X08 的真实日期和匹配重训。

这些缺口意味着 **main 全部 P0 尚未完成**。当前可执行审计、诊断、F1 生成链核验、F2 组件重组与近期配对。远端新增的[独立 TabPFN 复现记录](../guides/tabpfn_best.md)和[MLP 复现包](../../reproduction/mlp/README.md)分别保留其训练环境与验证范围；它们不替代缺失的 F2 年度或截图证据。

## Whole-main portability check

Concat training used POSIX-only `fcntl`; the runner now provides Windows/POSIX exclusive locking and a data-copy fallback when symlinks require privileges. This changes runtime plumbing and new checkpoint fingerprints, not the architecture, frozen weights, original CSV or selection settings. All frozen artifacts were byte-checked against the prior integration. Raw feature preparation completed on 150,000/50,000 rows; a separate 32-train/16-query, two-epoch CPU smoke fit saved and reloaded successfully. No new research model selection or full eight-epoch refit was performed. Evidence: `review/checks/main-final/concat-small-training-smoke.json` and `reproduction/concat_transformer/portability.json`.
