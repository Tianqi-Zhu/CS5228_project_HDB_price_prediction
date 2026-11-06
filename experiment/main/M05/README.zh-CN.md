# M05 Complex-model and fixed-rule blend controls

| Item | Status |
|---|---|
| Requirement | See [requirement cards](../../../docs/planning/main-appendix-experiment-plan.md) |
| Current execution | completed controlled recalculation; annual PLR fusion regression preserved |
| Scientific scope | Retrospective development; original data and predictions preserved |
| Implementation | `python -m analysis.main planned` |

## New evidence

- [controlled_structure_deltas.csv](evaluation/controlled_structure_deltas.csv)
- [selection_audit.json](evaluation/selection_audit.json)

## Existing evidence

- [common_27](common_27/)
- [concat_transformer](concat_transformer/)
- [selection_records](selection_records/)
- [structure_control](structure_control/)

The common-27 comparison, matched PLE/PLR controls and [Concat Transformer](concat_transformer/README.md) are separate comparisons. Frozen numeric weights differ when member sets change; the fitting rule is held fixed. Epoch/parameter budgets remain visible.

## 已整合的独立MLP管线

[MLP训练与提交](../../../reproduction/mlp/README.md)提供全量CPU重训、时间权重、区间/清洗和437.34455／439.38193原提交。它使用44个原始输入字段，与共同27列模型块、F1及Concat分别记录，不作为同输入架构排名；不修改原F1融合。
