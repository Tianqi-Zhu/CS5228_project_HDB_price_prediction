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
