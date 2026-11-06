# M08 Matched F1/F2 evaluation

| Item | Status |
|---|---|
| Requirement | See [requirement cards](../../../docs/planning/main-appendix-experiment-plan.md) |
| Current execution | recent J paired evaluation complete; annual Y pending |
| Scientific scope | Retrospective development; original data and predictions preserved |
| Implementation | `python -m analysis.main final-pair` or `planned` |

## New evidence

- [protocol_checks.json](evaluation/protocol_checks.json)
- [final_pair_metrics.csv](evaluation/final_pair_metrics.csv)
- [monthly_pair.csv](evaluation/monthly_pair.csv)
- [grouped_pair.csv](evaluation/grouped_pair.csv)

## Existing evidence



On the same 28,995 recent records, F1 RMSE/MAE are 441.8345/321.8208 and F2 440.9520/318.8879 SGD. F2 has lower monthly RMSE in nine of ten months. Both are development pipelines with different inputs and pretrained sources: common IDs and dates do not prove identical information policies or an isolated architecture effect. Annual F2 predictions and the complete information-policy audit remain unavailable. No full-data model is used to backtest records it fitted.
