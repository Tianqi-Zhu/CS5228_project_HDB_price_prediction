# M09 Independent final-method reproduction

| Item | Status |
|---|---|
| Requirement | See [requirement cards](../../../docs/planning/main-appendix-experiment-plan.md) |
| Current execution | F1 generation/replay passed; F2 component reconstruction passed |
| Scientific scope | Retrospective development; original data and predictions preserved |
| Implementation | `python -m analysis.main planned` |

## New evidence

- [reproduce.py.log](evaluation/reproduce.py.log)
- [verification.json](evaluation/verification.json)
- [verify.py.log](evaluation/verify.py.log)
- [Fresh fixed-recipe Y/J training and checkpoint-reload evidence](../../../reproduction/f1/validation_evidence/audit.json)
- [F2 component reconstruction](evaluation/f2_reconstruction.json)

F2's 50,000-row candidate reproduces its saved equal-weight components within half a six-decimal CSV unit. The minimum component-replay acceptance is passed. Independent validation is recorded in [tabpfn-replication](../../../review/checks/tabpfn-replication/summary.json), and subsequent full-data original/grouped fits in [tabpfn-ensemble](../../../review/checks/tabpfn-ensemble/audit.json). These source runs are distinguished from the local CPU audit, which performs no fresh checkpoint inference. Current machine dependencies/resources and historical screenshot association remain separate boundaries.
