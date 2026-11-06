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

F2's 50,000-row candidate reproduces its saved equal-weight components within half a six-decimal CSV unit. This is component reconstruction, not fresh checkpoint inference or full refitting. The independent standalone TabPFN validation replay is recorded in [tabpfn-replication](../../../review/checks/tabpfn-replication/summary.json); the current machine's SDK/checkpoint availability must be checked separately. The historical screenshot association remains unverified.
