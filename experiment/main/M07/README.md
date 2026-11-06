# M07 F2 identity and available TabPFN background

| Item | Status |
|---|---|
| Requirement | See [requirement cards](../../../docs/planning/main-appendix-experiment-plan.md) |
| Current execution | partial; candidate source/export integrated, screenshot identity unverified |
| Scientific scope | Retrospective development; original data and predictions preserved |
| Implementation | `python -m analysis.main final-pair` or `planned` |

## New evidence

- [f2_manifest.json](evaluation/f2_manifest.json)
- [source_manifest.json](evaluation/source_manifest.json)

## Existing evidence

- [tabpfn](tabpfn/)

The integrated candidate is an equal blend of the earlier seven-tree ensemble and frozen TabPFN87, not F1 plus TabPFN. The audit checks component hashes (including proven LF/CRLF conversion), the candidate formula and all 50,000 IDs. The ten added anniversary columns and 87-input count are recorded; the full exported ordered dictionary remains missing. The candidate's association with the historical 413.19635 screenshot is unverified. Standalone TabPFN validation replay is separately documented in [its run guide](../../../docs/guides/tabpfn_best.md); it does not verify the blended candidate's Kaggle score.

The subsequent [ensemble run](../../../docs/guides/tabpfn_ensemble.md) records 28 tree fits/reloads and original/grouped full TabPFN inference. Both new 50,000-row CSVs are shipped. `final-pair` additionally verifies their IDs, values, hashes and the audit/training-runner identity; the fresh original agrees with archived F2 to six-decimal precision. The grouped variant's 440.7707 RMSE is a source-reported sensitivity result, not a replacement for the original-category paired predictions. Neither new CSV has a verified Kaggle score.
