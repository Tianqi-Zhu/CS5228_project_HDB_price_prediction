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
