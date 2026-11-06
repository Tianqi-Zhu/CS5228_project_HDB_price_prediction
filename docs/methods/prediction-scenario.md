[Chinese version](prediction-scenario.zh-CN.md)

# Current prediction scenario

Main experiments use **the supplied static amenity snapshot and a fixed historical origin**. See machine-readable [prediction-scenario.json](prediction-scenario.json). This unifies interpretation across main tables without changing raw training data, features or existing scores.

| Item | Adopted boundary | Verification evidence |
|---|---|---|
| Task | Individual HDB monthly rent in SGD | Original `MONTHLY_RENT`; retain every validation record |
| Y | Fit months <2024-04; query 2024-04 through 2025-03 | 114,842 / 35,158 rows |
| J | Fit months <2024-06; query 2024-06 through 2025-03 | 121,005 / 28,995 rows; overlaps Y |
| Rental history | Fitting rows use earlier months only; query histories freeze at the origin | Exclude query labels from model fitting; use fixed training lengths and weights |
| Learned transforms | Learn encoding, imputation and scaling only on fitting data | Frozen F1 runtime and new Y/J generation records |
| Amenities | Use supplied coordinate and STATUS snapshots | Monthly opening dates are unverified; original `OPEN_MRT` denotes the source snapshot rule only |
| Macro inputs | F1 does not use realized query-month macro observations | Component input inventories; other branches' macro information policies do not apply to F1 |
| Full-data submission | Fit all 150,000 labels and predict 50,000 rows | Of these, 380 test records share 2025-03 with the latest fitting labels. The supervised model sees labels from that month, while history aggregation still excludes same-month labels. |
| Model selection | Y/J are previously examined development sets | Overlapping, repeatedly inspected windows are not independent tests |
| Cleaning control | Separately fits 105,837 rows through 2023-12 and calibrates in Q1 2024 | Compare within its matched groups, not against F1 in a combined ranking |
| F2 | Equal blend of seven trees and TabPFN87; same-row comparison on recent J, with complete input-policy equivalence not yet accepted | [F2 manifest](../../experiment/main/M07/evaluation/f2_manifest.json) and [paired protocol](../../experiment/main/M08/evaluation/protocol_checks.json); same-row numerical comparison does not isolate one architecture's contribution |

Strict historical facility availability belongs to appendix X08 and requires reliable opening dates, matched changes to training/inference features and retraining. Current results support the competition-snapshot setting, not strict historical deployment performance. Adjusted spatial EDA associations are not causal facility effects.
