[Chinese version](README.zh-CN.md)

# Main experiment index

[Execution status and gaps](../../docs/planning/main-execution-status.md) · [Current results](evaluation-summary.json) · [Requirement cards](../../docs/planning/main-appendix-experiment-plan.md)

| ID | Current status | Content |
|---|---|---|
| [M00](M00/README.md) | F1 generation and F2 recent identities verified | Audit of 26 archived predictions; fixed F1 generation chain; F2 row IDs/labels and candidate-component provenance |
| [M01](M01/README.md) | completed for snapshot | Shared EDA, controlled spatial associations and matched cleaning comparisons; explicit snapshot scenario |
| [M02](M02/README.md) | archived/verified | C7 mean, Ridge, RandomForest, LightGBM and CatBoost baselines |
| [M03](M03/README.md) | completed | Feature ladder and paired increments; ordered schemas, heterogeneous components and two different 62-column sets |
| [M04](M04/README.md) | completed | Monthly, flat-type, coverage, tail and same-input diagnostics for F1 and two 62-column controls |
| [M05](M05/README.md) | completed within recorded scope | 27-column architecture comparison; 62-column PLE/PLR and fixed-rule blend increments; separate Concat exploration |
| [M06](M06/README.md) | archived/verified | F1 selected from three existing submissions; other versions retained in the appendix |
| [M07](M07/README.md) | candidate reconstructed; partial identity | Seven-tree + TabPFN87 equal-weight formula and CSV verifiable; complete ordered dictionary and screenshot association pending |
| [M08](M08/README.md) | J complete; Y pending | Recent 28,995-row paired comparison; annual predictions and full information-policy equivalence audit pending |
| [M09](M09/README.md) | component replay passed | F1 verification/CPU reconstruction and generation chain pass; F2 component reconstruction passes, without a fresh full GPU inference run this round |

Each card's `evaluation/` holds new analyses; original directories retain earlier experiments. Interpret paired gains only within matched data, configurations and protocols; Y/J are not independent tests. Main P0 is not fully complete, with gaps listed by card.

Full cleaned/uncleaned MLP submissions, time weighting and interval workflows are linked through [M05's standalone entry](M05/README.md), not treated as a third final endpoint. Run `python -m analysis.main final-pair` to inspect the recent F1/F2 pairing and F2 component identity separately.
