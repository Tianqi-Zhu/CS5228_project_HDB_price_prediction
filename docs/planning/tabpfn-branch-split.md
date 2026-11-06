[Chinese version](tabpfn-branch-split.zh-CN.md)

# TabPFN branch-split decisions

| Provenance | Frozen commit | Action this round |
|---|---|---|
| `experiment/tabpfn` | `c0c987e26da7de953ecb6cd01539488976e6ddd1` | Split code, results, environments and derived data; directory decisions in `tabpfn-classification.csv` |
| `experiment/tabpfn-anniversary-history` | `1e69bb7995686d6a2bb7d502475e19ae72f081e3` | Check merge differences, reuse already separated zch content and retain integration evidence; not another model experiment |

## Minimal main experiment set

| Content | Location | Included scope | Acceptance boundary |
|---|---|---|---|
| 77 inputs, matched-origin histories and temporal baseline | `experiment/main/M07/tabpfn/tabpfn_v3_5_fast_aligned_temporal_80_20/` | Existing baseline/residual-representation controls | Summary/monthly metrics verifiable; row predictions and feature caches missing |
| Month-encoding control | `experiment/main/M07/tabpfn/tabpfn_v3_5_fast_calendar_80_20/` | Integer/categorical are matched controls; other candidates supplement the appendix | 455.863/455.814 are existing development results, not new tests |
| DiagScale components | `experiment/main/M07/tabpfn/tabpfn_components_80_20/` | Essential frozen/MSE-only comparison; retain complete candidates for provenance | Component gains are not evidence for final F2 |
| Required code and tests | `workflows/tabpfn/analysis/`, `workflows/tabpfn/tests/` | Dependency closure; dependencies are not automatically main experiments | Isolated from root `analysis/` and frozen F1 runtime |
| Final F2 | Existing `M07/M08/M09` plan | Still outstanding | User requested splitting current materials first; do not infer87 fields, members, weights or submission hashes |

Original tables retain all controls; the table above defines main selections. Detailed candidates belong to appendix categories. The future report's page limit does not justify deleting provenance code/tables.

## Appendix classification

| Category | Directory | Presentation value |
|---|---|---|
| Data, EDA, cleaning and deduplication | `T01`, `data/processed/` | Retain filtering, low-rent and deduplication alternatives without replacing raw training data |
| Temporal, market and economic alternatives | `T02` | Positive/negative macro, stock, dynamic-market and temporal-factor results |
| Spatial/external features | `T03` | Amenity, income, local-economy and external-data alternatives |
| TabPFN context and representations | `T04` | Context-size, sampling, history and calendar controls |
| Adaptation/residuals | `T05` | DiagScale, learning rates, LoRA, residual heads and API route |
| Ensembles/historical submissions | `T06` | ABCDE, ABDE, BDE controls and old submission recipes/selection rules |
| Additional complex models | `T07` | TabM, scratch Transformers, architecture and loss exploration |
| Other baselines/integration records | `T08` | Original environments, merge differences and early-baseline provenance |

Appendix preserves81 original result directories, including three shared main controls not counted as new experiments. Future reports should select figures by scientific question and use appendices sparingly; complete repository archives need not all appear in the paper.

## Comparability and reproduction gaps

| Issue | Action | Assessment implication |
|---|---|---|
| Only screenshot/user provenance for F2 filename | Preserve413.19635 as screenshot information; generator, 87-input schema, actual blend members, submission CSV and row predictions remain outstanding | Methods60%:final method/comparison not accepted; reproducibility10%:generation chain incomplete |
| ABCDE/ABDE have different identities from F2 | Original A/B/C use 77 fields, D84, E89; do not rename old equal ABCDE as F2 | Prevent invented methods and misattribution |
| Different validation windows/counts | Main TabPFN controls use 28,995 June2024–March2025 rows, separate from F1's35,158 annual rows; pairing still needs IDs/predictions | Methods: do not rank minimum RMSEs across windows |
| Repeated inspection of one development set | Retain selection rules, failed gates and negative results; not independent tests | Methods/report quality: explain selection bias and applicability |
| Final CSVs, members and several feature/model caches absent from Git | Explicitly mark missing items; verify available source files/aggregate tables only | Reproducibility: old `verified=true` is not this round's rerun |
| Historical settings/current snapshot hashes may differ | `review_tabpfn.py preflight` distinguishes LF/CRLF-only differences from other revisions; preserve historical records | Reproducibility: match original generator version or record a complete fresh run |
| Original environment lacks SDK; specified checkpoint missing | Preserve environments; run original tests with isolated dependencies; no checkpoint download or paid API call | Only executed work can pass; full inference still awaits resources |
| Same-name `analysis` conflicts | Assemble original architecture under `output/`,then launch a separate process | Keep verified F1 runnable without silently changing preprocessing/models |

## Alignment with assessment

| Assessment category | Delivered this round | Incomplete work |
|---|---|---|
| Method quality60% | Original EDA, controls, alternatives, negative results, selection gates and window distinctions | Actual F2 generator and same-row F1/F2 pairing |
| Report quality30% | Main/appendix classification and traceable index; no manuscript drafting this round | Future PDF: 8-page body, self-containment, citations and team details |
| Reproducibility10% | Isolated architecture, environments, raw/derived data, hash inventories, test/metric commands | Missing weights, caches, final predictions; full TabPFN retraining/fresh installation |

The course provides no itemized deduction formula or separate novelty weight; this plan adds neither.
