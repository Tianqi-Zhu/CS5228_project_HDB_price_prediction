[Chinese version](README.zh-CN.md)

# Ordered feature catalog

| Table | Purpose |
|---|---|
| [Schema registry](registry.csv) | Feature-set ID, column count, ordered-name hash, source and statistical level |
| [Names, recorded types and sources](schemas.json) | Unified 7/19/27/47/62, adjusted62, heterogeneous F1 members and Concat inputs; dtypes only where sources record them |
| [Transitions](transitions.csv) | Added/removed columns and whether sets are truly nested |
| [F1 component inputs](F1_member_inputs.csv) | Actual column counts, rounds, checkpoint hashes and generating commits for 30 base models across two windows |

| Feature set | Definition and comparison scope |
|---|---|
| C7 | Initial traditional models; integer month encoding, not a strict subset of static19 |
| ladder19 | Property, space/time and facility snapshot; month encoded with sin/cos |
| ladder27 | 19 + eight historical statistics; also the raw shared-input comparison set, encoded separately by each model |
| ladder47 | 27 + 20 market-relative historical statistics |
| ladder62 | 47 + five statistics each for 12/24/36-month anniversary windows |
| adjusted62 | ladder62 minus five 36-month statistics plus five COHORT24/market-adjusted features; same count, different inputs |
| F1 | 12 logical predictions and 15 base models; heterogeneous 22/62/67/87 inputs cannot all be called 62-column models |
| Concat | Same raw columns as adjusted62; encoded into 95 numeric inputs including missingness flags and four categorical inputs, recorded separately from the 62 raw columns |

**Metadata correction:** the old `M00/inputs/metadata.json` listed 57-column PLE/PLR schemas, omitting `adjusted=True` enforced by `sixth_wave_deep.prepare`. Training records actually use 62 columns; newly saved PLR models also verify all 62 ordered columns. Original predictions and metadata remain intact. This catalog lists four corrections, and new `planned` registrations use corrected inputs. The cataloging script is fixed without rewriting original metrics or predictions.

Run `python -m analysis.main schemas` to verify and write fresh `output/feature_catalog/` outputs. Equal column counts do not establish matched experiments when target reconstruction, fitting cutoffs or budgets differ. F2 has an 87-column count, ten added anniversary fields and reproduction records, but still lacks a complete ordered dictionary. F1 broad87 does not establish F2 identity; see [M07](../../M07/README.md).
