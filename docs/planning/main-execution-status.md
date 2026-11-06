[Chinese version](main-execution-status.zh-CN.md)

# Main requirement-card execution status

Initial main offline audits/diagnostics added zero fits. This round completes [five items](main-five-item-completion.md),including 15 fixed-recipe F1 base models per Y/J window, saved and reloaded without reselecting models, weights or hyperparameters. `planned` still audits without training. Acceptance follows actual inputs, controls and results; unmet requirements remain registered.

| Card | Work this round | Current status | Evidence |
|---|---|---|---|
| M00 | Audit26 old prediction groups, F1 fixed generation, F2 recent IDs/labels and candidate-member identity | F1 complete; F2 recent numerical pairing executable; old generator commits not fully recovered | [Protocol registry](../../experiment/main/M00/evaluation/protocol_registry.csv), [new F1 generation](../../reproduction/f1/validation_evidence/audit.json) |
| M01 | Raw counts/data policies; add spatial associations, controlled bins, matched cleaning and explicit scenario | Supplied-snapshot scenario complete; original regressions not independently reestimated; historical facilities remain X08 | [Data policy](../../experiment/main/M01/evaluation/data_policy_audit.json), [new M01 evidence](../../experiment/main/M01/README.md), [prediction scenario](../methods/prediction-scenario.md) |
| M02 | Reuse verified/retrained C7 classical-model table | Existing experiment usable; no duplicate training this round | [C7 controls](../../experiment/main/M02/README.md) |
| M03 | Recompute16 ladder predictions; register ordered schemas/component inputs and correct PLE/PLR metadata omissions | Complete; two62-input schemas named separately | [Increment table](../../experiment/main/M03/evaluation/feature_ladder_deltas.csv), [feature inventory](../../experiment/main/M03/feature_schemas/README.md) |
| M04 | F1 and 62-input LightGBM/CatBoost by month, type, 24-/36-month support and fitting q10/q90 rent bands; 144 group records and identical-input diagnostics | Matched diagnostics complete on both windows | [Error groups](../../experiment/main/M04/evaluation/error_groups.csv), [support audit](../../experiment/main/M04/evaluation/feature_support_audit.json), [identical inputs](../../experiment/main/M04/evaluation/identical_input_diagnostics.csv) |
| M05 | Pair PLE/PLR standalone and fixed-rule blends with no addition/PLE/PLR; retain budget differences/negative results | Recomputable evidence complete; common27-input block retained separately | [Structural increments](../../experiment/main/M05/evaluation/controlled_structure_deltas.csv), [selection audit](../../experiment/main/M05/evaluation/selection_audit.json) |
| M06 | Reuse identities and selection rationale for three historical submissions | Existing experiment usable; history/macro alternatives retained in appendix | [F1 selection](../../experiment/main/M06/f1_selection.json) |
| M07 | Verify seven-tree/TabPFN87 composition, fixed equal weights, source hashes and 50,000-row candidate | Candidate reconstruction passed; complete ordered dictionary and screenshot association outstanding | [F2 manifest](../../experiment/main/M07/evaluation/f2_manifest.json) |
| M08 | Recent F1/F2 numerical comparison on 28,995 matched rows with month/type/support groups | J complete; Y and full information-policy equivalence audit outstanding | [Protocol checks](../../experiment/main/M08/evaluation/protocol_checks.json), [paired metrics](../../experiment/main/M08/evaluation/final_pair_metrics.csv) |
| M09 | F1 verification/CPU reconstruction/fixed generation; F2 candidate-member reconstruction | F1 passes stated scope; F2 reconstruction passes without being presented as fresh GPU inference | [Acceptance](../../experiment/main/M09/evaluation/verification.json), [F2 reconstruction](../../experiment/main/M09/evaluation/f2_reconstruction.json) |

## Interpreting this round's results

- Y: 2024-04 through 2025-03,35,158 rows; J: 2024-06 through 2025-03,28,995 rows. They overlap and cannot be pooled as independent tests.
- Adding PLR versus the same-rule blend with no new network changes Y RMSE by **+0.164455** SGD and J by **−0.124548**. Standalone, blend and window comparisons can differ.
- F1's834 Y rows below fitting q10 contribute about **21.03%** of SSE; 640 corresponding J rows contribute **19.40%**. These groups use true validation rents for post-hoc diagnosis only, not features or deletion rules.
- Identical-input groups refer only to actual ordered inputs of the two62-column controls. Within-group variation is local descriptive evidence, not a global irreducible-error floor or F1's complete heterogeneous input set.
- Original cached predictions/summaries have hashes; branch commits supply provenance context, not necessarily each cache's exact generating commit.

## Execution and reproduction

```powershell
python -m analysis.main planned
python -m analysis.main final-pair
python -m analysis.validation.repository_audit
```

`planned` uses a [5.8 MB frozen prediction bundle](../../experiment/main/M00/inputs/validation.npz),without old weights/workspace caches. Rebuilding support uses pre-origin training rows and removes query rents before feature generation. Outputs go to `output/planning_validation/main/<content-digest>/` without overwriting runs. Use `--output-dir output/planning_validation/check-2` for another run.

Card results are archived in each `evaluation/`; the [summary](../../experiment/main/evaluation-summary.json) records execution and [M04](../../experiment/main/M04/README.md) holds charts. Input [metadata.json](../../experiment/main/M00/inputs/metadata.json) records prediction/configuration hashes, schemas, learner settings and old selection records. `python -m analysis.validation.prepare_inputs --destination output/planning_validation/new-inputs` rebuilds source caches and requires listed local artifacts; ordinary recomputation does not.

## Evidence still required

1. Complete ordered87-field F2 export dictionary and exact association between the candidate and historical413.19635 screenshot. Members, equal-weight formula, candidate and inference settings are now inspectable.
2. Pre-Y fitted F2 predictions and full input/information-policy equivalence audit. Recent J numerical pairing is complete. Do not backtest150,000-row models on records used in fitting.
3. Original generating code trees remain incomplete for old caches; new F1 Y/J predictions were regenerated from a fixed commit and verified. Strict historical-amenity claims still require X08 dates/matched refits.

Therefore **not all main P0 requirements are complete**. Audits, diagnostics, F1 generation checks, F2 reconstruction and recent pairing are executable. Remote [standalone TabPFN reproduction](../guides/tabpfn_best.md) and [MLP reproduction](../../reproduction/mlp/README.md) retain their own environments/scope; neither replaces missing F2 annual/screenshot evidence.

## Whole-main portability check

Concat training used POSIX-only `fcntl`; the runner now provides Windows/POSIX exclusive locking and a data-copy fallback when symlinks require privileges. This changes runtime plumbing and new checkpoint fingerprints, not the architecture, frozen weights, original CSV or selection settings. All frozen artifacts were byte-checked against the prior integration. Raw feature preparation completed on 150,000/50,000 rows; a separate 32-train/16-query, two-epoch CPU smoke fit saved and reloaded successfully. No new research model selection or full eight-epoch refit was performed. Evidence: `review/checks/main-final/concat-small-training-smoke.json` and `reproduction/concat_transformer/portability.json`.
