[Chinese version](CHECKS.zh-CN.md)

## Documentation language migration (2026-11-07)

English is now the default for all 95 tracked documents that contained Chinese prose. Each has a Chinese companion; original Chinese bytes are preserved and listed in the [language-pair manifest](../docs/documentation-translations.json). Default navigation uses English counterparts and translated section anchors. See the [language policy](../docs/documentation-languages.md) for scope, historical evidence and generated-output boundaries.

Verification checks language coverage, original-snapshot hashes, code spans, Markdown fences and newly introduced link issues. Current source-integrity hashes are refreshed separately from frozen experiment provenance. This documentation change performs no model training or Kaggle submission.

# Branch execution verification

## Current verification after main refresh

Local records based on `4b91fad` and the new blend commit `c81f771` are in [main-refresh/summary.json](checks/main-refresh/summary.json). All 85 main tests, 32 standalone MLP tests, 38 isolated TabPFN tests and integrated `planned` pass. Optional dependencies reuse isolated locations; the TabPFN SDK is not merged into F1's environment.

New `analysis.main final-pair` reconstructs F2's 50,000-row candidate from two saved components with maximum difference about `5.0e-7` SGD, within six-decimal rounding tolerance, and completes a recent 28,995-row F1/F2 pairing. Annual F2, full ordered dictionary, complete information-policy equivalence and historical screenshot mapping remain unaccepted; main P0 is not fully complete.

Later integration retains both newly published original/grouped-category CSVs and provenance for 28 tree training/reload runs and full TabPFN inference, independently checking CSV, runner and audit hashes. Recent pairing still uses original-category F2; grouped 440.7707 is a separate development sensitivity result, not a new Kaggle score.

The [clean export check](checks/main-refresh/clean-export.json) for `157d7a7` passes: 4,297 files, 671 Python syntax checks, 85 main tests, integrated `planned`, four MLP submission verifications and independent Concat/TabPFN package checks. No local training cache was exported; third-party dependencies were reused through external paths, without claiming a new Conda environment. The [main-test log](checks/main-refresh/clean-main-tests.log) remains; temporary export/archive were removed afterward.

Below are historical records from different commits, machines and environments. Old missing-F2-generator and Python3.13 byte-reconstruction failures do not replace current local Python3.11 status. F1's fixed generation chain passes locally; no full research-model retraining or TabPFN checkpoint inference was freshly run this round.

This branch retains the original architecture, three environment files, requirements/ and all nine raw CSVs. Data and original submissions are SHA256-checked, without regeneration or reduction.

Initial split records are in [checks/summary.json](checks/summary.json) and [checks/export_check.json](checks/export_check.json): 52 of 53 selected tests pass, with one skipped for an omitted historical cache. CPU reconstruction exactly matches all 50,000 original predictions. This does not establish completion of all historical research experiments.

Latest pre-push scope, full tests and clean-export verification are in [checks/push-validation/](checks/push-validation/). Training uses local cs5228; extra dependencies come from local third-party locations, excluding original-workspace checkpoints and feature caches. Fresh Conda installation was not verified.

| Entry point | Purpose |
|---|---|
| `python review.py verify` | Verify branch file and data hashes |
| `python reproduction/f1/verify.py` | Verify frozen recipe, components and original submission |
| `python reproduction/f1/reproduce.py` | CPU reconstruction without training |
| `python reproduction/f1/train.py` | Retrain fixed recipe from raw data |
| `python reproduction/f1/check_trained.py` | Reload newly trained checkpoints and compare predictions |

Historical Kaggle scores apply only to original submissions; new training outputs are unscored. F2 generation remains pending and is not marked complete by split/execution checks.

TabPFN integration checks are recorded separately in [checks/tabpfn-split/](checks/tabpfn-split/). These do not replace the earlier F1 full-training evidence. F2 remains pending.

Latest main requirement-card execution and whole-tree verification: [checks/main-final/](checks/main-final/). New scientific results are in [experiment/main/](../experiment/main/README.md). This supplements the earlier F1 full-training and TabPFN-isolation logs.

## Acceptance of the five completion items

Current materials and entry points: [five improvements](../docs/planning/main-five-item-completion.md). Logs: [checks/five-item-validation/](checks/five-item-validation/).

| Scope | Observed result |
|---|---|
| Main tests | 73 pass, no skips |
| Original isolated TabPFN tests | 30 pass, no skips; no API calls or model downloads |
| New F1 Y/J generation | 15 base-model training/reload runs pass per window; 39 training sources verified against actual commits per window; new blend CSVs reconstruct byte-for-byte |
| Original F1 submission | Integrated `planned` executes verification and 50,000-row CPU reconstruction, preserving the original CSV |
| Concat | 28-file verification and 50,000-row CPU replay pass; maximum difference 0.105321 SGD is within original `0.10 + 1e-4 × abs(value)` tolerance |
| Integrated audit | Verification passes for 26 original predictions, 144 grouped-error rows, feature catalogs, scenario, spatial/cleaning sources and available row-level predictions |
| Whole-tree static verification | 625 registered files, 200 Python files and current navigation links pass |

Clean export of `8ebf6eb` also passes: 625 registered-file checks, 73 main tests, 30 isolated TabPFN tests, integrated `planned`, new F1 command help and Concat 50,000-row CPU replay. No local caches or existing training outputs were exported; third-party dependencies were reused in isolated environments. See [summary.json](checks/five-item-validation/summary.json) and `clean-*.log`.

Fresh training was limited to fixed F1 Y/J recipes, excluding full historical searches, complete eight-epoch Concat retraining or new independent holdouts. Large checkpoints remain in local ignored directories; compact predictions/provenance are saved with the branch. F2 generation evidence and full main P0 remain pending.

## Structure and intermediate-artifact cleanup

Rules and retention rationale: [layout guide](../docs/guides/repository_layout.md); itemized actions: [cleanup.json](checks/structure-cleanup/cleanup.json). Old exports/archives were checked against Git objects before deletion, retaining branch working directories, model outputs and installed dependencies. A byte-identical local backup preserves the old interim PDF.

All 78 main tests pass. Three default-EDA levels executed, with 18 tables matching frozen M01 and 11 main figures generated; raw CSVs remain unchanged. Integrated `planned`, F1 generation chain, Concat/TabPFN file checks and new help entry points pass. Training recipes, final results and frozen runtimes are unchanged; research models were not retrained. Records: [checks/structure-cleanup/](checks/structure-cleanup/).

Clean export of `c854542` again passes all 78 tests, 626-file static verification, integrated `planned` and default main EDA. It excludes local caches/training outputs. The temporary export was removed after verification; [committed-export.json](checks/structure-cleanup/committed-export.json) and logs remain.


## Latest anniversary branch merge

Current integration evidence is in [anniversary-merge/summary.json](checks/anniversary-merge/summary.json). The unchanged main package passes 78 tests; the isolated latest workflow passes 38 tests without skips. The original F1 50,000-row numerical replay passes. Source mapping and conflict choices are recorded separately from scientific model results.

The strict fresh-refit serialized CSV SHA check in `analysis.main planned` fails in this runtime. Separate Y/J numerical replay reproduces RMSE/MAE exactly and bias within 4e-15 SGD; this does not establish byte-identical reproduction. The checker and frozen scientific inputs are unchanged. The candidate F2 export/source is present, while its association with the historical displayed score remains unverified.
