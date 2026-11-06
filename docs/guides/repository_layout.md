[Chinese version](repository_layout.zh-CN.md)

# Main directory and artifact-retention rules

| Directory | Role | Retention rule |
|---|---|---|
| `analysis/pipeline/` | Current shared data, feature and learner implementations | Retain; run through the corresponding experiment entry points |
| `analysis/experiments/` | Main implementations and dependencies grouped by purpose and round | Legacy module names support imports and checkpoint compatibility; do not delete based on whether they appear directly in the paper |
| `analysis/validation/` | Main protocol, feature, provenance and reproduction verification | Retain; primary entry point is `python -m analysis.main planned` |
| `analysis/eda/`, `analysis/studies/` | EDA and supplementary research implementations | New outputs default to `output/eda/` and `output/studies/`, respectively |
| `data/` | Nine original data files | Preserve original bytes; do not move, reduce or overwrite with cleaned data |
| `experiment/main/M00`—`M09` | Frozen experiment evidence, main results and supporting tables/figures | Preserve provenance and verifiable artifacts; this is not a temporary directory for every run |
| `reproduction/f1/` | Fixed F1 recipe, original submissions, component predictions and new Y/J training evidence | Retain; the independent frozen runtime reproduces the original model |
| `reproduction/concat_transformer/` | Integrated Concat architecture, weights, preprocessing and replay inputs | Retain; large `bank.npz` supports offline replay and is not a disposable cache |
| `reproduction/mlp/` | Independent CPU MLP runtime, four full-data submissions, cleaning/temporal-weighting results and original provenance | Retain; `artifacts/fullfit/source/` snapshots the old submission generator and differs from current `runtime/`. Do not delete matching names as duplicates. |
| `workflows/tabpfn/` | Isolated TabPFN workflow and dependency instructions | Retain; generated working directories belong under `output/` |
| `docs/` | Running guides, methods, plans and current status | Navigate from the root README; old plans in `docs/reviews/` are redirects only |
| `requirements/` and root environment files | Core and optional-model environments | Retain regardless of what is installed locally |
| `review/` | File/source manifests and acceptance records for each stage | Retain; historical logs describe different commits/stages and are not current results |
| `output/` | New models, predictions, feature caches, charts, temporary exports and local backups | Git-ignored; clean specific paths after confirming provenance and purpose |
| `.cache/` | Local dependencies, author code, fixed weights and some rebuildable caches | Assess individually; dependencies, fixed models and source snapshots are not ordinary temporary files |
| `.integration-work/` | Working directories for other branches | Independent of current main; contains appendix and other branches and must not be deleted wholesale |

## Current entry points

```powershell
python -m analysis.main --help
python -m analysis.main planned
python -m analysis.main final-pair
python -m analysis.main eda
python -m analysis.validation.repository_audit
```

EDA defaults to `output/eda/` consistently at every layer. `python -m analysis.eda.pdf_report` generates the earlier midterm slides from frozen M01 EDA at `output/reports/progress_report.pdf`; add `--eda-dir output/eda` to use new analyses. The midterm PDF is not the final scientific report.

`python -m analysis.report_bundle` is a legacy reading-bundle helper that requires existing historical experiment outputs and defaults to `output/reading_bundle/`. It is not a main acceptance command and no longer copies intermediates into obsolete locations such as `experiment/01_final`. Legacy `project_review` and `artifact_catalog` tools still support imports and local historical artifacts; do not remove them solely because they are absent from the entry-point list.

## Cleanup completed in this round

| Content | Action | Recovery |
|---|---|---|
| Old M01 `progress_report.pdf` | Removed from Git main; a local SHA256-identical backup is retained, approximately 1 MB | Git history, local `output/archive/main-structure-cleanup/`, or regeneration with the retained script |
| Eight old export directories and six corresponding ZIPs | Verified all tracked files against their commits before removal; also checked export directories for extra user files | Run `git archive` at the recorded commit, then rebuild temporary outputs using the running guide |
| Pip download cache from branch splitting | Removed; installed dependency directories were preserved | Dependency installation can recreate the download cache |
| Nonexistent commands and incorrect EDA output descriptions in old test docs | Updated to current main files and commands | Follow this document, test README and root README |

Removed 15 temporary targets locally, approximately **4.90 GiB**. See the [cleanup record](../../review/checks/structure-cleanup/cleanup.json) for identities, sizes and actions. Removing PDFs does not automatically shrink Git history; history was not rewritten.

## Retained as useful artifacts

Raw data, final F1/Concat results and checkpoints, new F1 Y/J models, blending predictions, incomplete F2 materials, six untracked review drafts, other branch worktrees and local dependencies were all preserved. Duplicate-input details support checks of genuine sample conflicts; source-branch tables/configurations support provenance. Content outside the main paper can still be useful.

See the [main execution table](../planning/main-execution-status.md) for research status. Cleanup does not resolve F2 evidence gaps.
