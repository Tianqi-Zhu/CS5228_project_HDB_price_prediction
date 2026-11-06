[Chinese version](README.zh-CN.md)

# Local maintenance and directory changes

This holds cleanup rules and change summaries only. Machine-check JSON goes to `output/checks/records/`, separate from usage docs.

## Six follow-up improvements

Cache guidance, review navigation, data/testing guides, read-only preflight and current model catalog were completed sequentially.
Each step ran all tests, seven existing-result checks and CLI checks; tests increased from 122 to 133.
All six preflight profiles pass; final SHA-256 values for 1,698 existing data/model/submission files are unchanged.
Scope and step records: [six-check entry](../../output/checks/runs/directory_polish_20261105/README.md).

## Current output layout

Runtime artifacts are grouped into submissions, models, detailed experiments, checks and backups.
Submission categories are `final/`, `candidates/`, `components/` and `baselines/`;
baseline-pipeline defaults also use these categories.
Ten directory groups and 184 existing files were moved, with no artifacts deleted.
All121 existing tests and one default-output-path test pass; the final 50,000-row blend differs from the original CSV by at most about 9.1×10⁻¹³.
See [output navigation](../../output/README.md) and [check records](../../output/checks/records/2026-11-05/output_layout.json).

## Current environment and end-to-end audit

Core dependencies are consolidated in root `environment.yml` with two optional extensions; the installed environment was not changed.
Local main workflow, 15 final-model reloads/blending and 121 tests pass.
Scope, path fixes and fresh-environment limitations: [environment/run audit](../../output/checks/runs/environment_merge_20261105/AUDIT.md).

## Completed cleanup and checks

The27 user-approved temporary targets were deleted: 4,192 files, about 150.77MiB.
Rules and targets: [cleanup criteria](#cleanup-criteria); results: [execution check](../../output/checks/records/2026-11-05/cleanup_result_2026-11-05.json).
All119 tests pass; final PLR-FT reproduces all 50,000 predictions with maximum difference0.
Earlier documentation/dependency restructuring: [project integrity check](../../output/checks/records/2026-11-05/project_integrity.json):
421 valid local links; seven dependency entries parse; version constraints unchanged.

User-reviewed pruning is complete, reducing `docs/` from 15 files to 11; dependency files unchanged.
Latest link/run checks: [documentation pruning](../../output/checks/records/2026-11-05/docs_pruning_execution.json).

## 2026-11-05: reorganization by reading and execution purpose

- Reading reports cover final conclusions, feature engineering, model comparisons and optimization history; EDA and interpretive studies stay separate.
- Detailed experiment records moved intact to `output/experiment_runs/`, retaining configurations and negative results.
- Model, candidate-submission and wave directories use meaningful names.
-86 experiment modules moved into six code categories; original commands and saved-model import names remain valid.
- Baseline-stage prose moved to run records for extraction by the reading updater; machine maintenance records moved to `output/checks/records/`.
- Raw data, prediction CSVs and model binaries were hash-checked; path changes do not alter model parameters.

Current sources: [reading-file manifest](../../output/checks/records/reading_bundle_sources.json);
migration and actual checks: [directory_layout_2026-11-05.json](../../output/checks/records/2026-11-05/directory_layout_2026-11-05.json).
Old paths/hashes belong to their historical stages and are not presented as fresh rerun evidence.

Full pre-edit copies, file mappings and hashes remain locally in
`.cache/maintenance/directory-rename-20261105/`; no commit or push was made.

## Earlier records

- [2026-11-05 navigation cleanup](../../output/checks/records/2026-11-05/local_navigation_2026-11-05.json)
- [2026-10-25 artifact cleanup](../../output/checks/records/2026-10-25/cleanup_2026-10-25.json)

The original model index is local at `output/models/index.json`, if present.

## Cleanup criteria

A deletion target must meet all conditions:

1. Current execution, model reloads, blending, submission export and report updates do not depend on it.
2. It is not the sole record of results, negative results, parameters/seeds, data versions or provenance checks.
3. An identical official copy exists, or regeneration is inexpensive and automatic.
4. Deletion does not alter raw data, predictions or models, impair retained experiments' reproduction, or touch other Git worktrees.

Retrainability does not imply cost-free deletion. Training cost, randomness and weight availability matter for retention.
No textual references alone does not prove uselessness; check dynamic paths, recipes, serialized models and generation workflows too.
