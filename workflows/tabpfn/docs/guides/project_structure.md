[Chinese version](project_structure.zh-CN.md)

# Project structure: conclusions, implementation and run details

## Three top-level activities

- Read research: start at `README.md`, then `experiment/`.
- Inspect or run code: use `analysis/` and consult `docs/` for commands/dependencies.
- Inspect run details/models: use `output/`; original inputs remain in `data/`.

```text
README.md
environment.yml                  Core Conda environment: 17 packages, pinned training versions and ranged tools
experiment/                         Findings for reading and citation
  README.md                      Reading order
  01_final/                      Final methods, full report and three-submission comparison
  02_feature_engineering/         Feature definitions, dictionaries, ablations and ladder
  03_model_comparison/            Shared-input models, recommendations and key figures
  04_optimization_history/        Ten waves, one summary Markdown per wave
  eda/                           Data quality, tables and descriptive figures
  studies/                       Economic, spatial and panel studies
analysis/                        Implementation and entry points
  main.py                        Unified CLI
  experiment_catalog.py          Read-only experiment index
  resource_preflight.py          Task-specific environment, file and recorded-hash checks
  artifact_catalog.py            Current model/parameter catalog without model loading
  report_bundle.py               Refresh selected reading reports
  pipeline/                      Shared data, features, models, evaluation and paths
  experiments/
    baselines/                   Baselines and early experiments
    feature_engineering/         Feature selection, representations and ablations
    model_comparison/            Search, shared inputs and preparation
    reporting/                   Model selection, reporting and branch inventory
    submissions/                 Local export and final selection
    optimization/
      round_01_history_target_search/
      round_02_anniversary_history/
      round_03_tree_transformer_ensemble/
      round_04_macro_spatial_diagnostics/
      round_05_feature_interactions/
      round_06_deep_models/
      round_07_pretrained_transformers/
      round_08_feature_pruning_seeds/
      round_09_grouped_blending/
      round_10_plr_transformer_final/
  eda/                           Data-analysis implementation
  studies/                       Interpretive-study implementation
  project_review.py              Baseline-27 report generator
docs/                            Usage and organization rationale
  README.md
  guides/
    running_experiments.md        Environment, commands, prerequisites and training boundaries
    project_structure.md         This document
  methods/
    baseline_features.md         Baseline-27 definitions
    history_feature_optimization.md History representations and target decomposition
    model_comparison.md          Model-search settings and protocol
  archive/                       Midterm, integration and old workflows
  maintenance/                   Cleanup rules and change summaries
data/                            Raw competition inputs, unchanged
  README.md                      Nine CSVs, target fields and information boundaries
requirements/                    Optional extensions to the same main environment
  README.md                      Installation entries by task
  optional_deep_models.txt       TabM / RealMLP
  optional_pretrained_transformers.txt TabDPT / Nori
tests/                           Automated checks
  README.md                      Test categories, commands and dependency boundaries
output/                          Runtime artifacts, ignored by Git
  experiment_runs/               All trial configurations, metrics, selection records and original reports
  models/                        Models, blend weights and aligned predictions
    current_catalog.json         Current catalog linking purpose and metric sources
    index.json                   Retained earlier index snapshot
  submissions/
    final/submission.csv         Exact copy of current final
    candidates/                  Three selected and other candidate CSVs
    components/
      history_macro_ensemble/    Full history/macro components
      round_08_ft_seed_average/   Wave-eight seed averaging
      round_09_grouped_blend/     Wave-nine blend weights
      round_10_final_structure/   Wave-ten final structure
    baselines/
      baseline_27_features/      Early 27-column reference
  checks/
    runs/                        Actual execution checks and logs
    records/                     Dated machine-check JSON and source manifests
    reviews/                     Review findings, patches and execution status
  archive/                       Earlier archived run copies
.cache/                          Feature caches, isolated dependencies, author code and public weights
.integration-work/               Other independent worktrees and temporary checks
```

## Naming rules

`baseline_*` denotes references, `features_*` feature experiments and `comparison_*` model comparisons;
`round_01_*` through `round_10_*` follow performance exploration chronologically.
Reading directories follow the narrative; run/model directories retain wave names for traceability.
Problematic trials remain in `archive_weight_scale_debug`, excluded from official rankings.

Python scripts retain original module commands, such as `analysis.experiments.tenth_wave_structures`.
Physical source locations are grouped; index JSON `source` locates each file.
Module names needed by saved models still resolve, without retraining or prediction changes due to renaming.

## Reading files versus runtime files

`experiment/` holds selected prose, summaries and key figures; `output/experiment_runs/` retains full traceable details,
including failed trials rather than deleting them for a simpler view.
`python -m analysis.main experiment-refresh` updates 25 reading files from existing local results without training;
handwritten section navigation is not overwritten.

The [maintenance directory](../maintenance/README.md) indexes provenance and migration checks; machine records are in `output/checks/records/`.
Old maintenance paths describe earlier layouts; current paths follow this migration map.
Dependencies and weights in `.cache/` require preparation again if deleted; the whole directory is not disposable.
