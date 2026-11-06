[Chinese version](running_experiments.zh-CN.md)

# Local experiment workflow

Applies to the full-project branch `zch-complete-proj` (formerly `feature/midterm-eda-models`); run from the repository root.
Start with [report navigation](../../experiment/README.md), [optimization history](../../experiment/04_optimization_history/README.md) for waves, and [experiment entries](../../analysis/experiments/README.md) for code.

## Choose the task before running commands

| Task | Entry point | What happens |
| --- | --- | --- |
| Read final conclusions | Open [FINAL_MODEL](../../experiment/01_final/README.md) | Read-only; no model dependencies |
| Find scripts | `analysis.main experiment-list` | AST lists entries without importing experiments |
| Refresh reading reports | `analysis.main experiment-refresh` | Update selected prose/tables from existing results without training |
| Compare same-input models | `analysis.main common-models` | Trains; requires earlier search records |
| Explain feature increments | `analysis.main feature-evidence` | Remove columns/groups and retrain |
| Retrain the 27-column reference pipeline | `analysis.main canonical-submit` | Generate early LightGBM predictions, not current final |
| Rebuild baseline-stage report | `analysis.main project-review` | Read existing results and rewrite stage report/feature dictionary |
| Run later performance exploration | `analysis.experiments.<wave_module>` | Train or fit blends according to arguments |
| Export history/macro CSVs | `analysis.experiments.best_ensemble_submission` | Reuse verified components, training if missing |
| Wave-ten checks and export | `analysis.experiments.tenth_wave_report` | Reload, recompute and write reports; may fully train/export |
| Archive final selection | `analysis.experiments.final_submission_selection` | Verify CSVs, copy final and rewrite selection records/conclusions |

Submission commands write local files only, without Kaggle uploads. Reading results does not require generators.

Code is grouped by purpose/wave; original module commands remain, with actual locations in index JSON `source`.
Training and stage generators write complete records to `output/experiment_runs/`; then `experiment-refresh` updates reading copies.

## Environment and index

Start with read-only resource preflight:

```powershell
python -m analysis.main preflight --profile core
python -m analysis.main preflight --profile reports
python -m analysis.main preflight --profile final
python -m analysis.main preflight --profile wave06
python -m analysis.main preflight --profile wave07
```

`core` checks nine competition CSVs, target boundaries and core versions; `reports` also checks selected reports' original sources;
`final` additionally checks 15 saved weights, feature bundles, 12 prediction streams, order and recorded input/prediction hashes;
`wave06`/`wave07` check optional packages, author sources or public weights. `all` combines profiles; `--json` gives full details.
Missing/mismatched resources produce exit code1 and repair guidance, without downloading, installing, deserializing models or training.

Preflight uses the standard library; standalone command: `python -m analysis.resource_preflight --profile final --json`.
Versions come from current-environment and existing isolated-directory metadata, showing each provider path; specialized models still need their author loaders to activate paths.
Passing preflight establishes static resources, not all transitive dependencies, CUDA inference or fresh installation. Use `verify` and tests for runtime checks.

```powershell
conda env create -f environment.yml
conda activate cs5228
python -m analysis.main --help
python -m analysis.main experiment-list --group wave10
python -m analysis.main experiment-list --group final --json
```

Index-only inspection does not require Conda modeling dependencies:

```powershell
python -m analysis.experiment_catalog --group wave06
```

Groups include `baseline`, `features`, `models`, `report`, `frontier`, `wave02` through `wave10`, `final` and `other`.
Modules without `__main__` are support code, not directly runnable scripts.
The index shows command prefixes; inspect required arguments first. Do not blindly append `--help` to old scripts without argument parsers:
they may ignore arguments and start work.

GPU experiments and layered dependencies: [requirements](../../requirements/README.md). The baseline Conda file does not lock all exploratory dependencies.
Wave six also needs `requirements/optional_deep_models.txt` and author sources; wave seven needs
`requirements/optional_pretrained_transformers.txt` and public weights. Downloaders/version records are in wave READMEs.
`.cache/` is not distributed through Git; missing dependency/weight subdirectories require preparation.

## Baseline features and model comparisons

Inspect saved models, blend parameters and metric sources without training:

```powershell
python -m analysis.main artifact-list
python -m analysis.main artifact-list --usage final
python -m analysis.main artifact-list --write --hashes
```

The last command hashes files and writes `output/models/current_catalog.json`; many files can take time.
`--json` shows per-file entries; `--usage` accepts `final`, `development`, `submission_component` or `discarded`.
Filtering cannot combine with `--write`, avoiding partial catalogs saved as complete indexes.
The catalog preserves original metric field paths and window context instead of automatically ranking RMSE across periods,
or assigning full-blend scores to each base model. Catalog generation does not change model values.

The following commands train; run individually as needed:

```powershell
python -m analysis.main feature-select
python -m analysis.main feature-optimize --stage all
python -m analysis.main feature-evidence
python -m analysis.main common-models --device cuda
python -m analysis.main canonical-submit
```

`common-models` reuses earlier `model-search` windows; see [model comparison](../methods/model_comparison.md) for new-machine preparation.
The report-selection 19/27/47/62 ladder has fixed settings and cannot be mixed with another wave's 27-column model table to estimate feature contributions.
Ladder training command:

```powershell
python -m analysis.experiments.report_selection_trials ridge
python -m analysis.experiments.report_selection_trials ladder --stage outer
python -m analysis.experiments.report_selection_trials ladder --stage june
```

Stage generators include `analysis.main project-review`, `report_selection_summary` and `report_best_scheme`.
They overwrite their Markdown outputs; choose the intended stage explicitly. Current results need no regeneration to read.

## Dependencies of later exploration

```text
Early search/features and row-level predictions
  → Wave-one frontier
  → Wave-two contemporaneous history and tree blending
  → Wave-three FT/XGBoost and historical main blend scoring442.90
  → Wave-four macro exploration → History/macro submission export

Earlier aligned predictions/models → Waves five–seven feature/deep/Transformer comparisons
Historical main blend → Wave-eight FT pruning and four-seed averaging
Earlier predictions → Wave-nine grouped blending
Wave nine + new wave-ten models → Structural blend and full-fit submission
Three exported CSVs + screenshot records → Final-selection archive
```

These are artifact dependencies, not requirements to include every branch in the final model. The final structural version excludes macro;
not every wave-five–seven candidate enters final either.
A full historical rerun requires restoring models, aligned predictions and selection records per wave README, not just running the final script.

Representative entries follow; each line illustrates one stage, not a complete rerun script:

```powershell
python -m analysis.experiments.frontier_search screen
python -m analysis.experiments.second_wave_anniversary screen
python -m analysis.experiments.third_wave_models screen
python -m analysis.experiments.fifth_wave_features screen
python -m analysis.experiments.sixth_wave_deep screen
python -m analysis.experiments.seventh_wave_transformers screen
python -m analysis.experiments.eighth_wave_refinement ft_bag
python -m analysis.experiments.tenth_wave_structures screen
python -m analysis.experiments.tenth_wave_structures outer
python -m analysis.experiments.tenth_wave_structures june
```

In waves two–seven, `screen` commonly selects on earlier windows, while `outer`/`june` refer to annual/recent development windows;
blend scripts often use `select` rather than `screen`. Check each parser, for example:

```powershell
python -m analysis.experiments.third_wave_ensemble select
python -m analysis.experiments.third_wave_ensemble outer
python -m analysis.experiments.third_wave_ensemble june
```

Wave-nine `ninth_wave_blending` fits/compares blend rules and has none of those positional arguments.
Development windows were repeatedly inspected; naming a command outer does not make it a new independent test.

## Local artifacts for three submissions

| File | Generation source | Prerequisites |
| --- | --- | --- |
| `kaggle_best_history.csv` | Historical export from `best_ensemble_submission`; candidate copy maintained by later archiving | Original ten-member recipe, selection records and raw data |
| `kaggle_best_macro.csv` | Same, plus supplied macro-data member | Historical/macro method records |
| `kaggle_structure_blend.csv` | Structural-blend export from `tenth_wave_report` | Wave-nine full-fit components, wave-eight FT seeds and wave-ten structure/blend records |

Candidates: `output/submissions/candidates/`.
Exact final copy: `output/submissions/final/submission.csv`; hashes/scores are in
[FINAL_SELECTION.json](../../experiment/01_final/submission_selection.json).

When all three CSVs and prerequisites exist and final archiving actually needs rebuilding, run:

```powershell
python -m analysis.experiments.final_submission_selection
```

This copies files and rewrites reports; it is not read-only. Screenshot scores are recorded manual evidence; the command does not query Kaggle for new scores.
Archive final selection last after rebuilding stage reports, checking generated text against screenshots, hashes and local metrics.

## Checks and tests

Lightweight checks for the new index:

```powershell
python -m unittest discover -s tests -p test_experiment_catalog.py -v
python -m analysis.main experiment-list --group wave10 --json
```

After model/blend changes, run relevant tests; these cover waves nine/ten only, without retraining all history:

```powershell
python -m unittest discover -s tests -p test_ninth_wave.py -v
python -m unittest discover -s tests -p test_tenth_wave.py -v
```

`python -m analysis.main verify` checks implemented environment/result coverage and reads local artifacts;
it is not ten-wave retraining from scratch. Full unit tests: `python -m unittest discover -s tests -v`,
subject to each test's environment/artifact requirements. Actual cleanup execution scope: [maintenance records](../maintenance/README.md).
