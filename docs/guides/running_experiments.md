[Chinese version](running_experiments.zh-CN.md)

# Local review and execution entry points

| Operation | Command | Behavior |
|---|---|---|
| Inspect experiment ownership | `python review.py list` | Read-only |
| Verify file and data hashes | `python review.py verify` | Read-only; no training |
| Verify the original submission | `python reproduction/f1/verify.py` | Standard library; read-only |
| Reconstruct predictions | `python reproduction/f1/reproduce.py` | CPU/NumPy; writes to ignored output/; no training |
| Full-data retraining of the original recipe | `python reproduction/f1/train.py` | CUDA/full training dependencies; writes results to output/ |
| Fixed-recipe Y/J retraining | `python reproduction/f1/validate.py --window Y --output-dir output/f1-validation/new-Y` | CUDA; use a separate new directory for J; 15 base models per window |
| Verify Y/J generation chains | `python -m analysis.main validation-refits` | CPU; verifies code/data identity, row alignment and fixed-blend reconstruction |
| F2 candidates and recent paired comparison | `python -m analysis.main final-pair` | CPU; reconstructs 50,000-row components, pairs 28,995 validation rows and groups by month/type/support; no training or test labels |
| Ordered feature lists | `python -m analysis.main schemas` | CPU; distinguishes the two 62-column schemas and component inputs |
| Spatial and cleaning evidence | `python -m analysis.main evidence` | CPU; verifies source bytes, data versions, matched controls and available row predictions |
| Training entry points | `python -m analysis.main --help` | Main exposes selected entry points; appendix retains the original CLI. Executing baseline/common-models/ladder and similar commands trains models. |

| Environment file | Purpose |
|---|---|
| environment.yml | Original full-project environment; contents unchanged |
| environment.review.yml | Original final-branch CPU review environment |
| environment.training.yml | Original final-branch CUDA training environment |
| requirements/ | Complete original optional-model dependencies |

| Data | Status |
|---|---|
| data/train.csv, test.csv, example-submission.csv | Complete original files with matching provenance hashes |
| data/auxiliary/ | All six supplied auxiliary tables preserved |
| Caches/large models | Original-workspace caches are not copied. Full historical training requires rebuilding configured resources; original-submission CPU reconstruction needs no checkpoints. |
| Historical metrics | Small local results ignored by Git were copied into experiment/main/ with provenance. Large row predictions and all development models are outside this split's scope. |

## Main training commands

```bash
python -m analysis.main eda --scope all --output-dir output/review_eda
python -m analysis.main baseline --output-dir output/review_baseline
python -m analysis.main compare --baseline-dir output/review_baseline --output-dir output/review_compare
python -m analysis.main ladder --stage outer
python -m analysis.main common-models --device cuda --output-dir output/review_common
```

Each subcommand's `--help` displays options without training. `compare` first generates a baseline in the requested directory if needed. `ladder` builds features directly from raw data, independently of old blend predictions. `common-models` uses local historical inner-search records where available, otherwise fixed records bundled in `experiment/main/M05/selection_records/`; it does not retune on outer evaluation data.

All EDA outputs default to `output/eda/`, including overview and both specialized audits; supplementary studies default to `output/studies/`. Frozen evidence stays in `experiment/main/`. Earlier PDFs and reading bundles go to `output/reports/` and `output/reading_bundle/`; see [repository layout and artifact rules](repository_layout.md).

## Standalone Concat Transformer

`python -m analysis.main concat verify` checks the archive; `python -m analysis.main concat predict --device cpu` loads final weights for replay. See [reproduction instructions](../../reproduction/concat_transformer/README.md) for raw-data reconstruction, validation training and full-data retraining. Only explicit select/refit commands start training.

## main requirement-card evaluation

`python -m analysis.main planned` audits the frozen Y/J predictions and generates per-card diagnostics without training. See [execution status](../planning/main-execution-status.md). Outputs go to a new content-addressed directory under `output/planning_validation/main/`; existing runs are never overwritten. `--output-dir output/planning_validation/recheck` selects another directory.

`python -m analysis.validation.repository_audit` checks all shipped hashes, every registered Python file syntax and active navigation. Model runtime checks and CPU replays are separately logged in `review/checks/main-final/`.

See [five improvements](../planning/main-five-item-completion.md) for the latest completion and acceptance record. `planned` also runs the three new offline audits; `new_models_fitted=0` means that audit command trains nothing, while separately recording 30 completed Y/J base-model fits. Original inputs are retained; a separate schema inventory corrects old 57-column PLE/PLR metadata without altering original predictions.

When reusing local third-party directories, F1/main use `.cache/deep-deps` and `.cache/model-deps`. SDK directory `.cache/tabpfn-split-deps` is only for isolated TabPFN tests/commands. Do not combine all three into main's general search path. Installations should follow the corresponding environment and requirements files; this round created no Conda environment.

## MLP ensemble training and prediction

The full MLP is integrated through an independent runtime. Run `python -m analysis.main mlp verify` to verify original submissions and `python -m analysis.main mlp fullfit --output-dir output/mlp/new-run` to retrain. `--half-life-months 36` is optional; see [MLP reproduction](../../reproduction/mlp/README.md) for interval and cleaning commands. New training files do not inherit historical screenshot scores.
