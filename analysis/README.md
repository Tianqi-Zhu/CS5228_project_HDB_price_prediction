[Chinese version](README.zh-CN.md)

# Main code entry points

| Task | Implementation | Command |
|---|---|---|
| Main requirement-card audit and error diagnostics | `validation/` | `python -m analysis.main planned` |
| Feature definitions and new M01 evidence | `validation/` | `python -m analysis.main schemas`, `evidence` |
| F1 Y/J generation chain | `reproduction/f1/validate.py`, `validation/refit_audit.py` | Fixed-recipe retraining; `python -m analysis.main validation-refits` verifies offline |
| F2 components and recent F1/F2 pairing | `validation/final_pair.py` | `python -m analysis.main final-pair`; no training |
| File integrity | Root `review.py` | `python review.py verify` |
| Static audit of all main content | `validation/repository_audit.py` | `python -m analysis.validation.repository_audit` |
| EDA | `eda/` | `python -m analysis.main eda --scope all` |
| C7 baseline and classical models | `experiments/baselines/` | `python -m analysis.main baseline`, `compare` |
| Controlled feature ladder | `experiments/reporting/report_selection_trials.py` | `python -m analysis.main ladder --stage outer` |
| Common 27-input models | `experiments/model_comparison/` | `python -m analysis.main common-models` |
| Concat model replay | `experiments/model_comparison/concat_transformer/` | `python -m analysis.main concat verify` |

See `python -m analysis.main --help` and the [running guide](../docs/guides/running_experiments.md) for training options. `planned` analyzes frozen predictions without training. Its outputs are development analyses; missing F2 evidence remains outstanding.

See [repository layout](../docs/guides/repository_layout.md) for responsibilities, retention reasons and cleanup records. New EDA and study outputs default to `output/`; `experiment/main/` stores frozen evidence.

`pipeline/` provides shared raw-data, feature and learner implementations. `experiments/` retains main dependencies by purpose and historical round; `studies/` holds research modules needed by tests and preprocessing. Legacy module aliases support checkpoint loading; dependencies are not new main experiments.

The fixed F1 recipe is isolated in `reproduction/f1/runtime/` so current code changes do not affect original-submission reproduction. TabPFN lives in `workflows/tabpfn/`; root `review_tabpfn.py` assembles its independent working directory under `output/`.

[Results and status](../experiment/main/README.md) · [Current requirement cards and plan](../docs/planning/main-appendix-experiment-plan.md)

## Standalone MLP commands

`python -m analysis.main mlp` lists commands; `mlp verify` verifies offline, `mlp fullfit` trains on all data using CPU, and `mlp recency-search` compares temporal weights. The runtime is in [reproduction/mlp](../reproduction/mlp/README.md) and does not overwrite the shared main pipeline.
