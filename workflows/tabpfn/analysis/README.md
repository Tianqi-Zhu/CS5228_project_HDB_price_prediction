[Chinese version](README.zh-CN.md)

# Code navigation

This checkout also retains the TabPFN branch's root-level experiment modules
and `reports/` outputs. The merged zch pipeline uses `analysis.pipeline`,
grouped experiments, and `output/`. For the shared names `baseline`,
`auxiliary_ablation`, `economic_feature_ablation`, and `spatiotemporal_hybrid`,
import zch through `analysis.experiments.baselines`; root-level imports retain
their existing TabPFN behavior. `python -m analysis.main` uses the zch workflow.

| What to inspect | Location |
| --- | --- |
| Shared data, features, training and prediction | `pipeline/` |
| Model and feature experiments | [experiments/](experiments/README.md) |
| Main/auxiliary data analysis | `eda/` |
| Economic, spatial and panel studies | `studies/` |
| Basic commands and training entry points | `main.py` |
| Experiment index without training | `experiment_catalog.py` |
| Refresh reading reports from existing results | `report_bundle.py` |
| Baseline-27 summary generator | `project_review.py` |
| Consistency checks on saved results | `verification.py` |
| Task-specific missing-resource and package-provenance checks | `resource_preflight.py` |
| Link saved weights, blend parameters and metrics | `artifact_catalog.py` |

```powershell
python -m analysis.main experiment-list
python -m analysis.main experiment-list --group wave10
python -m analysis.main experiment-refresh
python -m analysis.main preflight --profile final
```

Experiment code is grouped by purpose and wave; old module names and commands remain usable.
Reading summaries: [experiment/](../experiment/README.md); complete runs: `output/experiment_runs/`.
Training and dependencies: [running guide](../docs/guides/running_experiments.md).
