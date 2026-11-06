[Chinese version](README.zh-CN.md)

# Experiment code by purpose

| Folder | Content |
| --- | --- |
| `baselines/` | Baseline trees, early deep networks, MLP and spatial baselines |
| `feature_engineering/` | Feature selection, history representations and removal/retraining |
| `model_comparison/` | Shared-input models, search and preparation tools |
| `optimization/` | Waves one through ten, each in a descriptive folder |
| `reporting/` | Model selection, report summaries and historical branch inventory |
| `submissions/` | Baseline references, history/macro variants and final-selection exports |

`optimization/` runs from `round_01_history_target_search` to
`round_10_plr_transformer_final`; models and run records use matching wave names.

## Commands and actual file locations

```powershell
python -m analysis.main experiment-list --group wave10 --json
python -m analysis.experiment_catalog --group final --json
```

JSON `source` gives the current actual file path; `command` retains the original module command.
For example, wave ten still runs with `python -m analysis.experiments.tenth_wave_structures screen`.
This preserves existing script calls and module references in checkpoints.

The index parses source only, without training. It lists command prefixes; required arguments and prerequisites are in the [running guide](../../docs/guides/running_experiments.md).
Reading reports are in [experiment/](../../experiment/README.md); running every historical experiment is unnecessary.
