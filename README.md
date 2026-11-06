[Chinese version](README.zh-CN.md)

English is the default documentation language; Chinese is available as an additional version. See the [language policy and translation index](docs/documentation-languages.md).

# HDB monthly rent prediction experiments

Predict individual HDB `MONTHLY_RENT` values in SGD. `main` retains essential controls and final methods; `appendix` retains alternatives, ablations, negative results and the full exploration. Original competition data, environment files and project architecture are preserved.

| Start here | Location |
|---|---|
| Experiment index and current status | [M00–M09](experiment/main/README.md) |
| Results and remaining gaps for the new requirement cards | [Execution status](docs/planning/main-execution-status.md) |
| Five completed improvements and acceptance checks | [Five improvements](docs/planning/main-five-item-completion.md) |
| Prediction scenario and information availability | [Scenario definition](docs/methods/prediction-scenario.md) |
| Original experiment requirements and main/appendix plan | [Unified plan](docs/planning/main-appendix-experiment-plan.md) |
| Environment, training and offline verification commands | [Running guide](docs/guides/running_experiments.md) |
| Code structure | [analysis/](analysis/README.md) |
| Directory responsibilities and intermediate-artifact cleanup | [Repository layout](docs/guides/repository_layout.md) |
| Course assessment criteria | [SKILL.md](skills/cs5228-final-report/SKILL.md) |

```powershell
python review.py list
python review.py verify
python -m analysis.main planned
python -m analysis.main schemas
python -m analysis.main evidence
python -m analysis.main validation-refits
python -m analysis.main final-pair
```

`planned` recomputes frozen development predictions, grouped errors and structural controls; verifies the F1 Y/J generation chain, feature lists and M01 provenance; and reconstructs F2 candidates and the recent paired comparison. It performs no training or search. `final-pair` runs the last comparison independently. New results go into ignored `output/`; existing runs are not overwritten, so repeat runs require a new `--output-dir`. Y/J are previously examined development periods.

| Method | Status and evidence | Entry point |
|---|---|---|
| F1: grouped blending + PLR-FT | Original displayed submission score 415.37529; annual/recent development RMSE 451.0295/441.8345. Frozen CSVs, component reconstruction and full-data training are retained separately. | [F1 reproduction package](reproduction/f1/README.md) |
| Concat Transformer | Independent architecture exploration with matched inputs; annual development RMSE 459.2189; team-recorded Public Score 418.38657. Integrated code, capacity comparisons, weights and replay commands are preserved. | [Method](experiment/main/M05/concat_transformer/README.md), [reproduction](reproduction/concat_transformer/README.md) |
| Independent MLP ensemble | Original full-data cleaned/uncleaned submissions display 437.34455/439.38193. CPU training, prediction intervals, temporal weighting and test commands are integrated; the F1 recipe is unchanged. | [MLP reproduction package](reproduction/mlp/README.md), [report](docs/mlp_prediction_report.md) |
| Standalone TabPFN87 | Recent RMSE 441.9593; independent remote fit/inference verification is recorded. Local execution still requires the SDK and original checkpoint. | [Standalone execution](docs/guides/tabpfn_best.md) |
| F2: equal blend of seven trees and TabPFN87 | Original-category recent RMSE 440.9520, independently pairable and reconstructable. The new grouped candidate scores 440.7707 and is listed separately as a category sensitivity variant. Both new submissions and complete inference records are integrated, with no new Kaggle score. Annual evaluation, the complete ordered dictionary and screenshot association remain outstanding. | [F2 status](experiment/main/M07/README.md), [paired results](experiment/main/M08/README.md), [new submissions and reproduction](docs/guides/tabpfn_ensemble.md) |

Run `python -m analysis.validation.repository_audit` for integrity checks; see [review/CHECKS.md](review/CHECKS.md) for actual verification scope and logs. Screenshot scores are not new execution results. Compare different information settings, windows and schemas separately.

`data/` holds 150,000 training records, 50,000 test records, the submission template and six auxiliary tables. `analysis/` contains shared implementations; `reproduction/` holds fixed models and recipes; `workflows/tabpfn/` contains the isolated workflow; `experiment/main/` stores scientific evidence. Rebuildable caches, training outputs and large temporary directories belong in ignored `output/` or `.cache/`.

## Independent workflows

| Workflow | Verification without training | Training or inference |
|---|---|---|
| F1 | `python reproduction/f1/verify.py` | [Fixed recipe](reproduction/f1/README.md) |
| Concat | `python -m analysis.main concat verify` | [Original-model replay and training](reproduction/concat_transformer/README.md) |
| MLP | `python -m analysis.main mlp verify` | [CPU full-data training, intervals and temporal weighting](reproduction/mlp/README.md) |
| TabPFN | `python review_tabpfn.py verify`; `python run_tabpfn_best.py preflight` | [SDK/checkpoint prerequisites and standalone execution](docs/guides/tabpfn_best.md) |

All workflows share the root `data/` directory; isolated runtimes must not overwrite one another. Full exploration and historical provenance are retained. MLP and Concat are not added directly to F1/F2. See [provenance and boundaries](docs/guides/anniversary_merge.md) for the TabPFN integration. Existing [report materials](reports/final_report/README.md) are retained; this integration round does not revise the manuscript.
