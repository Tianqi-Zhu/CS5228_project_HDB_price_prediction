[Chinese version](branch_integration.zh-CN.md)

# Integration of five working branches

This records integration scope on `feature/midterm-eda-models`. Changes affect that branch only; other remote branches remain historical sources.

## Sources and destinations

| Source branch | Unique retained content | Unified destination |
| --- | --- | --- |
| `feature/midterm-eda-models` | Main-data EDA, A/B/C features, four-model comparison, temporal validation and submission pipeline | `analysis/pipeline/`, `analysis/eda/` and baseline experiments |
| `feature/temporal-auxiliary-analysis` | Auxiliary ablations, lagged economic signals, grouped temporal panels and fixed-origin historical spatial models | `analysis/experiments/baselines/auxiliary_ablation.py`, `analysis/studies/`, `spatiotemporal_hybrid.py` |
| `feature/neural-tabm-experiment` | TabM-style network and blending with historical trees | `analysis/experiments/baselines/neural_tabm.py` |
| `relu-rental-prediction` | Nested temporal validation and ReLU MLP with fitting-only IQR cleaning | `analysis/experiments/baselines/relu_mlp.py` |
| `ziyuan——spatialfeatures` | Spatial dictionary, descriptive analysis, fixed effects, joint facility controls and overlap-stratified matching | `analysis/studies/spatial/` and unified `spatial-controls` entry |

`main` has no unique project content relative to these ancestor commits, so no empty scaffold is copied.

## Deduplication principles

- All commands use `python -m analysis.main`.
- Raw data come only from `data/*.csv` and `data/auxiliary/*.csv`; no second processed CSV is committed.
- Spatial analysis shares `analysis/pipeline/auxiliary.py`; `spatial-controls` builds one canonical in-memory feature definition.
- Duplicate wrappers/progress reports, row-level validation/test predictions, model binaries and large intermediate tables are not copied.
- Small metrics, figures and summaries remain reviewable; large regenerable artifacts are excluded by `.gitignore`.

## Metric comparability

| Route | RMSE | Validation definition | Interpretation |
| --- | ---: | --- | --- |
| Main-table LightGBM | 485.09 | 2024-04 through 2025-03 | Unified 12-month baseline |
| Spatial/facility ablation E | 476.97 | Same as above | Static building/facility snapshots provide the main gain |
| Fixed blend of deep model, auxiliary trees and historical prior | **466.47** | Same as above | Best saved result under the current matched protocol; MAE 341.14 |
| Historical spatial tree | 469.61 | First fixed-origin fold | Three-fold pooled RMSE 543.10; longer extrapolation is substantially harder |
| TabM + historical tree | 468.83 | First fixed-origin fold | Only 6% neural weight; three-fold pooled RMSE 542.78 |
| ReLU MLP | 463.97 | 2025-01 to 2025-03 | Only three months with nested epoch selection; not directly rankable against 12-month results |

Spatial fixed effects and matching are interpretive studies without predictive RMSE rankings. Original spatial model binaries and row predictions were not copied; compact LightGBM, CatBoost and KNN metrics are in `output/experiment_runs/baseline_spatial_models/`. Compact historical ReLU metrics are in `output/experiment_runs/baseline_relu_mlp/historical_results.json`.

## Recommended reproduction order

```powershell
conda activate cs5228
python -m unittest discover -s tests -v
python -m analysis.main aux-ablation
python -m analysis.main spatial-controls
python -m analysis.main deep-summary
```

`tabm`, `spatiotemporal` and `relu` are further experiments with different windows; read split dates in result files rather than comparing RMSE alone.
