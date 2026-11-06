# CS5228 Project (Team 30): Singapore HDB Monthly Rent Prediction

> **Beyond Property Attributes: Chronological Local-History Aggregations and Hybrid Tree–Transformer Ensembles for Singapore HDB Rent Forecasting**
> ([Chinese version / 中文版说明](README.zh-CN.md) · [Documentation Language Policy](docs/documentation-languages.md))

This repository contains the complete, end-to-end reproducible codebase, feature engineering pipelines, model implementations, and frozen validation evidence for **Team 30**'s CS5228 Knowledge Discovery and Data Mining project on forecasting Singapore HDB monthly flat rents (`MONTHLY_RENT` in SGD).

---

## 1. Project Overview & Key Results

We forecast individual HDB monthly rental approval prices across **150,000 training records** and **50,000 test records** (`2021-01` to `2025-03`) supplemented by six auxiliary spatial and macroeconomic tables. Because rental prices exhibit strong post-2021 inflation (+44.6% from 2021 to 2024), our methodology enforces strict **chronological anti-leakage protocols** across two rolling-origin validation windows:
- **Annual Window ($Y$)**: Fit on `2021-01`–`2024-03` (114,842 rows), validate on `2024-04`–`2025-03` (35,158 rows).
- **Recent Window ($J$)**: Fit on `2021-01`–`2024-05` (121,005 rows), validate on `2024-06`–`2025-03` (28,995 rows).

| Pipeline / Model Family | Feature Schema | Annual ($Y$) RMSE | Recent ($J$) RMSE | Kaggle Public Score | Reproduction Entry Point |
|---|---|---:|---:|---:|---|
| **F1: 12-Channel Support-Regime Tree + PLR-FT Blend** | Nested `19 / 27 / 47 / 62` ladder | **451.03** | **441.83** | **415.38** | [reproduction/f1/README.md](reproduction/f1/README.md) |
| **F2: 7-Tree + TabPFN v2.5 Equal Hybrid** (Original / Grouped) | `62` tree + `87` TabPFN | — | **440.95 / 440.77** | **413.20** | [docs/guides/tabpfn_ensemble.md](docs/guides/tabpfn_ensemble.md) |
| **Standalone TabPFN v2.5 (87-feature)** | `77` base + `10` anniversary (`87`) | — | **441.96** | — | [docs/guides/tabpfn_best.md](docs/guides/tabpfn_best.md) |
| **Concat Transformer (Learned History Tokenizer)** | `60` base + `35` history (`95`) | **459.22** | — | **418.39** | [reproduction/concat_transformer/README.md](reproduction/concat_transformer/README.md) |
| **Standalone CPU MLP Ensemble** (Cleaned / Uncleaned) | `44`-column tabular pipeline | **469.92** | — | **437.34 / 439.38** | [reproduction/mlp/README.md](reproduction/mlp/README.md) |

---

## 2. Quick Start: 60-Second Offline Verification & Replay (CPU Only)

An outsider can verify data integrity, check all 4,367 shipped files, and reproduce our validation metrics and 50,000-row submission blends on any CPU machine without needing a GPU or external network access.

### Step 1: Set Up the Environment
```bash
# Option A: Lightweight CPU verification & replay environment (NumPy, pandas, matplotlib)
conda env create -f environment.review.yml
conda activate cs5228-review

# Option B: Full training environment (LightGBM, CatBoost, XGBoost, scikit-learn, PyTorch)
conda env create -f environment.training.yml
conda activate cs5228-training
```

### Step 2: Run the Offline Verification & Reproduction Suite
All commands below run deterministically on CPU and write any new artifacts to the git-ignored `output/` directory:

```bash
# 1. Verify SHA-256 integrity of all tracked code, raw datasets, and evidence files
python review.py verify
python -m analysis.validation.repository_audit

# 2. Reproduce the 50,000-row F1 submission CSV from saved 12-channel component predictions (max diff ~1.8e-12 SGD)
python reproduction/f1/verify.py
python reproduction/f1/reproduce.py

# 3. Verify the fresh 30-model F1 chronological validation refits (windows Y and J) and pre-origin blend replay
python -m analysis.main validation-refits

# 4. Reconstruct the 50,000-row F2 submission and run the paired 28,995-row F1 vs. F2 comparison on window J
python -m analysis.main final-pair

# 5. Verify ordered feature schemas (19/27/47/62/87/95) and M01 spatial/cleaning provenance
python -m analysis.main schemas
python -m analysis.main evidence

# 6. Verify standalone neural & TabPFN packages (Concat Transformer, CPU MLP, and TabPFN workflows)
python -m analysis.main concat verify
python -m analysis.main mlp verify
python review_tabpfn.py verify

# 7. Run the complete end-to-end validation audit across all experiment stages (M00–M09)
python -m analysis.main planned
```

---

## 3. Retracing the Report Step-by-Step (Code & Command Map)

To make it effortless to trace every section, table, and figure in our report back to source code, use the map below:

| Report Section | Topic & Methodology | Source Code Location | Command to Reproduce / Retrain |
|---|---|---|---|
| **Section II** | **EDA, Data Cleaning & Spatial Auxiliaries**<br>Text normalization, duplicate policies, training-only outlier screening (`M01`), Haversine `BallTree` distances (`0.5/1.0/2.0 km`), COE & stock macro series | [`analysis/eda/`](analysis/eda/)<br>[`analysis/pipeline/data.py`](analysis/pipeline/data.py)<br>[`analysis/pipeline/auxiliary.py`](analysis/pipeline/auxiliary.py) | `python -m analysis.main eda --scope all --output-dir output/review_eda`<br>`python -m analysis.main evidence` |
| **Section III** | **Validation Protocol & Classical Baselines**<br>Chronological windows $Y$ & $J$ vs. random K-fold leakage (`431.78` vs. `484.33`), `C7` classical baselines (Mean, Ridge, RF, LightGBM, CatBoost; `M02`) | [`analysis/experiments/baselines/`](analysis/experiments/baselines/)<br>[`docs/methods/prediction-scenario.md`](docs/methods/prediction-scenario.md) | `python -m analysis.main baseline --output-dir output/review_baseline`<br>`python -m analysis.main compare --baseline-dir output/review_baseline --output-dir output/review_compare` |
| **Section IV** | **Nested Feature Engineering Ladder (`19→27→47→62`)**<br>Causal prior-month hierarchy (`History27`), macro/spatial (`Aux47`), lease-anniversary & kernel cohorts (`Full62`), and TabPFN `77→87` schema (`M03`) | [`analysis/pipeline/features.py`](analysis/pipeline/features.py)<br>[`analysis/experiments/feature_engineering/`](analysis/experiments/feature_engineering/)<br>[`experiment/main/M03/feature_schemas/README.md`](experiment/main/M03/feature_schemas/README.md) | `python -m analysis.main schemas`<br>`python -m analysis.main ladder --stage outer` |
| **Section V** | **Tree Ensembles, Neural Architectures & TabPFN**<br>Matched 27-feature comparison (`LightGBM`, `CatBoost`, `XGBoost`, `MLP`, `FT-Transformer`, `TabM`), `PLR-FT` vs. `PLE-FT`, `Concat Transformer` (`M05`), and `TabPFN v2.5` (`M07`) | [`analysis/experiments/model_comparison/`](analysis/experiments/model_comparison/)<br>[`reproduction/concat_transformer/README.md`](reproduction/concat_transformer/README.md)<br>[`reproduction/mlp/README.md`](reproduction/mlp/README.md)<br>[`workflows/tabpfn/README.md`](workflows/tabpfn/README.md) | `python -m analysis.main common-models --device cuda --output-dir output/review_common`<br>`python -m analysis.main concat predict --device cpu`<br>`python -m analysis.main mlp fullfit --output-dir output/mlp/new-run`<br>`python run_tabpfn_best.py preflight` |
| **Section VI** | **Final Ensembles F1 & F2**<br>• **F1**: 12-channel support-regime blend (`2×2` indicators on `ANN_BLOCK_24/36`)<br>• **F2**: Equal `0.5 / 0.5` hybrid of 7 trees + TabPFN87 (`M04`, `M08`, `M09`) | [`reproduction/f1/`](reproduction/f1/)<br>[`run_tabpfn_ensemble.py`](run_tabpfn_ensemble.py)<br>[`analysis/validation/final_pair.py`](analysis/validation/final_pair.py) | **CPU Replay**: `python reproduction/f1/reproduce.py` & `python -m analysis.main final-pair`<br>**Full GPU Refit**: `python reproduction/f1/validate.py --window Y --output-dir output/f1-Y` & `python reproduction/f1/train.py` |
| **Section VII** | **Post-Hoc Error Diagnostics**<br>Support-regime error decomposition, rent-bracket tail bias, and block-cohort sparsity analysis | [`analysis/validation/main.py`](analysis/validation/main.py)<br>[`experiment/main/README.md`](experiment/main/README.md) | `python -m analysis.main planned` |

---

## 4. Repository Structure & Organization

```text
├── data/                           # Read-only competition datasets (150k train, 50k test) & 6 auxiliary tables
│   └── auxiliary/                  # MRT stations, malls, primary schools, hawker centres, COE, stock prices
├── analysis/                       # Shared core Python package (CLI entry point: `python -m analysis.main`)
│   ├── eda/                        # Exploratory data analysis & visualization generators
│   ├── pipeline/                   # Shared data cleaning, spatial/macro joins, 19/27/47/62 feature builders
│   ├── experiments/                # Modular experiment runners (baselines, feature ladder, model comparison)
│   ├── studies/                    # Supplementary temporal and spatial domain studies
│   └── validation/                 # Offline reproducibility & anti-leakage audit suite (M00–M09)
├── reproduction/                   # Self-contained reproduction packages for final & standalone models
│   ├── f1/                         # Final F1 (12-channel tree + PLR-FT): verify.py, reproduce.py, validate.py, train.py
│   ├── concat_transformer/         # Standalone 95-input Concat Transformer weights, replay & training scripts
│   └── mlp/                        # Standalone 44-input CPU MLP ensemble, uncertainty intervals & recency weighting
├── workflows/tabpfn/               # Isolated TabPFN v2.5 & 7-tree F2 workflow (accessed via review_tabpfn.py / run_tabpfn_ensemble.py)
├── experiment/main/                # Frozen validation metrics, schemas, and audit logs organized by stage (M00–M09)
├── docs/                           # Detailed methodology notes, execution guides, and architecture documentation
├── requirements/                   # Pinned pip requirement files for core, deep learning, and TabPFN environments
├── review/                         # SHA-256 repository manifest (manifest.json) and verification logs (CHECKS.md)
└── tests/                          # Automated unit and integration test suite
```

### Detailed Navigation Guides
| Topic | Documentation Link |
|---|---|
| Complete CLI reference & full training commands | [Running Experiments Guide](docs/guides/running_experiments.md) |
| Directory layout & artifact retention rules | [Repository Layout Guide](docs/guides/repository_layout.md) |
| Shared `analysis/` package architecture | [Analysis Package Guide](analysis/README.md) |
| Experiment stage index & frozen evidence (`M00`–`M09`) | [Experiment Index (`M00`–`M09`)](experiment/main/README.md) |
| Validation execution status & audit summary | [Execution Status](docs/planning/main-execution-status.md) · [Five Verification Checks](docs/planning/main-five-item-completion.md) |
| Temporal cutoffs & prediction scenario definitions | [Prediction Scenario Definition](docs/methods/prediction-scenario.md) |
| Unified main & appendix experiment design | [Experiment Design Plan](docs/planning/main-appendix-experiment-plan.md) |
| F2 TabPFN + 7-Tree ensemble guide & provenance | [F2 Ensemble Guide](docs/guides/tabpfn_ensemble.md) · [M07 Status](experiment/main/M07/README.md) · [M08 Paired Results](experiment/main/M08/README.md) · [TabPFN Integration Notes](docs/guides/anniversary_merge.md) |
| Verification logs & manifest checks | [Verification Scope (`review/CHECKS.md`)](review/CHECKS.md) |
| Raw & auxiliary dataset documentation | [Data Directory README](data/README.md) |
| Unit & integration test suite | [Tests README](tests/README.md) |
| Method deep-dives (baseline, history, model comparison, MLP) | [Baseline Features](docs/methods/baseline_features.md) · [History Optimization](docs/methods/history_feature_optimization.md) · [Model Comparison](docs/methods/model_comparison.md) · [MLP Report](docs/mlp_prediction_report.md) |
