[Chinese version](README.zh-CN.md)

# CS5228 HDB monthly rent prediction

This branch contains the TabPFN workflows in `analysis/` and `reports/` and the merged zch workflow in `analysis/pipeline/`, grouped experiment modules, and `experiment/`. Existing TabPFN entry points and report paths remain available. The [zch workflow guide](#zch-project-workflow) below describes its pipeline, results and output directories.

The [industry-stock experiment](reports/stock_sector_signals/report.md) compares
eight Singapore-listed business-exposure baskets, broad-market controls, delayed
returns, momentum, and volatility. It tests both the property-adjusted monthly
rent index and individual-rental errors, with earlier-period sector selection,
mature-label forecast cutoffs, and separate monthly-updated and fixed-origin
scenarios. Reproduce with
`.venv/Scripts/python.exe analysis/compare_stock_sectors.py`.

The [company-worksite stock experiment](reports/company_worksite_stocks/report.md)
tests local rent changes around 11 documented engineering/manufacturing sites.
It adjusts for building/flat mix and nearby comparison-area rent changes,
controls broad and peer stock signals, and separates primary 2 km tests from
earlier-selected radius/lag sensitivities. Reproduce with
`.venv/Scripts/python.exe analysis/acquire_company_worksites.py` followed by
`.venv/Scripts/python.exe analysis/compare_company_worksite_stocks.py`.

The [company-worksite follow-up findings](reports/company_worksite_followup/findings.md)
freeze the earlier K&S hypothesis and add US semiconductor/currency controls,
32 matched geographic placebos, distance bands, a controlled 77-feature
CatBoost ablation, and an exactly local correction of saved TabPFN backtests.
The follow-up does not support promoting company-location stock features;
the existing submission files are retained. See the findings for commands,
availability scenarios, source audits and limits of the reused validation period.

The [overall stock-index and regional rent analysis](reports/stock_index_regions/report.md)
constructs a 22-stock historical-cap-seeded index from archived December 2020
STI ETF weights, with equal-weight controls. It compares raw and property-adjusted
mean-rent growth across Singapore's five regions, tests regional slope equality
with synchronized month-block resampling, and checks forward prediction value.
Reproduce with `.venv/Scripts/python.exe analysis/compare_stock_index_regions.py`.

The [financial-sector and mean-rent experiment](reports/financial_sector_rents/report.md)
compares DBS/OCBC/UOB, broader available financial-services baskets, and individual
stocks against national arithmetic mean HDB rent growth. Lagged correlations,
composition adjustments, market controls and expanding monthly forecasts do not
support adding these signals to the current feature set. Reproduce with
`.venv/Scripts/python.exe analysis/compare_financial_sector_rents.py`.

The [financial-stock experiment by room type and flat model](reports/financial_sector_rent_categories/report.md)
tests five room types and nine sufficiently populated flat-model categories.
It compares raw and property-adjusted mean-rent growth, corrects tests across
categories, directly compares category slopes, and evaluates forward forecasts.
The results do not support adding category-specific financial-stock features.
Reproduce with
`.venv/Scripts/python.exe analysis/compare_financial_sector_rent_categories.py`.

The [targeted spatial-temporal and economic experiments](reports/targeted_spatial_economic/report.md)
compare hierarchical town/street histories, archived town housing completions,
and historical residual smoothing against the optimized frozen three-member
TabPFN ensemble. The study preserves all original member inputs and targets,
reuses audited predictions, and documents unavailable historical vintages for
quarterly rental-cycle and workforce-demand inputs. See the report for separate
ablation results, chronological selection, uncertainty, and reproduction commands.

The [A/B/C/D/E weighted-ensemble comparison](reports/tabpfn_abcde_ensemble/report.md)
tests the MSE-only DiagScale, original frozen, relative-history, seasonal and
spatial models using identical historical queries and earlier-only weights.
It contrasts ABCDE with CDE/BDE and focused subsets, separates the earlier-selected
family from the lowest observed later score, and retains both prior submissions.
Reproduce with `.venv/Scripts/python.exe analysis/compare_tabpfn_abcde.py --stage all`.

The [two full-data ensemble submissions](reports/tabpfn_top2_final_submission/report.md)
prepare equal-weight ABCDE and weighted ABDE using all 150,000 training rows and
50,000 test rows. They reuse audited full-data A/B/D/E predictions, newly fit the
relative-history C model, and preserve the original ensemble weights. The report
links both upload CSVs and distinguishes development scores from the unlabeled
Kaggle test. Reproduce with
`.venv/Scripts/python.exe analysis/train_tabpfn_top2_submission.py --stage all`;
independently verify with `.venv/Scripts/python.exe analysis/audit_tabpfn_top2_submission.py`.

The [revised dynamic rental surface experiment plan](EXPERIMENT_PLAN_DYNAMIC_SURFACE.md) prioritizes a stronger historical row model and a coarse market-index test before any grid/PCA hybrid.

The completed [full-history tree, market-index, and coarse-surface experiment](reports/dynamic_market_index/report.md) selects CatBoost without a market correction. Its strict February 2025-origin candidate submission is in `reports/dynamic_market_index/submission_strict_feb2025.csv`.

The [factored dynamic rental-surface experiment](reports/factored_rental_surface/report.md) evaluates monthly smoothed surfaces, PCA factors, fixed-origin temporal forecasts, and direct/residual CatBoost hybrids. The residual hybrid improves pooled RMSE by S$1.23 but is less consistent across folds, so the selected candidate remains full-history CatBoost.

## TabPFN experiment

The [TabPFN API workflow for the original 77 features](TABPFN_API_77.md) supports
newer managed model variants, Thinking mode, cost estimates, resumable fits and
inference, chronological validation, and separate submission CSVs.

The chronological 80/20 follow-up uses January 2021-May 2024 as context
(121,005 rows) and June 2024-March 2025 as validation (28,995 rows).
The [aligned temporal residual experiment](reports/tabpfn_v3_5_fast_aligned_temporal_80_20/report.md)
scores S$455.86 RMSE. Its histories and temporal forecasts share the same
information cutoff for every context row.

The [comprehensive temporal experiment](reports/tabpfn_v3_5_fast_dynamic_temporal_80_20/report.md)
tests a property-adjusted monthly market index, short and long trends,
seasonality, local activity and volatility, and optional lagged category-A COE
effects. Earlier forecast periods select the temporal corrections. The saved
candidate scores S$457.97 RMSE and worsens all ten validation months against
the aligned residual model, which remains the selected TabPFN setup.
The feature bundle contains raw rent labels: train TabPFN on
`y_train - train_baseline` and reconstruct with
`predicted_residual + query_baseline`. Run
`.venv/Scripts/python.exe analysis/compare_tabpfn_dynamic_temporal_80_20.py`
or add `--prepare-only` to save features without GPU inference. These 80/20
scores are development results and are not directly comparable with the
multi-fold scores below.

The [controlled month-encoding and calendar experiment](reports/tabpfn_v3_5_fast_calendar_80_20/report.md)
holds the aligned residual baseline and non-month features fixed. Categorical
month scores S$455.81 RMSE, effectively tied with the S$455.86 integer control.
Cyclical month, intake-window proxies, and school interactions score S$459.07,
S$457.03, and S$457.42 respectively. The earlier-selected calendar candidate
fails outer confirmation, so the integer control remains recommended. All new
features are complete in context and validation; 35 tests passed. Reproduce
with `.venv/Scripts/python.exe analysis/compare_tabpfn_calendar_80_20.py`.

The [TabPFN/CatBoost ensemble experiment](reports/tabpfn_catboost_ensemble_80_20/report.md)
fits blend weights on earlier out-of-time forecasts. The selected blend uses
66.3% aligned TabPFN and 33.7% original-feature CatBoost and scores S$456.01
validation RMSE (MSE 207,942.32), slightly worse than TabPFN alone. The 50/50
blend scores S$456.63. The best observed alternative scores S$455.76 but was
not the pair selected on earlier data and improves its TabPFN component by
only S$0.05. Retain the original aligned TabPFN. Reproduce with
`.venv/Scripts/python.exe analysis/compare_tabpfn_catboost_ensemble.py`.

The [frozen Fast embedding-head experiment](reports/tabpfn_fast_embedding_head_80_20/report.md)
uses the numerically best categorical-month setup, all 121,005 context rows,
and 10,800 earlier out-of-time query embeddings to train and select an MLP.
The selected raw-embedding MLP improves historical calibration RMSE from
S$544.60 to S$525.56, but raises the complete 80/20 holdout RMSE from
**S$455.81 to S$460.90**; all ten validation months worsen. Its learned
downward correction follows a historical TabPFN overprediction bias that
largely disappears on the outer holdout. Retain the frozen TabPFN baseline.
Reproduce with
`.venv/Scripts/python.exe analysis/compare_tabpfn_fast_embedding_head_80_20.py`.

The [TabPFN Kaggle submission](reports/tabpfn_fast_final_submission/report.md)
fits the categorical-month Fast regressor on all 150,000 labeled training rows
and writes 50,000 predictions to
`reports/tabpfn_fast_final_submission/submission_tabpfn_fast_150k.csv`.
March 2025 test histories stop at February; later test histories stop at
March 2025. The temporal forecast is held at its month-ten level for the
18,077 test rows beyond its validated range. The test set has no rent labels,
so this file has no measurable test RMSE. Reproduce with
`.venv/Scripts/python.exe analysis/train_tabpfn_fast_submission.py`.

The completed [TabPFN context comparison](reports/tabpfn/comparison_report.md)
retains **CatBoost at S$539.29 pooled RMSE**. TabPFN-3.5 with rental histories
scores S$545.76 with 20,000 context rows and S$541.46 with 50,000. The best
forward blend scores S$540.13. The larger context improves TabPFN, but neither
standalone model nor blend beats CatBoost on pooled RMSE. Detailed reports
include fold/monthly scores, plots, timings, memory and verified row predictions.

The subsequent [feature-engineering and frozen-head experiment](reports/tabpfn_adaptation/report.md)
uses a matched single-estimator/10,000-context backbone. Recent rent features
improve its pooled RMSE from **S$546.55 to S$538.58**, slightly below CatBoost's
S$539.29. The gain is S$0.71 and the second/third folds worsen, so the
prespecified consistency rule retains CatBoost. Frozen embedding Ridge and MLP
heads score S$628.41 and S$630.54; every residual head worsens every fold.
The conditional backbone-gradient pilot was not triggered. The
[experiment plan](EXPERIMENT_PLAN_TABPFN_ADAPTATION.md) records the design and
gate before scoring. This follow-up verifies 24,000 historical query tasks,
79,817 outer rows and chronological inner-head selection.

`analysis/compare_tabpfn.py` compares local TabPFN-3.5 with spatial-only inputs,
TabPFN with strictly earlier rental histories, and full-history CatBoost on the
same three disjoint fixed-origin folds. A CatBoost/TabPFN-history blend learns
its weight only from earlier folds; the first fold uses CatBoost alone.
Reports, row predictions, source hashes, package versions, GPU memory and
timings are written to `reports/tabpfn/`. No future COE/stock observations enter
the experiment. The existing CatBoost submission candidate is retained.

Install into the project virtual environment and authenticate interactively:

```powershell
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements-tabpfn.txt
.venv/Scripts/python.exe analysis/authenticate_tabpfn.py
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe analysis/compare_tabpfn.py --check-access
$env:TABPFN_MAX_BATCHED_ESTIMATOR_ROWS = $null
.venv/Scripts/python.exe analysis/compare_tabpfn.py --resume
.venv/Scripts/python.exe analysis/plot_tabpfn.py --input reports/tabpfn

# Larger-context comparison, with estimators evaluated sequentially.
$env:TABPFN_MAX_BATCHED_ESTIMATOR_ROWS = '0'
.venv/Scripts/python.exe analysis/compare_tabpfn.py --context-rows 50000 --batch-size 200 --output reports/tabpfn_50k --resume
.venv/Scripts/python.exe analysis/plot_tabpfn.py --input reports/tabpfn_50k
.venv/Scripts/python.exe analysis/summarize_tabpfn_contexts.py
```

The default uses four estimators, 20,000 context rows per estimator and cached
prediction in 500-row batches. The initial 2,000-row prediction batch exceeded
the RTX 4080's memory; a 500-row probe passed with the same context and estimators.
A context of `0` uses full history. Use a
different output directory for different settings, for example
`--context-rows 50000 --batch-size 200 --output reports/tabpfn_50k`. The larger
run uses `TABPFN_MAX_BATCHED_ESTIMATOR_ROWS=0` to limit simultaneous estimator
memory without changing the ensemble count. `--baseline-only` runs the
matched CatBoost controls without requiring TabPFN access. Completed folds can
be resumed only with matching settings, source/code hashes and prediction hashes.
Model weights are cached under the ignored `.cache/` directory. Authentication
uses the upstream package's user-local token cache; credentials are never
included in experiment artifacts.

If the browser login fails, open `https://platform.priorlabs.ai` directly,
sign in, accept the TabPFN-3.5 license under Licenses, and copy your API key.
The authentication helper prompts for the key with hidden input by default;
it does not use a localhost callback. `--api-key` explicitly selects this same
mode. Upstream TabPFN 9.2.0 uses `select.select()` on terminal input in its
browser flow, which is incompatible with Windows console handles; `--browser`
retains that optional flow for other environments.

Run the feature/head follow-up using the same installed dependencies:

```powershell
$env:OMP_NUM_THREADS = '1'
.venv/Scripts/python.exe analysis/compare_tabpfn_adaptation.py --smoke
.venv/Scripts/python.exe analysis/compare_tabpfn_adaptation.py --resume
.venv/Scripts/python.exe analysis/plot_tabpfn_adaptation.py
```

The follow-up caches query embeddings under ignored `.cache/tabpfn_adaptation/`.
It saves feature/head metrics, source-row predictions, fitted residual heads,
learning curves and context/query audits under `reports/tabpfn_adaptation/`.
Ridge and MLP each have numeric-only and embedding controls. The heads use
quarterly historical origins with 1/3/6/12-month horizons; the final three
pre-outer months select hyperparameters. Query rents never enter their labelled
TabPFN context, and outer labels never train or select a head.

Predict monthly Singapore HDB rent (SGD) from property attributes, location and rental approval time. The project uses Kaggle training, test and auxiliary data; competition evaluation uses root mean squared error (RMSE).

## Data and task

| Path | Purpose |
| --- | --- |
| `data/train.csv` | Training data labeled with `MONTHLY_RENT` |
| `data/test.csv` | Prediction inputs without rent labels |
| `data/example-submission.csv` | Kaggle submission template: `Id,Predicted` |
| `data/auxiliary/` | Building, MRT, mall, school, COE and stock data |

Raw CSVs remain unchanged. Cleaning and feature construction use in-memory copies; figures and metrics go to `reports/`.

## Project structure

```text
analysis/
  eda_train_test.py          Main-data EDA entry
  eda_auxiliary_data.py      Auxiliary-data EDA entry
  eda_temporal_correlations.py  Monthly rent/economic correlation figures
  train_baseline.py          CatBoost feature comparison entry
  compare_model_families.py  Model-family comparison entry
  compare_auxiliary_features.py  Incremental auxiliary-feature ablation entry
  compare_spatiotemporal_hybrid.py  Fixed-origin spatiotemporal hybrid evaluation/prediction entry
  config.py                  Paths, validation dates, feature groups and fixed parameters
  data.py                    Validation, feature construction, temporal splits and address normalization
  evaluation.py              RMSE, MAE, mean bias and grouped errors
  modeling.py                Model-input preprocessing and pipelines
  eda/                       Two EDA implementations
  experiments/               Three model-experiment implementations
data/                         Raw Kaggle data
reports/                      Rebuildable summaries, figures and experiment metrics
midterm_report.md             Midterm analysis, results and next steps
environment.yml              Standalone Conda environment definition
requirements-eda.txt         Main-data EDA dependencies
requirements-model.txt       Dependencies for all existing scripts
```

`analysis/` contains entry points and reusable modules. Two EDA workflows inspect train/test and auxiliary data separately; current modeling uses only main-data features.

## Installation and execution

Create a standalone environment from the project root:

```powershell
conda env create -f environment.yml
conda activate cs5228-hdb
```

With an existing Miniconda `cs5228` environment, install dependencies there:

```powershell
conda activate cs5228
python -m pip install -r requirements-model.txt
```

Run in order:

```powershell
python analysis/eda_train_test.py
python analysis/eda_auxiliary_data.py
python analysis/eda_temporal_correlations.py
python analysis/eda_economic_signals.py
python analysis/eda_group_temporal_panel.py
python analysis/train_baseline.py
python analysis/compare_model_families.py
python analysis/compare_auxiliary_features.py
python analysis/compare_spatiotemporal_hybrid.py
```

`compare_model_families.py` reads CatBoost results from `train_baseline.py`, so run the latter first. `compare_auxiliary_features.py` runs independently with a 2024-04 split, adding building, facility, COE, stock and interaction features in stages. Each run uses one model: LightGBM when available, otherwise scikit-learn HistGradientBoosting. Outputs under `reports/auxiliary_ablation/`: `ablation_metrics.csv`, `monthly_errors.csv`, `feature_coverage.csv`, `feature_importance.csv`, `ablation_rmse.png` and `results.json`. Small functional checks: `python -m unittest discover -s tests`.

`compare_spatiotemporal_hybrid.py` compares spatial baselines, historical-rent features, recursive spatiotemporal residual correction and their combination at multiple fixed origins. Validation labels never feed back as lagged features. Results and final test predictions go to `reports/spatiotemporal_hybrid/`.

`python analysis/compare_local_comparable_graph.py` evaluates a block-level nearby-comparable graph against the historical-rent baseline. It uses fixed-origin 6-, 9-, 12-, and 16-month validation windows and writes metrics, coverage, settings, and a concise report to `reports/local_comparable_graph/`. The final 16-month window overlaps the earlier windows and is evaluated separately.

`python analysis/compare_bayesian_amenity_factors.py` evaluates a Gaussian Bayesian additive model with separate MRT and mall distance curves, pooled town/subzone effects, a monthly random-walk effect, and optional rent histories. It compares four ablations with the historical tree baseline on the same forward windows. Posterior-mean RMSE, sampled 90% interval coverage, amenity contrasts, priors, and limitations are saved in `reports/bayesian_amenity_factors/`. It uses the existing SciPy dependency; PyMC is not required.

Economic features use values from 1, 3 and 6 calendar months before approval, suitable for monthly updates. Fixed-cutoff forecasts cannot treat subsequently released data as known. Facility tables are current snapshots without historical opening dates; interpret their features as geographic proxies only.

`eda_temporal_correlations.py` writes four plots (standardized levels, monthly changes, correlation matrix and lagged correlations), monthly data, coefficients and notes to `reports/temporal_correlations/`. Correlations use months before 2024-04 only; later months appear only in descriptive plots. Rent-composition adjustment uses flat type, flat model, Town, area and lease commencement year, excluding approval month.

## Current progress

First-round EDA of main and six auxiliary datasets and initial models without auxiliary features are complete. Validation forecasts forward by approval month; no final Kaggle submission is generated yet. See the [midterm report](midterm_report.md) and [detailed auxiliary report](reports/auxiliary_data_eda/report.md) for design, metrics, limitations and preprocessing recommendations.

## zch project workflow

## Where to start

| Your goal | Start here |
| --- | --- |
| Understand final methods and scores | [Final conclusions](experiment/01_final/README.md) |
| Read the full research process from baseline to final | [Full experiment report](experiment/01_final/EXPERIMENT_REPORT.md) |
| Understand why feature engineering works | [Feature engineering](experiment/02_feature_engineering/README.md) |
| Compare models | [Model comparison](experiment/03_model_comparison/README.md) |
| Inspect each exploration round and negative results | [Optimization history](experiment/04_optimization_history/README.md) |
| Run code | [Running guide](docs/guides/running_experiments.md) |
| Understand raw data and usage boundaries | [Data guide](data/README.md) |
| Check implementation and existing results | [Testing guide](tests/README.md) |
| Find saved models, parameters and metric sources | [Artifact catalog guide](docs/guides/running_experiments.md#baseline-features-and-model-comparisons) |

## Three final submissions

| Method | Local annual RMSE | Local recent RMSE | Displayed Kaggle score |
| --- | ---: | ---: | ---: |
| Main history-feature blend | 451.1810 | 442.9025 | 415.86912 |
| Provided macro-data augmentation | 451.2081 | 442.8892 | 415.78232 |
| **Grouped blend + PLR-FT (final)** | **451.0295** | **441.8345** | **415.37529** |

Local annual validation spans 2024-04 to 2025-03; recent validation spans 2024-06 to 2025-03. Windows overlap and have been used for development.
Kaggle values come from user screenshots without public/private labels. Final is selected among three submissions; wave nine still has lower annual RMSE.
See the [final report](experiment/01_final/README.md) for methods and information boundaries.

## Directory guide

```text
experiment/       Read: final conclusions, features, models, optimization and data analysis
analysis/      Implementation: shared pipeline and grouped experiment code
docs/          Usage: execution, layout, dependencies and maintenance history
data/          Raw competition data, unchanged
output/        Runtime artifacts: trials, models, row-level predictions and submissions
requirements/  Layered dependencies
tests/         Automated checks
```

`experiment/` retains summaries and reading materials. Full trial configurations, metrics and selection records are in
`output/experiment_runs/`, models in `output/models/`, final submission at
`output/submissions/final/submission.csv` and candidates in `output/submissions/candidates/`.
Runtime artifacts are not distributed through Git. See the [layout guide](docs/guides/project_structure.md).

## Environment and common commands

Run from the repository root:

```powershell
conda env create -f environment.yml
conda activate cs5228
python -m analysis.main --help

# Inspect entry points without training
python -m analysis.main experiment-list --group wave10

# Check environment/resources without installation or training
python -m analysis.main preflight --profile final

# Refresh reading reports from existing results without training
python -m analysis.main experiment-refresh
```

For an index-only standard-library entry, use `python -m analysis.experiment_catalog --group final --json`.
GPU and author-model dependencies: [requirements](requirements/README.md) and [running guide](docs/guides/running_experiments.md).

All three final methods use competition training labels and supplied auxiliary data. Test tables are query inputs; true test rents are not used.
Public pretrained weights in other explorations are identified per wave. Directory cleanup does not change raw data or prediction values.
