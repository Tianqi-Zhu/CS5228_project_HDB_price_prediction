# CS5228 HDB 月租预测

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
prepare equal-weight ABCDE and weighted ABDE using all150,000 training rows and
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

根据新加坡 HDB 组屋的属性、位置和租赁审批时间预测每月租金（SGD）。项目使用 Kaggle 提供的训练集、测试集与辅助数据；比赛评价指标为均方根误差（RMSE）。

## 数据与任务

| 路径 | 用途 |
| --- | --- |
| `data/train.csv` | 带有 `MONTHLY_RENT` 标签的训练数据 |
| `data/test.csv` | 待预测数据，没有租金标签 |
| `data/example-submission.csv` | Kaggle 提交格式示例：`Id,Predicted` |
| `data/auxiliary/` | 楼栋、MRT、商场、学校、COE 和股票数据 |

所有原始 CSV 保持不变。数据清理与特征构造在内存中的副本上完成，图表和指标写入 `reports/`。

## 项目结构

```text
analysis/
  eda_train_test.py          主数据 EDA 运行入口
  eda_auxiliary_data.py      辅助数据 EDA 运行入口
  eda_temporal_correlations.py  租金与经济指标的月度相关性图表
  train_baseline.py          CatBoost 特征比较运行入口
  compare_model_families.py  模型家族比较运行入口
  compare_auxiliary_features.py  辅助特征逐步消融实验入口
  compare_spatiotemporal_hybrid.py  固定起点时空混合模型评估与预测入口
  config.py                  路径、验证时间点、特征组和固定模型参数
  data.py                    数据校验、特征构造、时间划分和地址规范化
  evaluation.py              RMSE、MAE、平均偏差及分组误差
  modeling.py                模型输入预处理与模型管线
  eda/                       两套 EDA 的实现
  experiments/               三组模型实验的实现
data/                         Kaggle 原始数据
reports/                      可重建的统计摘要、图表和实验指标
midterm_report.md             中期分析、实验结果与后续计划
environment.yml              独立 Conda 环境定义
requirements-eda.txt         主数据 EDA 所需依赖
requirements-model.txt       全部现有脚本所需依赖
```

`analysis/` 包含运行入口和可复用模块。两套 EDA 分别检查 train/test 和辅助数据，模型实验目前只使用主数据特征。

## 安装与运行

在项目根目录创建独立环境：

```powershell
conda env create -f environment.yml
conda activate cs5228-hdb
```

如果已有 Miniconda `cs5228` 环境，也可以在该环境中安装依赖：

```powershell
conda activate cs5228
python -m pip install -r requirements-model.txt
```

按顺序运行：

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

`compare_model_families.py` 读取 `train_baseline.py` 的 CatBoost 结果，因此要先运行后者。`compare_auxiliary_features.py` 可独立运行，采用 2024-04 时间切分，依次加入楼栋、周边设施、COE、股票和交互特征。每次运行固定使用同一种模型：环境中有 LightGBM 时使用 LightGBM，否则使用 scikit-learn 的 HistGradientBoosting。输出写入 `reports/auxiliary_ablation/`：`ablation_metrics.csv`、`monthly_errors.csv`、`feature_coverage.csv`、`feature_importance.csv`、`ablation_rmse.png` 和 `results.json`。运行小型功能检查：`python -m unittest discover -s tests`。

`compare_spatiotemporal_hybrid.py` 使用多个固定预测起点比较空间基线、仅使用历史租金特征、递归时空残差校正及两者组合。验证期标签不会作为滞后特征反馈给模型。结果和最终测试集预测写入 `reports/spatiotemporal_hybrid/`。

`python analysis/compare_local_comparable_graph.py` evaluates a block-level nearby-comparable graph against the historical-rent baseline. It uses fixed-origin 6-, 9-, 12-, and 16-month validation windows and writes metrics, coverage, settings, and a concise report to `reports/local_comparable_graph/`. The final 16-month window overlaps the earlier windows and is evaluated separately.

`python analysis/compare_bayesian_amenity_factors.py` evaluates a Gaussian Bayesian additive model with separate MRT and mall distance curves, pooled town/subzone effects, a monthly random-walk effect, and optional rent histories. It compares four ablations with the historical tree baseline on the same forward windows. Posterior-mean RMSE, sampled 90% interval coverage, amenity contrasts, priors, and limitations are saved in `reports/bayesian_amenity_factors/`. It uses the existing SciPy dependency; PyMC is not required.

经济特征采用审批月之前 1、3、6 个日历月的数值，适用于逐月更新预测的情景；若在训练截止月一次性预测以后所有月份，不能把之后公布的经济数据当作已知。设施表是当前快照，缺少历史开通日期，因此相关特征仅作地理代理变量解释。

`eda_temporal_correlations.py` 在 `reports/temporal_correlations/` 输出标准化水平、月度变化、相关矩阵和滞后相关性四张图，以及月度数据、相关系数和分析说明。相关系数仅使用 2024-04 之前的月份；后续月份只用于描述性时间序列图。租金组合调整模型只使用房型、房屋模型、Town、面积和租约开始年份，不使用审批月份。

## 当前进度

已完成主数据和六份辅助数据的第一轮 EDA，以及尚未加入辅助特征的初步模型比较。验证采用按审批月份向前预测的划分；目前没有生成最终的 Kaggle 提交文件。实验设计、指标、局限和辅助数据预处理建议见 [中期实验报告](midterm_report.md)与[辅助数据详细报告](reports/auxiliary_data_eda/report.md)。

## zch project workflow

## 先看什么

| 你的目的 | 从这里进入 |
| --- | --- |
| 了解最终方法与成绩 | [最终结论](experiment/01_final/README.md) |
| 从基础到final阅读完整研究过程 | [完整实验报告](experiment/01_final/EXPERIMENT_REPORT.md) |
| 理解特征工程为什么有效 | [特征工程](experiment/02_feature_engineering/README.md) |
| 比较不同模型 | [模型对比](experiment/03_model_comparison/README.md) |
| 查看每轮探索及负结果 | [优化历程](experiment/04_optimization_history/README.md) |
| 运行代码 | [运行指南](docs/guides/running_experiments.md) |
| 了解原始数据及使用边界 | [数据说明](data/README.md) |
| 检查实现与既有结果 | [测试说明](tests/README.md) |
| 查看已保存模型、参数及指标来源 | [产物清单用法](docs/guides/running_experiments.md#基础特征与模型对照) |

## 最终三份提交

| 方法 | 本地年度RMSE | 本地近期RMSE | Kaggle显示分数 |
| --- | ---: | ---: | ---: |
| 历史特征主融合 | 451.1810 | 442.9025 | 415.86912 |
| 所给宏观增强 | 451.2081 | 442.8892 | 415.78232 |
| **分组融合＋PLR-FT（final）** | **451.0295** | **441.8345** | **415.37529** |

本地年度为2024-04至2025-03，近期为2024-06至2025-03；窗口重叠且已用于开发。
Kaggle数值来自用户截图，未标注公榜／私榜。final是三份提交中的选择，第九轮的年度分数仍更低。
最终方法和信息边界见[最终报告](experiment/01_final/README.md)。

## 目录怎么读

```text
experiment/       给人读：最终结论、特征、模型、优化历程、数据分析
analysis/      看实现：共享管线与按用途分组的实验代码
docs/          查用法：运行、目录、依赖说明与历史维护
data/          原始比赛数据，保持不变
output/        运行产物：试验明细、模型、逐行预测、提交文件
requirements/  分层依赖
tests/         自动检查
```

`experiment/`只保留汇总与阅读材料。完整试验配置、逐次指标与选择记录在
`output/experiment_runs/`，模型在 `output/models/`，最终提交在
`output/submissions/final/submission.csv`，候选在 `output/submissions/candidates/`。
这些运行产物不随Git分发。详细目录见[结构说明](docs/guides/project_structure.md)。

## 环境与常用命令

在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda activate cs5228
python -m analysis.main --help

# 只查看入口，不训练
python -m analysis.main experiment-list --group wave10

# 只检查环境与资源，不安装、不训练
python -m analysis.main preflight --profile final

# 从已有结果更新阅读报告，不训练
python -m analysis.main experiment-refresh
```

仅查看索引可用标准库入口 `python -m analysis.experiment_catalog --group final --json`。
GPU及额外作者模型依赖见[requirements说明](requirements/README.md)和[运行指南](docs/guides/running_experiments.md)。

最终三份方案使用比赛训练标签与所给辅助数据；测试表作为查询输入，不使用其真实租金。
其他探索中的公开预训练权重在相应轮次注明。原始数据和预测数值未因目录整理而改变。
