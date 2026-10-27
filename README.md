# CS5228 HDB 月租预测

The [revised dynamic rental surface experiment plan](EXPERIMENT_PLAN_DYNAMIC_SURFACE.md) prioritizes a stronger historical row model and a coarse market-index test before any grid/PCA hybrid.

The completed [full-history tree, market-index, and coarse-surface experiment](reports/dynamic_market_index/report.md) selects CatBoost without a market correction. Its strict February 2025-origin candidate submission is in `reports/dynamic_market_index/submission_strict_feb2025.csv`.

The [factored dynamic rental-surface experiment](reports/factored_rental_surface/report.md) evaluates monthly smoothed surfaces, PCA factors, fixed-origin temporal forecasts, and direct/residual CatBoost hybrids. The residual hybrid improves pooled RMSE by S$1.23 but is less consistent across folds, so the selected candidate remains full-history CatBoost.

## TabPFN experiment

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
