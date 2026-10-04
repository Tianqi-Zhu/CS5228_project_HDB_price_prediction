# CS5228 HDB 月租预测

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
