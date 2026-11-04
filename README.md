# CS5228 HDB 月租预测：最终融合与三份提交对照

本分支为 `results/hdb-rental-final-ensemble`。三份提交的方法、分数、环境与复现入口从 [TEAM_REVIEW.md](TEAM_REVIEW.md) 开始；内层README讲实现原理，RUNNING.md讲执行方法。后文保留基础项目说明。

根据新加坡 HDB 组屋的房屋属性、位置和租赁审批月份预测月租金（SGD）。项目使用比赛提供的训练集、测试集和六份辅助数据，主要评价指标为 RMSE。

项目代码统一放在 `analysis/`：包括 EDA、数据处理、传统机器学习实验、时间验证和预测文件生成。项目支持特征消融、Ridge、Random Forest、CatBoost、LightGBM 比较、模型保存及 Kaggle 格式预测文件生成。

## 数据

| 路径 | 用途 |
| --- | --- |
| `data/train.csv` | 150,000 条带 `MONTHLY_RENT` 标签的记录 |
| `data/test.csv` | 50,000 条待预测记录，无租金标签 |
| `data/example-submission.csv` | `Id,Predicted` 提交格式及 ID 顺序 |
| `data/auxiliary/` | 楼栋、MRT、商场、学校、COE、股票数据 |

所有预处理在数据副本上进行，保留重复记录和测试行顺序；训练、预测和综合 EDA 会核对源文件哈希。命令行不允许把输出写入 `data/`。训练数据止于 2025-03，测试数据延伸至 2026-07，因此正式验证默认按时间划分。

## 项目结构

```text
analysis/
  main.py                    唯一命令行入口
  pipeline/                  共享数据与建模管线
    config.py                路径、特征组、固定模型参数
    data.py                  读取、清理、特征构造、时间划分、源文件保护
    models.py                四种模型及训练期预处理管线
    training.py              共用训练器和模型保存／加载
    workflow.py              时间验证、全量重训和提交文件
    evaluation.py            指标、分组误差和模型误差绘图
    spatial.py               设施筛选与球面距离计算
  eda/
    run.py                   EDA 统一入口
    overview.py              综合 EDA：11 图、18 表和摘要
    pdf_report.py            15 页阶段 PDF
    audits/main_data.py      train/test 专项审计
    audits/auxiliary.py      六份辅助数据专项审计
  studies/
    economic_signals.py      先行经济信号分析
    temporal_correlations.py 时间序列相关与滞后分析
    group_temporal_panel.py  分组时间面板
    spatial/                 空间描述、固定效应和匹配分析
  experiments/
    baseline.py              CatBoost A/B/C 特征比较
    compare_models.py        四模型同切分比较
    deep_learning.py         GPU 类别嵌入与特征 Transformer
    auxiliary_boosting.py    同辅助特征的 LightGBM 对照
    auxiliary_ablation.py    空间与宏观辅助特征逐组消融
    spatiotemporal_hybrid.py  只使用预测时点之前记录的历史空间模型
    neural_tabm.py           TabM 风格神经网络及树模型融合
    relu_mlp.py              嵌套时间验证的 sklearn ReLU MLP
data/                        原始数据
reports/eda/                 核心 EDA、专项审计和阶段 PDF
reports/studies/             解释性时间与空间研究
reports/experiments/         预测实验指标、图表和精简历史结果
models/                      本地模型产物（.joblib 不提交 Git）
output/submissions/          本地预测文件（不提交 Git）
tests/                       数据完整性和管线回归检查
docs/project_structure.md    合并对应关系与工作流程
midterm_report.md            中期分析、已跑结果和后续实验计划
environment.yml              完整 Conda 环境
requirements.txt             完整 pip 依赖入口
requirements-deep.txt        可选的 CUDA 深度学习依赖
```

## 环境

首次创建完整的项目环境（环境名由 `environment.yml` 定义为 `cs5228`）：

```powershell
conda env create -f environment.yml
conda activate cs5228
```

已有 `cs5228` 环境时，建议通过 YAML 同步全部依赖：

```powershell
conda env update -n cs5228 -f environment.yml --prune
conda activate cs5228
```

也可以只使用 pip：激活目标环境后运行 `python -m pip install -r requirements.txt`。

依赖文件按用途逐层包含：`requirements-eda.txt` 提供数据分析和预处理依赖；`requirements-model.txt` 在其基础上增加建模依赖；`requirements.txt` 是完整入口，再增加 PDF 报告和 Excel 读取依赖。使用 Conda 时优先以 `environment.yml` 为准，无需另建 venv。

GPU 深度学习实验另需安装 `python -m pip install -r requirements-deep.txt`。该文件使用 PyTorch CUDA 12.8 wheel；没有兼容 NVIDIA GPU 时应改装适合本机平台的 PyTorch 版本。

## 统一运行方式

在仓库根目录运行。所有功能都通过 `python -m analysis.main` 进入；不带子命令会显示帮助。

```powershell
python -m analysis.main --help

# 全部 EDA，也可 --scope main / auxiliary / overview
python -m analysis.main eda

# EDA 的直接入口，输出结构与上一条相同
python -m analysis.eda.run

# 重建原有 CatBoost A/B/C 特征实验和四模型比较
python -m analysis.main baseline
python -m analysis.main compare

# 统一时间验证：默认 2024-04 起验证，C 特征，LightGBM
python -m analysis.main validate --model lightgbm --features C

# 比较近期训练窗口，输出到独立目录
python -m analysis.main validate --model catboost --features C --fit-window-months 12

# 重建综合 EDA 的阶段 PDF（不包含全部 ML 实验）
python -m analysis.main report

# 四类互补空间分析共用一次内存特征构建，不生成 processed CSV
python -m analysis.main spatial-controls

# 时间、经济和历史空间分析
python -m analysis.main aux-ablation
python -m analysis.main economic-signals
python -m analysis.main economic-ablation
python -m analysis.main group-temporal
python -m analysis.main spatiotemporal

# 两条独立神经网络路线
python -m analysis.main tabm
python -m analysis.main relu

# 检查环境、源数据哈希、必需报告和跨报告指标一致性
python -m analysis.main verify
```

`compare` 若没有 CatBoost 结果会先生成；已有结果时会检查源文件、特征组和验证窗口是否一致。`validate` 记录模型配置、版本、源文件哈希、RMSE/MAE/偏差，以及按月份、Town、户型分组的误差。

一次完成 EDA、两组历史模型实验和 PDF，另存结果以保留历史快照：

```powershell
python -m analysis.main all --run-dir reports/experiments/runs/run01
```

不指定 `--run-dir` 会重建对应默认输出目录。`all` 默认不会生成提交文件；增加 `--submit` 可在上述步骤之后重训并生成本地预测。

## 辅助特征与深度学习实验

深度实验将 HDB 楼栋属性、经纬度、MRT／商场／小学距离和密度加入主表，并使用只从拟合集计算的留一目标编码。固定时间验证仍为 2024-04 至 2025-03，不使用测试标签或验证期后的经济数据。

当前两个特征 Transformer 与一个类别嵌入 ResNet 的等权集成 RMSE 为 **469.98**；加入辅助 LightGBM 后为 469.18。进一步使用只依赖过去月份的楼栋历史先验训练残差 Transformer，并按固定权重融合后达到 **466.47**（MAE 341.14），相比原主表 LightGBM 的 485.09 下降 18.62（3.84%）。由于模型选择反复使用同一个验证窗口，最终选择前还应增加滚动时间验证。完整配置、逐模型指标和月度误差见 [`reports/experiments/deep_learning/`](reports/experiments/deep_learning/README.md)。

```powershell
python -m analysis.main deep-summary
```

## 最终训练与预测

确认模型和特征后，使用全部有标签数据重新训练，再生成完整测试集预测：

```powershell
python -m analysis.main submit --model lightgbm --features C

# 使用已保存模型重复预测，不重新训练
python -m analysis.main predict --model-path models/lightgbm_C_property_time_town.joblib
```

`submit` **只写文件，不上传 Kaggle**。它按照模板保留 `Id` 和行顺序，校验行数和预测值，保存模型与训练元信息。默认使用全部训练记录；指定 `--fit-window-months` 时，只使用最后 N 个训练月份。验证时拟合的模型与最终重训模型分开使用。

## 已有成果与边界

已经保留主数据和六份辅助数据的 EDA、设施距离的固定效应与重叠分层分析、近期中位数基线，以及 Ridge、Random Forest、CatBoost、LightGBM、深度网络、TabM 风格网络和 ReLU MLP。辅助楼栋、空间与历史租金特征已进入独立监督学习实验；同月宏观特征未进入最终组合，设施开放日期仍需按预测时点审查。

已有数字、验证区间、局限和后续实验见 [中期报告](midterm_report.md)及 [EDA 主报告](reports/eda/report.md)。逐文件辅助数据审计位于 [补充报告](reports/eda/supplementary/auxiliary/report.md)，目录和统计口径见 [EDA 目录说明](reports/eda/README.md)。[阶段 PDF](reports/eda/progress_report.pdf) 展示综合 EDA，完整 ML 实验以中期报告为准。保存的结果均为本地验证分数，并不保证参数最优或代表 Kaggle 测试成绩。

## 结构与检查

代码职责、EDA 文件命名和两套统计口径详见 [项目结构说明](docs/project_structure.md)。

整合前后的指标、数据完整性及完整预测流程核验记录见 [整合核验](docs/integration_verification.md)。

五个工作分支的来源映射、保留/排除内容和指标可比性见 [分支整合说明](docs/branch_integration.md)。

报告总索引见 [reports/README.md](reports/README.md)，跨实验结果差异和原因见 [结果一致性检查](docs/result_consistency.md)，当前 `cs5228` 环境的实际运行清单见 [运行核验](docs/runtime_verification.md)。

```powershell
python -m unittest discover -s tests -v
python -m analysis.main verify
```
