# CS5228 项目 (Team 30)：新加坡 HDB 月租预测

> **Beyond Property Attributes: Chronological Local-History Aggregations and Hybrid Tree–Transformer Ensembles for Singapore HDB Rent Forecasting**
> ([English default version](README.md) · [文档多语言说明](docs/documentation-languages.md))

本仓库包含 **Team 30** 在 CS5228 课程项目（预测新加坡 HDB 组屋月租金 `MONTHLY_RENT`，单位 SGD）中的完整可复现代码、特征工程流水线、各模型家族实现与冻结验证证据。

---

## 1. 项目概览与核心实验结果

我们在 **150,000 条训练记录** 与 **50,000 条测试记录**（`2021-01` 至 `2025-03`）及 6 张空间与宏观辅助表上构建预测模型。针对 2021 年后显著的租金通胀趋势（2021 至 2024 年涨幅 +44.6%），所有实验严格遵循**固定时间原点（Rolling-Origin）防泄漏验证协议**：
- **年度窗口 ($Y$)**：在 `2021-01`–`2024-03`（114,842 行）上拟合，在 `2024-04`–`2025-03`（35,158 行）上验证。
- **近期窗口 ($J$)**：在 `2021-01`–`2024-05`（121,005 行）上拟合，在 `2024-06`–`2025-03`（28,995 行）上验证。

| 模型流水线 / 架构 | 特征方案 | 年度窗 ($Y$) RMSE | 近期窗 ($J$) RMSE | Kaggle Public Score | 复现入口文档 |
|---|---|---:|---:|---:|---|
| **F1：12 路历史支撑分组树 + PLR-FT 融合** | 嵌套 `19 / 27 / 47 / 62` 阶梯 | **451.03** | **441.83** | **415.38** | [reproduction/f1/README.md](reproduction/f1/README.md) |
| **F2：7 树 + TabPFN v2.5 等权融合**（原始 / 归并房型） | `62` 树特征 + `87` TabPFN 特征 | — | **440.95 / 440.77** | **413.20** | [docs/guides/tabpfn_ensemble.md](docs/guides/tabpfn_ensemble.md) |
| **独立 TabPFN v2.5（87 特征）** | `77` 基础 + `10` 周年特征（`87`） | — | **441.96** | — | [docs/guides/tabpfn_best.md](docs/guides/tabpfn_best.md) |
| **Concat Transformer（可学习历史摘要分词器）** | `60` 基础 + `35` 历史（`95`） | **459.22** | — | **418.39** | [reproduction/concat_transformer/README.md](reproduction/concat_transformer/README.md) |
| **独立 CPU MLP 集成**（清洗 / 未清洗） | `44` 列表格流水线 | **469.92** | — | **437.34 / 439.38** | [reproduction/mlp/README.md](reproduction/mlp/README.md) |

---

## 2. 快速上手：60 秒纯 CPU 离线验证与结果复现

外部审阅人无需 GPU 或联网，即可在任意 CPU 机器上校验全部 4,367 个文件的 SHA-256 完整性，并复现报告中的验证指标与 50,000 行测试集融合预测。

### 第一步：配置环境
```bash
# 方案 A：轻量级 CPU 离线核验与复现环境（NumPy, pandas, matplotlib）
conda env create -f environment.review.yml
conda activate cs5228-review

# 方案 B：完整模型训练环境（LightGBM, CatBoost, XGBoost, scikit-learn, PyTorch）
conda env create -f environment.training.yml
conda activate cs5228-training
```

### 第二步：一键运行离线核验与复现套件
以下命令全部在 CPU 上确定性运行，新生成产物统一写入被 Git 忽略的 `output/` 目录：

```bash
# 1. 核验全仓代码、原始数据集与实验证据的 SHA-256 哈希及导航链接
python review.py verify
python -m analysis.validation.repository_audit

# 2. 从 12 路基础预测重组 50,000 行 F1 提交 CSV（最大绝对误差约 1.8e-12 SGD）
python reproduction/f1/verify.py
python reproduction/f1/reproduce.py

# 3. 核验 Y/J 两窗共 30 个全新重训的 F1 基础模型与原点前融合重放
python -m analysis.main validation-refits

# 4. 重组 50,000 行 F2 提交并在 J 窗口（28,995 行）上执行 F1 vs. F2 配对比较
python -m analysis.main final-pair

# 5. 核验有序特征清单（19/27/47/62/87/95）及 M01 空间与清洗证据
python -m analysis.main schemas
python -m analysis.main evidence

# 6. 核验独立神经网络与 TabPFN 复现包（Concat Transformer、CPU MLP、TabPFN）
python -m analysis.main concat verify
python -m analysis.main mlp verify
python review_tabpfn.py verify

# 7. 运行覆盖全部实验阶段（M00–M09）的端到端离线复现审计
python -m analysis.main planned
```

---

## 3. 报告章节与源码/复现命令对照表

| 报告章节 | 主题与方法内容 | 源码目录位置 | 复现 / 重训命令 |
|---|---|---|---|
| **Section II** | **探索性数据分析（EDA）、数据清洗与空间辅助表**<br>文本规范化、重复行策略、仅训练集异常低租筛查（`M01`）、Haversine `BallTree` 距离（`0.5/1.0/2.0 km`）、COE 与股票宏观指标 | [`analysis/eda/`](analysis/eda/)<br>[`analysis/pipeline/data.py`](analysis/pipeline/data.py)<br>[`analysis/pipeline/auxiliary.py`](analysis/pipeline/auxiliary.py) | `python -m analysis.main eda --scope all --output-dir output/review_eda`<br>`python -m analysis.main evidence` |
| **Section III** | **时间验证协议与传统基线**<br>时间切分窗口 $Y$ & $J$ vs. 随机 K 折泄漏（`431.78` vs. `484.33`）、`C7` 传统基线（Mean, Ridge, RF, LightGBM, CatBoost；`M02`） | [`analysis/experiments/baselines/`](analysis/experiments/baselines/)<br>[`docs/methods/prediction-scenario.md`](docs/methods/prediction-scenario.md) | `python -m analysis.main baseline --output-dir output/review_baseline`<br>`python -m analysis.main compare --baseline-dir output/review_baseline --output-dir output/review_compare` |
| **Section IV** | **嵌套特征工程阶梯（`19→27→47→62`）**<br>因果历史均价层级（`History27`）、空间/宏观特征（`Aux47`）、租约周年与核回归同群特征（`Full62`）、TabPFN `77→87` 特征（`M03`） | [`analysis/pipeline/features.py`](analysis/pipeline/features.py)<br>[`analysis/experiments/feature_engineering/`](analysis/experiments/feature_engineering/)<br>[`experiment/main/M03/feature_schemas/README.md`](experiment/main/M03/feature_schemas/README.md) | `python -m analysis.main schemas`<br>`python -m analysis.main ladder --stage outer` |
| **Section V** | **树模型集成、深度表格网络与 TabPFN**<br>27 特征同输入对比（`LightGBM`, `CatBoost`, `XGBoost`, `MLP`, `FT-Transformer`, `TabM`）、`PLR-FT` vs. `PLE-FT`、`Concat Transformer`（`M05`）、`TabPFN v2.5`（`M07`） | [`analysis/experiments/model_comparison/`](analysis/experiments/model_comparison/)<br>[`reproduction/concat_transformer/README.md`](reproduction/concat_transformer/README.md)<br>[`reproduction/mlp/README.md`](reproduction/mlp/README.md)<br>[`workflows/tabpfn/README.md`](workflows/tabpfn/README.md) | `python -m analysis.main common-models --device cuda --output-dir output/review_common`<br>`python -m analysis.main concat predict --device cpu`<br>`python -m analysis.main mlp fullfit --output-dir output/mlp/new-run`<br>`python run_tabpfn_best.py preflight` |
| **Section VI** | **最终集成方案 F1 与 F2**<br>• **F1**：基于 `ANN_BLOCK_24/36` `2×2` 历史支撑状态的 12 路分组融合<br>• **F2**：7 树 + TabPFN87 的 `0.5 / 0.5` 等权混合集成（`M04`, `M08`, `M09`） | [`reproduction/f1/`](reproduction/f1/)<br>[`run_tabpfn_ensemble.py`](run_tabpfn_ensemble.py)<br>[`analysis/validation/final_pair.py`](analysis/validation/final_pair.py) | **CPU 重放**：`python reproduction/f1/reproduce.py` & `python -m analysis.main final-pair`<br>**GPU 全量重训**：`python reproduction/f1/validate.py --window Y --output-dir output/f1-Y` & `python reproduction/f1/train.py` |
| **Section VII** | **事后误差分解与诊断分析**<br>历史支撑状态分组误差、高租金分位段尾部偏差与稀疏楼栋同群样本分析 | [`analysis/validation/main.py`](analysis/validation/main.py)<br>[`experiment/main/README.md`](experiment/main/README.md) | `python -m analysis.main planned` |

---

## 4. 仓库目录结构与详细文档导航

| 主题 | 文档链接 |
|---|---|
| 完整命令行说明与全量训练指南 | [运行与复现指南](docs/guides/running_experiments.md) |
| 目录职责与中间产物清理规范 | [仓库结构说明](docs/guides/repository_layout.md) |
| `analysis/` 共享核心包结构 | [Analysis 模块说明](analysis/README.md) |
| 实验阶段索引与冻结证据（`M00`–`M09`） | [实验索引 (`M00`–`M09`)](experiment/main/README.md) |
| 验证执行状态与五项核验记录 | [执行状态](docs/planning/main-execution-status.md) · [五项核验记录](docs/planning/main-five-item-completion.md) |
| 时间切分边界与预测场景定义 | [预测场景定义](docs/methods/prediction-scenario.md) |
| 主线与附录实验统一规划 | [统一实验规划](docs/planning/main-appendix-experiment-plan.md) |
| F2 TabPFN + 七树集成指南与来源 | [F2 集成指南](docs/guides/tabpfn_ensemble.md) · [M07 状态](experiment/main/M07/README.md) · [M08 配对结果](experiment/main/M08/README.md) · [TabPFN 合并说明](docs/guides/anniversary_merge.md) |
| 核验范围与日志记录 | [核验日志 (`review/CHECKS.md`)](review/CHECKS.md) |
| 原始数据集与辅助表说明 | [数据目录说明](data/README.md) |
| 单元与集成测试说明 | [测试目录说明](tests/README.md) |
| 方法专题文档（基线、历史特征、模型对比、MLP） | [基线特征](docs/methods/baseline_features.md) · [历史特征优化](docs/methods/history_feature_optimization.md) · [模型对比](docs/methods/model_comparison.md) · [MLP 报告](docs/mlp_prediction_report.md) |
