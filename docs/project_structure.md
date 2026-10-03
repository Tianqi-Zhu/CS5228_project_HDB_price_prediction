# 项目结构说明

项目只保留一个 Python 包 `analysis/`。数据处理、EDA、实验、训练和预测从同一套
模块导入，不再维护 `src/` 与 `analysis/` 两套相互转发的目录。

## 最终结构

```text
analysis/
├── main.py                 # 唯一 CLI
├── pipeline/
│   ├── config.py           # 路径、特征组和固定模型参数
│   ├── data.py             # 读取、清理、特征构造、时间划分和源文件保护
│   ├── models.py           # 四种模型及训练期预处理管线
│   ├── training.py         # 训练器和模型保存／加载
│   ├── workflow.py         # 验证、重训、提交文件和保存模型预测
│   ├── evaluation.py       # 指标、分组误差、评估接口和误差绘图
│   └── spatial.py          # 设施筛选与球面距离
├── experiments/
│   ├── baseline.py         # 均值与 CatBoost A/B/C 特征比较
│   └── compare_models.py   # Ridge、RF、CatBoost、LightGBM 比较
├── eda/
│   ├── run.py              # EDA 唯一入口
│   ├── overview.py         # 综合 EDA：11 图、18 表和摘要
│   ├── pdf_report.py       # 15 页阶段 PDF
│   └── audits/
│       ├── main_data.py    # train/test 专项审计和 5 张补充图
│       └── auxiliary.py    # 六份辅助数据审计和 3 张补充图
```

合并后的职责关系：

- 原先分开的数据读取和 `HDBPreprocessor` 合并到 `pipeline/data.py`。
- 指标、`HDBEvaluator`、通用图和模型误差图合并到 `pipeline/evaluation.py`。
- `trainer.py` 更名为 `training.py`，与执行完整任务的 `workflow.py` 清晰区分。
- EDA 的含义不清文件名改为 `overview.py`、`pdf_report.py`；两类专项分析放进
  `eda/audits/`。
- `model_families.py` 改为 `compare_models.py`，文件名直接表达用途。

## 数据与模型流程

```text
原始 CSV
  → pipeline/data.py 在副本中清理并构造确定性特征
  → 按审批月份划分拟合集和验证集
  → pipeline/models.py 只在拟合集学习类别编码、补值和模型
  → pipeline/evaluation.py 报告整体及分组误差
  → 人工选定方案
  → pipeline/workflow.py 使用有标签数据重训
  → 按原始 test 行序预测并套用提交模板
```

默认时间验证为 2021-01 至 2024-03 拟合、2024-04 至 2025-03 验证。
`FLAT_AGE_YEARS` 表示审批年份减租约起始年份，不等于辅助楼栋表完工年份定义的
实际楼龄；负值设为缺失，不修改原始 `LEASE_COMMENCE_DATE`。

## EDA 统计口径

- `overview.py` 按租赁记录统计设施距离，用于描述距离与租金关系；小学只保留
  `MAINLEVEL_CODE` 以 `primary` 开头的记录。
- `audits/auxiliary.py` 按唯一楼栋统计设施覆盖；小学还包含 `p1` 混合学段。

所以历史摘要中的 177 与 180 个小学地点以及距离分布不应互换。中期正文使用综合
口径，逐文件质量和楼栋覆盖使用补充报告。

## 运行和输出

```powershell
python -m analysis.main eda
python -m analysis.main baseline
python -m analysis.main compare
python -m analysis.main validate --model lightgbm --features C
python -m analysis.main report
```

| 命令 | 默认输出 |
| --- | --- |
| `eda` | `reports/eda/` 全部 EDA |
| `eda --scope overview` | `reports/eda/` 主报告图表 |
| `eda --scope main` | `reports/eda/supplementary/train_test/` |
| `eda --scope auxiliary` | `reports/eda/supplementary/auxiliary/` |
| `baseline` | `reports/model_baseline/` |
| `compare` | `reports/model_families/` |
| `validate` | `reports/validation/<模型_特征_切点_窗口>/` |
| `report` | `reports/eda/progress_report.pdf` |
| `submit` | `output/submissions/` 和 `models/` |

辅助特征尚未进入已报告的四模型实验。设施开放日期及宏观数据的预测时点仍需在后续
特征工程中审查。
