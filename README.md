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
  train_baseline.py          CatBoost 特征比较运行入口
  compare_model_families.py  模型家族比较运行入口
  config.py                  路径、验证时间点、特征组和固定模型参数
  data.py                    数据校验、特征构造、时间划分和地址规范化
  evaluation.py              RMSE、MAE、平均偏差及分组误差
  modeling.py                模型输入预处理与模型管线
  eda/                       两套 EDA 的实现
  experiments/               两组模型实验的实现
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
python analysis/train_baseline.py
python analysis/compare_model_families.py
```

最后一个脚本读取前一个模型脚本生成的 CatBoost 结果，因此要先运行 `train_baseline.py`。各脚本只把结果写到对应的 `reports/` 子目录。

## 当前进度

已完成主数据和六份辅助数据的第一轮 EDA，以及尚未加入辅助特征的初步模型比较。验证采用按审批月份向前预测的划分；目前没有生成最终的 Kaggle 提交文件。实验设计、指标、局限和辅助数据预处理建议见 [中期实验报告](midterm_report.md)与[辅助数据详细报告](reports/auxiliary_data_eda/report.md)。
