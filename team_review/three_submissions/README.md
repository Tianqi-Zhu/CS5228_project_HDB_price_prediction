# 三份提交：分数、方法与运行入口

## 先看结果

| 原始提交文件 | 年度本地RMSE | 近期本地RMSE | Kaggle显示分数 | 用途 |
|---|---:|---:|---:|---|
| [历史版](submissions/kaggle_best_history.csv) | 451.1810 | 442.9025 | 415.86912 | 强基线 |
| [宏观版](submissions/kaggle_best_macro.csv) | 451.2081 | 442.8892 | 415.78232 | 宏观信息消融 |
| [结构融合版](submissions/kaggle_structure_blend.csv) | 451.0295 | 441.8345 | **415.37529** | **推荐final** |

Kaggle分数由用户提供的提交列表截图读取；截图未标明public/private类型。三份均为50,000行、`Id,Predicted`格式，原文件哈希见[manifest.json](manifest.json)。结构融合版比历史版低0.49383，比宏观版低0.40703；这只是三份候选的比较，不代表独立统计显著性结论。

年度窗口为2024-04至2025-03、35,158行；近期窗口为2024-06至2025-03、28,995行，二者重叠且反复用于开发。本地分数与Kaggle分数分别报告，不采用固定减20的换算。

## 三种运行方式

从本分支仓库根目录运行。

### 1. 核查原始提交：仅需Python

```powershell
python team_review/three_submissions/verify.py
```

检查三份CSV的SHA-256、50,000行、完整ID顺序、有限预测值，以及重组输入和代码快照哈希；不需要GPU、训练缓存或网络。

### 2. 独立重组预测：CPU、只需NumPy

```powershell
conda env create -f environment.review.yml
conda run -n cs5228-review python team_review/three_submissions/reproduce.py
```

使用包内 `components.npz` 的13路已训练基础预测和 `weights.json` 的已学习权重，重新生成三份CSV。逐行与原始提交比较，容差为1e-8 SGD；浮点运算和文本格式可能令重新写出的文件字节不同。此模式不训练模型、不读取目标答案，也不依赖原工作区、原始数据或大型检查点。

### 3. 从原始训练数据重训全部基础模型：CUDA GPU

```powershell
conda env create -f environment.training.yml
conda run -n cs5228-final-training python team_review/three_submissions/train.py
```

训练配方固定在 [training_recipes.json](training_recipes.json)。入口直接使用仓库 `data/train.csv` 和所给辅助表，训练历史版10个基础模型、4个删减36月字段的FT种子、新PLR-FT及宏观CatBoost，共16个基础预测器，随后应用已学习融合权重生成三份新CSV。模型权重和输出保存在 `output/team_review_trained/`，不覆盖包内原始提交。

这是最终管线的基础模型重训，不是从零重跑数百个历史开发实验，也不重新搜索融合权重。CUDA重训可能有数值差异，415.37529等分数属于包内原始提交，新训练文件需另行提交才知道分数。此分支不包含大型 `.joblib`、`.pt` 检查点，检查点由训练入口生成。

重训后可运行 `python team_review/three_submissions/check_trained.py`，重新加载16份检查点核对保存的逐行预测。本机本次全量重训的三份预测数值与原提交完全一致；重新序列化的换行或浮点文本格式可能不同，详见检查记录。

## Conda与测试

- `environment.review.yml`：Python 3.11＋NumPy 2.4.6，供CPU重组。
- `environment.training.yml`：完整训练依赖及CUDA版PyTorch；版本根据原 `cs5228` 环境记录。
- 实际本机验证使用的Python、NumPy、GPU与通过项记录在 `CHECKS.md`、`checks/` 中。环境文件的创建命令不等于已在新机器上创建测试。

## 队友建议先读

1. [METHODS.md](METHODS.md)：三种方案的差异、时间边界和报告主线。
2. [comparison.csv](comparison.csv)：准确分数和文件对应。
3. [weights.json](weights.json)：原始历史权重、宏观权重、最终分组权重。
4. [evidence/](evidence/)：模型对照、特征阶梯、融合增量和此前复现核查。
5. [source_manifest.json](source_manifest.json)：本分支保留的相关训练代码快照及来源哈希。

当前分支只做本地整理，不自动推送或发送给任何人。
