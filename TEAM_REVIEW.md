# 三份 Kaggle 提交的队友审阅入口

本地分支：`review/three-kaggle-submissions-20261105`。本分支整理三份真实提交、准确分数、训练代码快照、已学习的融合权重、CPU预测重组入口和全量重训入口。

先读 [审阅说明](team_review/three_submissions/README.md)，再看 [方法对照](team_review/three_submissions/METHODS.md)。

推荐 final：`kaggle_structure_blend.csv`，用户截图中的Kaggle显示分数 **415.37529**。

```powershell
conda env create -f environment.review.yml
conda run -n cs5228-review python team_review/three_submissions/verify.py
conda run -n cs5228-review python team_review/three_submissions/reproduce.py
```

完整基础模型重训使用 `environment.training.yml`，需要NVIDIA CUDA GPU：

```powershell
conda env create -f environment.training.yml
conda run -n cs5228-final-training python team_review/three_submissions/train.py
```

重组与重训输出分别进入 `output/team_review_reproduced/`、`output/team_review_trained/`。原始提交文件保持不变。完整重训使用已选定超参数和已学习的融合权重，不重复历轮开发搜索；新训练输出的Kaggle分数未知。
