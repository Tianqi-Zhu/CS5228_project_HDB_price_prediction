# 分支运行核验

本分支保留原项目架构、三个环境文件、requirements/ 和全部九份原始 CSV。数据与原始提交按 SHA256 核对，未重新生成或缩减。

初次拆分记录在 [checks/summary.json](checks/summary.json) 和 [checks/export_check.json](checks/export_check.json)：选定的 53 项测试中 52 项通过、1 项因未附带历史缓存而跳过；原提交 CPU 重组在 50,000 行上的预测差为零。这些记录不代表完成了所有历史研究实验。

推送前的最新运行范围、完整测试结果及干净导出验证见 [checks/push-validation/](checks/push-validation/)。训练环境使用本机 cs5228 环境；额外模型依赖来自本机第三方包目录，不带入原工作区的训练检查点或特征缓存。未验证全新 Conda 环境安装。

| 入口 | 目的 |
|---|---|
| `python review.py verify` | 核对本分支文件和数据哈希 |
| `python reproduction/f1/verify.py` | 核对冻结配方、组件和原提交 |
| `python reproduction/f1/reproduce.py` | CPU 重组原提交，不训练 |
| `python reproduction/f1/train.py` | 从原始数据重训固定配方 |
| `python reproduction/f1/check_trained.py` | 重新加载新训练检查点并核对预测 |

历史 Kaggle 分数只对应原提交文件；新训练输出未经 Kaggle 评分。F2 的生成链仍待补充，本次拆分及运行核验不会将其标成已完成。

TabPFN integration checks are recorded separately in [checks/tabpfn-split/](checks/tabpfn-split/). These do not replace the earlier F1 full-training evidence. F2 remains pending.

Latest main requirement-card execution and whole-tree verification: [checks/main-final/](checks/main-final/). New scientific results are in [experiment/main/](../experiment/main/README.md). This supplements the earlier F1 full-training and TabPFN-isolation logs.
