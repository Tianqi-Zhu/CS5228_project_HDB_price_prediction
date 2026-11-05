# 三份历史提交与宏观对照复现入口

| 项目 | 值 |
|---|---|
| 来源 | results/hdb-rental-final-ensemble @ 6f97e107ed2db72347c3d7d1f85b63999b3fcbbb |
| 范围 | 三份原提交及其固定权重对照，供X03/X13审查 |
| 标准库核查 | `python reproduction/three_submissions/verify.py` |
| CPU预测重组 | `python reproduction/three_submissions/reproduce.py`（需要NumPy） |
| 全量重训 | `python reproduction/three_submissions/train.py`（CUDA、完整训练环境；本次不执行） |
| 重载新检查点 | `python reproduction/three_submissions/check_trained.py` |
| 数量 | 13路组件、16个基础预测器、3份CSV |
| 数据 | 直接使用本分支根目录data/；不复制第二份数据 |
| 代码隔离 | runtime/保存原flat模块；config只调整项目根路径；不与根目录分组模块混用 |
| 已知限制 | 原分数是用户截图显示值、公私榜未知；重组不是新训练；新重训CSV无已知Kaggle成绩 |
