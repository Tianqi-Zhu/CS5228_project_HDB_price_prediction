# HDB月租实验：main 主线

| 项目 | 内容 |
|---|---|
| 分支 | main |
| 审查入口 | [实验索引](experiment/main/README.md) |
| 文件与原始数据核验 | `python review.py verify` |
| 运行/环境 | [执行指南](docs/guides/running_experiments.md) |
| 当前实验规划 | [A–G规划与验收](docs/planning/main-appendix-experiment-plan.md) |
| 评分依据 | [SKILL.md](skills/cs5228-final-report/SKILL.md)：方法60%、报告30%、复现10% |
| F1来源 | zch-complete-proj @ 048eca86；results/hdb-rental-final-ensemble @ 6f97e107 |
| 本分支范围 | 必要EDA/C7基线/固定特征阶梯/复杂模型对照/F1；F2接口为待补外部依赖 |
| 保留项目架构 | analysis/、data/、docs/、experiment/、requirements/、tests/、environment.yml；另有隔离reproduction/固定提交包 |
| 最终F1 | kaggle_structure_blend.csv，用户截图415.37529；仅此最终CSV进入本分支复现包 |
| 外部F2 | 用户截图413.19635；代码/87列/成员/验证[待补充]，已导入TabPFN背景与探索；最终F2生成链仍待补 |
| 归档与验证 | 推送前运行记录见 [review/CHECKS.md](review/CHECKS.md)；新训练输出未上传 Kaggle |
| 对照分支 | appendix |
| 文件换行 | .gitattributes禁止隐式文本转换，以保证数据、冻结源和证据SHA256在检出后仍一致 |

## 独立架构探索：Concat Transformer

在团队统一的最终历史特征输入上，比较宽度与深度，选定128维6层concat Transformer。年度开发验证RMSE459.2189；全量重训8轮后的Kaggle Public Score418.38657（用户提交截图）。它作为M05独立架构探索补充，不替换现有F1/F2定义。

- [方法与容量对照](experiment/main/M05/concat_transformer/README.md)
- [最终预测重放、原始数据特征重建与训练](reproduction/concat_transformer/README.md)
- 只读核验：`python -m analysis.main concat verify`

| Integration | Entry |
|---|---|
| TabPFN workflow | [Integration guide](docs/guides/tabpfn_workflow.md) |
| TabPFN classification | [Decisions](docs/planning/tabpfn-branch-split.md), [directory table](docs/planning/tabpfn-classification.csv) |
