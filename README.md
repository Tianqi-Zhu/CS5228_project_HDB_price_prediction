# HDB月租实验：main本地审查分支

| 项目 | 内容 |
|---|---|
| 本地分支 | main |
| 审查入口 | [实验索引](experiment/main/README.md) |
| 文件与原始数据核验 | `python review.py verify` |
| 运行/环境 | [执行指南](docs/guides/running_experiments.md) |
| 当前实验规划 | [A–G规划与验收](docs/planning/main-appendix-experiment-plan.md) |
| 评分依据 | [SKILL.md](skills/cs5228-final-report/SKILL.md)：方法60%、报告30%、复现10% |
| 两个来源 | zch-complete-proj @ 048eca86；results/hdb-rental-final-ensemble @ 6f97e107 |
| 本分支范围 | 必要EDA/C7基线/固定特征阶梯/复杂模型对照/F1；F2接口为待补外部依赖 |
| 保留项目架构 | analysis/、data/、docs/、experiment/、requirements/、tests/、environment.yml；另有隔离reproduction/固定提交包 |
| 最终F1 | kaggle_structure_blend.csv，用户截图415.37529；仅此最终CSV进入本分支复现包 |
| 外部F2 | 用户截图413.19635；代码/87列/成员/验证[待补充]，未从其他成员分支导入 |
| 归档与验证 | 本次只整理及核查，不进行新训练/新实验或Kaggle上传；此前训练检查记录不冒充本次执行 |
| 对照分支 | appendix |
| 文件换行 | .gitattributes禁止隐式文本转换，以保证数据、冻结源和证据SHA256在检出后仍一致 |
