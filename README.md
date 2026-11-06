# HDB月租实验：appendix 附录

| 项目 | 内容 |
|---|---|
| 分支 | appendix |
| 审查入口 | [实验索引](experiment/appendix/README.md) |
| 文件与原始数据核验 | `python review.py verify` |
| 运行/环境 | [执行指南](docs/guides/running_experiments.md) |
| 当前实验规划 | [A–G规划与验收](docs/planning/main-appendix-experiment-plan.md) |
| 评分依据 | [SKILL.md](skills/cs5228-final-report/SKILL.md)：方法60%、报告30%、复现10% |
| F1来源 | zch-complete-proj @ 048eca86；results/hdb-rental-final-ensemble @ 6f97e107 |
| 本分支范围 | 消融、另两提交、调参/负结果、长窗、来源审计及复现细节 |
| 保留项目架构 | analysis/、data/、docs/、experiment/、requirements/、tests/、environment.yml；另有隔离reproduction/固定提交包 |
| 最终F1 | kaggle_structure_blend.csv，用户截图415.37529；作为另两提交的对照保留 |
| F2当前证据 | J 同行预测、tree7＋TabPFN等权及50,000行组件重组可核验；年度、完整有序字典/信息政策、历史截图分数关联仍缺 |
| 归档与验证 | 推送前运行记录见 [review/CHECKS.md](review/CHECKS.md)；新训练输出未上传 Kaggle |
| 对照分支 | main |
| 文件换行 | .gitattributes禁止隐式文本转换，以保证数据、冻结源和证据SHA256在检出后仍一致 |

## 按实验问题阅读

本分支以 main 之外的“问题/假设 → 匹配对照 → 观察 → 取舍”为入口，不把所有实验排成同一性能榜。

- [13张决策卡与主题索引](experiment/appendix/README.md)
- [本轮状态、未完成训练与材料缺口](docs/planning/appendix-execution-status.md)
- [可重算解释分析和复现指南](docs/guides/appendix_review.md)

```powershell
python review.py verify
python -m analysis.appendix_review verify
python -m analysis.appendix_review run --output-dir output/appendix-review/new-run
```

最后一条复算 X08/X09/X11/X12/X13 的已保存预测/特征诊断，不训练、不使用API，不改main或报告。匹配种子、严格历史设施重训、最终管线长窗与新Conda/GPU重训按真实缺口保持待做。仓库的完整试验档案与最终PDF的精简附录不是同一篇幅要求。

## Concat Transformer development appendix

Ziyuan’s earlier [spatial/temporal feature explorations](experiment/appendix/X05/concat_explorations/README.md) and [architecture comparisons](experiment/appendix/X06/concat_explorations/README.md) are archived separately from the final model in main. Verify the imported snapshot with `python reproduction/concat_explorations/verify.py`. No training is launched.

| Integration | Entry |
|---|---|
| TabPFN workflow | [Integration guide](docs/guides/tabpfn_workflow.md) |
| TabPFN classification | [Decisions](docs/planning/tabpfn-branch-split.md), [directory table](docs/planning/tabpfn-classification.csv) |

## MLP解释性实验

[完整报告与历史证据](experiment/appendix/X06/mlp_explain/README.md)包括误差分组、相同输入不同租金、预测区间、清洗对照和历史风险修正。运行 `python reproduction/mlp_explain/run.py verify` 核验归档；`diagnose` 从逐行预测复算诊断；训练命令见该目录说明。
