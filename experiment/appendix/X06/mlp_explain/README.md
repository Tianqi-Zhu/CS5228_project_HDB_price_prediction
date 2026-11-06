# MLP 解释性实验

本目录将 `mlp-eda-ensemble` 的历史解释性实验合入 appendix。源提交为 `b3a6291fbfeb30ce7ff0d0495d9371d801ff8d1c`；已完成实验的预测和证据保持原始字节。此次只复算诊断并运行代码测试，没有重新训练全部网络，也没有提交 Kaggle。

- [预测报告](mlp_prediction_report.md)：模型、预测区间、误差集中与局部限制。
- [EDA与清洗报告](mlp_eda.md)：重复记录、低价候选、删除规则和失败对照。
- [误差诊断](uncertainty_diagnostics/summary.json)、[相同输入案例](uncertainty_diagnostics/indistinguishable_inputs_proof.json)。
- [剩余误差分组](eda/summary.json)、[训练案例](training_cases/)、[清洗对照](cleaning/comparison_metrics.csv)。
- [历史风险修正](risk_correction/result.json)、[低价回缩候选](recent_tail/outer_metrics.csv)、[历史架构实验目录](historical_catalog.csv)。
- [逐行验证预测](validation/)、[六份测试区间预测](predictions/)。测试预测没有真实标签。

原报告的运行段落已替换为本分支入口。其他历史实验仅归档结果，不代表全部搜索脚本已接入；以下入口支持区间基线、清洗、风险修正和低价回缩重训。权重不随分支上传，重新训练会生成权重。

## 不训练即可核验与解释

在仓库根目录运行；`verify` 只需 Python 标准库：

```bash
python reproduction/mlp_explain/run.py verify
python reproduction/mlp_explain/run.py diagnose --output-dir output/mlp_explain/diagnostics-1
```

`verify` 检查273份原始归档、复算5份验证预测的RMSE、核对6份测试CSV各50,000行及ID和区间顺序。原始数据仅允许LF/CRLF换行等价，不允许修改数值。

`diagnose` 额外生成 `summary.json`、`rent_bands.csv`、`monthly.csv`，复算原未清洗融合的尾部误差、覆盖率和相同输入案例，以及风险修正候选中“大误差但低分歧”的数量。输出目录必须是新的。编码后输入相等来自原始审计记录，此入口不重新加载旧权重验证编码。

原融合最差5%记录贡献41.53%的平方误差；≤1,900 SGD组占3.71%记录、贡献26.36%的平方误差。风险修正候选的1,422条误差≥1,000记录中，1,182条集成标准差仍<100。这些数字对应不同模型，不能混作最终全量MLP的成绩。

## 重新训练对照

安装依赖并测试：

```bash
python -m pip install -r reproduction/mlp_explain/requirements.txt
python reproduction/mlp_explain/run.py test
```

以下命令依次执行，均会训练；每个目录需保持独立：

```bash
python reproduction/mlp_explain/run.py intervals --output-dir output/mlp_explain/base --predict-test
python reproduction/mlp_explain/run.py cleaning --baseline-dir output/mlp_explain/base --output-dir output/mlp_explain/cleaning
python reproduction/mlp_explain/run.py risk --source-dir output/mlp_explain/base --output-dir output/mlp_explain/risk
python reproduction/mlp_explain/run.py tail --source-dir output/mlp_explain/base --oof-dir output/mlp_explain/risk --output-dir output/mlp_explain/tail
```

训练/验证时间界限沿用历史协议：截至2023-12拟合，2024-01至03校准，2024-04至2025-03验证。已有开发集反复被查看；这是诊断和敏感性分析，不是新的独立泛化评估。不能按真实租金在推理时分流，也不能将局部相同输入案例的最小误差解释为全数据的不可约误差。

macOS 如需动态库设置，在命令前加 `DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。本次测试使用CPU，37项测试通过；端到端训练命令本次未重新执行。模型文件和训练输出写入 `output/`。

[集成来源与边界](../../../../reproduction/mlp_explain/provenance.json) · [此次诊断复算](../../../../reproduction/mlp_explain/validation/summary.json)
