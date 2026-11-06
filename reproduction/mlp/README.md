# MLP集成、全量训练与时间加权

完整整合`mlp-eda-ensemble`的MLP运行实现、配置、全量提交和时间加权实验。原main的F1、Concat和TabPFN模块保持原实现；MLP在独立进程加载`runtime/analysis`，使用同一份根目录`data/`。

| 已有MLP提交 | 训练行数 | Kaggle截图显示分数 |
|---|---:|---:|
| [全量未清洗](artifacts/fullfit/submission_mlp_fullfit_uncleaned.csv) | 150,000 | 439.38193 |
| [全量明显低价清洗](artifacts/fullfit/submission_mlp_fullfit_cleaned.csv) | 147,983 | **437.34455** |

成绩来自原提交截图，公私榜未知；本次合并没有重新上传Kaggle。CSV和[截图/分数对应记录](artifacts/fullfit/kaggle_results.json)保持原字节。全量使用截至2025-03的已知标签，三个随机种子、各3轮，预测50,000行。置信区间模型使用更早截止点，不能将其旧区间直接用于全量模型。

## 从main根目录运行

```bash
# 仅核验文件、原始数据和四份提交，不导入训练依赖
python -m analysis.main mlp verify
# MLP相关单元测试，不训练完整数据
python -m analysis.main mlp test
# 重训原全量MLP、保存重载并生成两份50,000行CSV
python -m analysis.main mlp fullfit --output-dir output/mlp/fullfit-new
# 半衰期36个月的可选时间加权候选
python -m analysis.main mlp fullfit --half-life-months 36 --output-dir output/mlp/weighted-new
# 早期两窗选权重，后续年度评估，再全量重训
python -m analysis.main mlp recency-search --output-dir output/mlp/recency-new
# 原分离校准流程，生成MLP/TabM区间及测试预测
python -m analysis.main mlp intervals --output-dir output/mlp/intervals-new --predict-test
# 在刚生成的基线上重做两种清洗对照
python -m analysis.main mlp cleaning --baseline-dir output/mlp/intervals-new --output-dir output/mlp/cleaning-new
```

也可使用`python reproduction/mlp/run.py ...`。输出目录必须是新目录；新训练权重、预处理和结果留在被忽略的`output/`。算法运行需要CPU版PyTorch及[MLP依赖](../../requirements/mlp.txt)，无需CUDA。已测试环境为Python3.12/PyTorch2.14.1；建议单独环境，不覆盖F1的CUDA环境。macOS若缺少OpenMP，在命令前设置`DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib`；可设置`MPLBACKEND=Agg MPLCONFIGDIR=/tmp/hdb-mpl`。

## 时间权重的结果边界

权重为`2 ** (-月龄 / 半衰期)`，归一化为平均1，作用于回归和辅助分类损失。6/12/24/36个月候选中36个月最好；早期合并验证仍是均匀权重最低。后续相同年度验证中RMSE从470.2343降到469.7138，MAE从341.4548到341.4755，未显示稳定优势。[加权记录与CSV](artifacts/recency/)保留作对照，没有Kaggle成绩。

MLP使用房屋/空间字段、地址嵌入、市场调整历史和残差目标；44个编码前字段，与F1及Concat的异质输入不同。本接入不声称同输入架构排名，也没有把MLP加入F1原来的融合权重。年度验证已反复用于开发。2025-03的380条测试记录与训练月份重叠，历史特征排除同月标签，但全量监督拟合包含该月已知标签。

[预测报告](../../docs/mlp_prediction_report.md) · [EDA](../../docs/mlp_eda.md) · [来源与路径适配](provenance.json)

源分支与main的原始CSV仅有LF/CRLF换行差异，逐字段值及行序一致。[数据身份记录](data_identity.json)同时保存两边字节哈希与统一LF后的哈希；核验只接受换行差异，不改写main数据。测试入口使用单个Torch线程避免本机OpenMP冲突；全量训练沿用原2线程设置。
