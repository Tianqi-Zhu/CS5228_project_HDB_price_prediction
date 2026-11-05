# 本地审查与执行入口

| 操作 | 命令 | 行为 |
|---|---|---|
| 查看实验归属 | `python review.py list` | 只读 |
| 核对文件与数据哈希 | `python review.py verify` | 只读，不训练 |
| 核对原提交 | `python reproduction/f1/verify.py` | 标准库，只读 |
| 重组预测 | `python reproduction/f1/reproduce.py` | CPU/NumPy，输出到忽略的output/，不训练 |
| 原配方全量重训 | `python reproduction/f1/train.py` | CUDA/完整训练依赖，结果写入 output/ |
| Y/J 固定配方重训 | `python reproduction/f1/validate.py --window Y --output-dir output/f1-validation/new-Y` | CUDA；J 使用另一个新目录；每窗 15 个基础模型 |
| Y/J 生成链核验 | `python -m analysis.main validation-refits` | CPU；校验代码/数据身份、逐行对齐与固定融合重组 |
| 有序特征清单 | `python -m analysis.main schemas` | CPU；区分两套 62 列及各组件输入 |
| 空间与清洗证据 | `python -m analysis.main evidence` | CPU；核验来源字节、数据版本、匹配对照和已附逐行预测 |
| 训练实验入口 | `python -m analysis.main --help` | main为精选入口；appendix保留原CLI；真正执行baseline/common-models/ladder等会训练 |

| 环境文件 | 用途 |
|---|---|
| environment.yml | 原完整项目环境，内容保持不变 |
| environment.review.yml | 原最终分支CPU审阅环境 |
| environment.training.yml | 原最终分支CUDA训练环境 |
| requirements/ | 原始可选模型依赖，保留完整 |

| 数据 | 状态 |
|---|---|
| data/train.csv、test.csv、example-submission.csv | 完整原文件，哈希与来源一致 |
| data/auxiliary/ | 六份所给辅助表，完整保留 |
| 缓存/大型模型 | 不复制原工作区缓存；完整历史训练需要按配置重建；原提交CPU重组不需要检查点 |
| 历史指标 | 被Git忽略的本地小型结果复制到experiment/main/中并记录来源；逐行大预测和全部开发模型不在此拆分范围 |

## main 训练命令

```bash
python -m analysis.main eda --scope all --output-dir output/review_eda
python -m analysis.main baseline --output-dir output/review_baseline
python -m analysis.main compare --baseline-dir output/review_baseline --output-dir output/review_compare
python -m analysis.main ladder --stage outer
python -m analysis.main common-models --device cuda --output-dir output/review_common
```

各子命令的 `--help` 只显示参数，不触发训练。`compare` 在缺少基线时先生成指定目录的基线。`ladder` 直接从原始数据构建特征，不依赖旧融合预测。`common-models` 优先使用本地历史内层搜索记录，缺少时使用随分支保存的 `experiment/main/M05/selection_records/` 固定记录，不在外层评估集上重新选参。

## 独立concat Transformer

`python -m analysis.main concat verify`核对归档；`python -m analysis.main concat predict --device cpu`加载最终权重重放。原始数据重建、验证训练和全量重训命令见[复现说明](../../reproduction/concat_transformer/README.md)。只有显式select/refit才启动训练。

## main requirement-card evaluation

`python -m analysis.main planned` audits the frozen Y/J predictions and generates per-card diagnostics without training. See [execution status](../planning/main-execution-status.md). Outputs go to a new content-addressed directory under `output/planning_validation/main/`; existing runs are never overwritten. `--output-dir output/planning_validation/recheck` selects another directory.

`python -m analysis.validation.repository_audit` checks all shipped hashes, every registered Python file syntax and active navigation. Model runtime checks and CPU replays are separately logged in `review/checks/main-final/`.

最新五项补齐与验收见 [五项优化](../planning/main-five-item-completion.md)。`planned` 同时调用三个新增的离线核验；其 `new_models_fitted=0` 表示该审计命令本身不训练，另外记录已完成的 30 次 Y/J 基础模型训练。原输入包保留，PLE/PLR 的旧 57 列元数据通过独立 schema 清单勘误，不修改原预测。

本地复用第三方依赖目录时，F1/main 使用 `.cache/deep-deps` 与 `.cache/model-deps`；TabPFN SDK 的 `.cache/tabpfn-split-deps` 只用于隔离工作目录的 TabPFN 测试/入口。不要把这三套目录合为 main 的通用搜索路径。正式安装仍以对应环境与 requirements 为准，本轮没有新建 Conda 环境。
