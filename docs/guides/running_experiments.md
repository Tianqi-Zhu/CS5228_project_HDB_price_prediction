# 本地审查与执行入口

| 操作 | 命令 | 行为 |
|---|---|---|
| 查看实验归属 | `python review.py list` | 只读 |
| 核对文件与数据哈希 | `python review.py verify` | 只读，不训练 |
| 核对原提交 | `python reproduction/f1/verify.py` | 标准库，只读 |
| 重组预测 | `python reproduction/f1/reproduce.py` | CPU/NumPy，输出到忽略的output/，不训练 |
| 原配方全量重训 | `python reproduction/f1/train.py` | CUDA/完整训练依赖，结果写入 output/ |
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
