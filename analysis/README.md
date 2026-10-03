# `analysis` package

项目的 Python 实现只维护在这里。

| 目录／文件 | 职责 |
| --- | --- |
| `main.py` | 唯一对外 CLI；运行 EDA、实验、验证、报告和提交 |
| `pipeline/` | 数据处理、模型、训练、评估和预测流程 |
| `eda/` | 综合 EDA、专项审计和 PDF 报告 |
| `experiments/` | 已报告的基线和四模型比较 |

依赖方向固定为：

```text
main.py
  ├── eda/ ─────────┐
  ├── experiments/ ─┼──> pipeline/
  └── pipeline/ ────┘
```

`pipeline/` 不反向导入 EDA 或 experiments，避免循环依赖。常用命令：

```powershell
python -m analysis.main eda
python -m analysis.main baseline
python -m analysis.main compare
python -m analysis.main validate --model lightgbm --features C
```
