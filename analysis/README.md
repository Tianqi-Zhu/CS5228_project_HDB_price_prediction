# `analysis` package

项目的 Python 实现只维护在这里。

| 目录／文件 | 职责 |
| --- | --- |
| `main.py` | 唯一对外 CLI；运行 EDA、实验、验证、报告和提交 |
| `pipeline/` | 数据处理、辅助特征、模型、训练、评估和预测流程 |
| `eda/` | 综合 EDA、专项审计和 PDF 报告 |
| `studies/` | 经济、时间面板和空间控制等解释性研究 |
| `experiments/` | 基线、四模型、辅助消融、历史空间、深度、TabM 和 ReLU 实验 |
| `verification.py` | 环境、报告完整性、源哈希和指标一致性检查 |

依赖方向固定为：

```text
main.py
  ├── eda/ ─────────┐
  ├── studies/ ─────┤
  ├── experiments/ ─┼──> pipeline/
  ├── verification.py
  └── pipeline/ ────┘
```

`pipeline/` 不反向导入 EDA 或 experiments，避免循环依赖。常用命令：

```powershell
python -m analysis.main eda
python -m analysis.main baseline
python -m analysis.main compare
python -m analysis.main validate --model lightgbm --features C
python -m analysis.main deep --architecture transformer --target-encoding --residual-baseline
python -m analysis.main deep-summary
python -m analysis.main spatial-controls
python -m analysis.main aux-ablation
python -m analysis.main economic-ablation
python -m analysis.main spatiotemporal
python -m analysis.main tabm
python -m analysis.main relu
python -m analysis.main verify
```
