# `cs5228` 运行核验

核验日期：2026-10-04。所有复跑输出写入 Git 忽略的 `.integration-work/project-check/`，没有覆盖正式报告或源数据。

## 环境

- Python 3.11.15，环境前缀 `D:\miniconda3\envs\cs5228`
- NumPy 2.4.6、pandas 3.0.5、scikit-learn 1.9.0
- CatBoost 1.2.10、LightGBM 4.7.0
- PyTorch 2.11.0+cu128，CUDA 12.8
- NVIDIA GeForce RTX 4060 Laptop GPU 可用
- `python -m pip check`：无破损依赖

## 已执行流程

| 流程 | 结果 |
| --- | --- |
| `compileall` | 通过 |
| `python -m unittest discover -s tests -v` | 30/30 通过 |
| `python -m analysis.main verify` | 所有结构、哈希和跨报告复用检查通过 |
| 完整 EDA | 150,000 训练行、50,000 测试行；11 图、18 表；楼栋匹配率 100% |
| CatBoost A/B/C 基线 | 与已保存 `model_metrics.csv` 完全一致 |
| Ridge/RF/CatBoost/LightGBM 比较 | 与已保存 `family_metrics.csv` 完全一致 |
| LightGBM 时间验证 | RMSE 485.09，成功生成月度、地区和房型误差 |
| `submit` → 保存模型 → `predict` | 50,000 行；列为 `Id,Predicted`；重新加载最大预测差为 0 |
| 深度模型 CUDA 冒烟 | 辅助特征 MLP 训练 1 epoch 成功 |
| 深度结果汇总 | 历史融合 RMSE 466.469947，和已保存汇总一致 |
| TabM CUDA 冒烟 | 1 个时间折、1 epoch、2 个成员成功 |
| ReLU 完整嵌套验证 | 选择 2 epoch；RMSE 463.776866 |
| 辅助特征消融 | 8 个阶段全部完成；正式指标完全一致 |
| 经济信号与经济特征消融 | 四个前向折全部完成；存在已解释的 sklearn 版本漂移 |
| 分组时间面板 | 完成；正式 pooled 指标完全一致 |
| 空间 overview、MRT 固定效应、匹配、设施固定效应 | 全部完成；抽查四份核心表完全一致 |
| 时空历史模型 | 三个前向折与最终 50,000 行预测完成；存在已解释的 sklearn 版本漂移 |
| 时间相关研究 | 完成；核心相关系数表完全一致 |
| PDF 报告 | 从临时 EDA 输出成功构建 |

`pytest` 未安装，但仓库测试由标准库 `unittest` 编写，README 规定的测试入口不依赖 `pytest`。

Windows 新版本缺少旧式物理 CPU 探测命令时，joblib 会打印一次警告并退回逻辑核心数。本次所有相关流程正常结束；这是性能探测提示，不是模型或依赖错误。

## 结果解释

固定版本的 CatBoost 与 LightGBM 流程可以精确复现。旧辅助分支没有保存完整的 resolved 环境，使用 scikit-learn 直方图梯度提升的报告在当前环境产生小幅数值漂移；数据 Git blob、行数、特征和时间切分一致。历史结果因此保留为分支实验记录，当前复跑结果不覆盖它们。详见 [`result_consistency.md`](result_consistency.md)。
