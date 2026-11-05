# 离线实验验收

| 模块 | 职责 | 依赖 |
|---|---|---|
| `main.py` | M00/M01/M03/M04/M05/M07/M08/M09 的冻结预测审计与诊断 | main 中的输入 NPZ、原始数据与常规科学计算环境 |
| `prepare_inputs.py` | 从原历史预测和设置构建冻结输入；重建合法支持与输入相等分组 | 本地历史模型产物；不训练 |
| `repository_audit.py` | 完整文件哈希、所有已注册 Python 的语法与当前导航链接 | 根目录 manifest；不导入训练器 |
| `feature_catalog.py` | 有序 schema、组件输入和旧 PLE/PLR 元数据勘误 | 冻结元数据及新 Y/J 保存模型输入记录 |
| `evidence_audit.py` | 空间来源/版本、设施统计口径、清洗匹配对照与逐行重算 | M01 来源和紧凑 NPZ；不重新训练 |
| `refit_audit.py` | 新 F1 Y/J 的代码提交/数据身份、模型重载记录与固定融合重组 | 30 次新训练记录和组件数组；审计本身只用 CPU |

[当前验收状态](../../docs/planning/main-execution-status.md) · [报告评分依据](../../skills/cs5228-final-report/SKILL.md)
