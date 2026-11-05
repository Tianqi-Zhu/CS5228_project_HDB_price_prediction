# 离线实验验收

| 模块 | 职责 | 依赖 |
|---|---|---|
| `main.py` | M00/M01/M03/M04/M05/M07/M08/M09 的冻结预测审计与诊断 | main 中的输入 NPZ、原始数据与常规科学计算环境 |
| `prepare_inputs.py` | 从原历史预测和设置构建冻结输入；重建合法支持与输入相等分组 | 本地历史模型产物；不训练 |
| `repository_audit.py` | 完整文件哈希、所有已注册 Python 的语法与当前导航链接 | 根目录 manifest；不导入训练器 |

[当前验收状态](../../docs/planning/main-execution-status.md) · [报告评分依据](../../skills/cs5228-final-report/SKILL.md)
