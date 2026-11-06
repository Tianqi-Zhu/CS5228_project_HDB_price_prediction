# X13 复现层次与环境边界

当前状态：**partial**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 文件身份、组件重放、模型推理和从零训练分别证明什么？ |
| 对照与口径 | 三份原CSV独立重组；F2保存树/TabPFN输出重建，对照六位小数原候选。 |
| 观察 | F2的50000行可在半个最末小数单位容差内重组；不是本轮新GPU推理。 |
| 取舍依据 | 保留命令与来源链；新Conda安装和完整重训分别记录待做，不能从pip解析成功推导训练成功。 |
| 服务 main | M09；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [component_replay.json](evaluation/component_replay.json)
- [README.md](../../../reproduction/three_submissions/README.md)
- [source_manifest.json](../../../reproduction/appendix_diagnostics/source_manifest.json)
