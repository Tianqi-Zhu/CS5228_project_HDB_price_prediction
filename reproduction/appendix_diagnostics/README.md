# Frozen appendix diagnostic inputs

来源与每份原始文件 SHA256 见 [source_manifest.json](source_manifest.json)。`inputs/predictions.npz` 为 main `d075c81` 的无损数值子集；不是从截图或文件名反推的数值。Y 仅含 F1，不含虚构的 F2 年度预测。全量查询仅有已保存预测，不含未知租金标签。

普通运行：`python -m analysis.appendix_review verify` 或按 [运行指南](../../docs/guides/appendix_review.md) 在新输出目录复算。输入、模型和权重只读。

`inputs/main_metadata.json` 保留主线原始元数据的原字节，包括当时记录的旧 schema 描述；本分析不以其57/62列历史错误项推断实际网络输入。实际 F1 异质输入与修正来源见 main `experiment/main/M03/feature_schemas/`。`f2_identity.json`、`paired_protocol.json` 和 `main_replay_scope.json` 同样为来源快照，之后的项目状态以本分支新执行表为准。
