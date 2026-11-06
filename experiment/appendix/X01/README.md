# X01 特征冗余与空间关联

当前状态：**archived**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。

| 推理环节 | 内容 |
|---|---|
| 问题/假设 | 历史特征是否重复编码？设施之间的关联是否受房型/月份混杂？ |
| 对照与口径 | 固定模型删列/组，与完整输入比较；空间调整模型另列，不把相关当预测增量。 |
| 观察 | 删除时间与历史的联合变化远大于单组；MRT×商场交互补充主线单变量关联。重叠组效应不可相加。 |
| 取舍依据 | 保留支持历史信息的消融和完整空间控制；没有证据据此推导设施因果效应。 |
| 服务 main | M01/M03/M04；方法质量60%、解释清晰度30%、来源/复现10% |

## 证据入口

- [group_feature_effects.csv](feature_evidence/group_feature_effects.csv)
- [mrt_mall_interaction_summary_zh.md](spatial_controls/reports/spatial_features/amenity_interactions/mrt_mall_interaction_summary_zh.md)
