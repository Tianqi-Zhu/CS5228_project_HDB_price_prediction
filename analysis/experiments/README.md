# appendix实验代码

| 内容 | 定位 |
|---|---|
| 活跃实验 | experiment/appendix/，以实验卡ID为准 |
| 原始架构 | 保留baselines/、feature_engineering/、model_comparison/、optimization/、reporting/、submissions/的原路径；main只留选定入口及静态依赖闭包 |
| 依赖身份 | review/source_partition.csv：入口/测试、共享运行依赖、附录专属 |
| 特别说明 | sixth_wave、旧模型训练器等可能是F1结构的运行依赖，不因此变成main的独立实验；appendix保留完整原始代码用于复现 |
| F1固定训练快照 | reproduction/three_submissions/runtime/，与根目录分组模块隔离 |
