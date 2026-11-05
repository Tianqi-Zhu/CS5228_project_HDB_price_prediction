# 项目结构：先读结论，再追实现和运行明细

## 顶层只有三类工作

- 读研究：从 `README.md` 进入 `experiment/`。
- 看实现或运行：进入 `analysis/`，用 `docs/` 查命令与依赖。
- 查明细和模型：进入 `output/`，原始输入仍在 `data/`。

```text
README.md
environment.yml                  核心Conda环境：17个包，训练版本固定、工具允许范围
experiment/                         阅读和引用的成果
  README.md                      阅读顺序
  01_final/                      最终方法、完整报告、三份提交对照
  02_feature_engineering/         特征说明、字段字典、消融与阶梯
  03_model_comparison/            共同输入模型、推荐模型、关键图表
  04_optimization_history/        十轮探索，每轮一份汇总Markdown
  eda/                           数据质量、表格和描述图
  studies/                       经济、空间和面板研究
analysis/                        实现与运行入口
  main.py                        统一命令
  experiment_catalog.py          只读实验索引
  resource_preflight.py          按任务检查环境、文件和记录哈希
  artifact_catalog.py            当前模型／参数清单，不加载模型
  report_bundle.py               更新精选阅读报告
  pipeline/                      共享数据、特征、模型、评估与路径
  experiments/
    baselines/                   基线与早期实验
    feature_engineering/         特征选择、表示与消融
    model_comparison/            模型搜索、共同输入与准备工具
    reporting/                   模型筛选、报告及分支盘点
    submissions/                 本地导出与最终选择
    optimization/
      round_01_history_target_search/
      round_02_anniversary_history/
      round_03_tree_transformer_ensemble/
      round_04_macro_spatial_diagnostics/
      round_05_feature_interactions/
      round_06_deep_models/
      round_07_pretrained_transformers/
      round_08_feature_pruning_seeds/
      round_09_grouped_blending/
      round_10_plr_transformer_final/
  eda/                           数据分析实现
  studies/                       解释性研究实现
  project_review.py              基础27列阶段报告生成器
docs/                            怎么用、为什么这样组织
  README.md
  guides/
    running_experiments.md        环境、命令、前置产物与训练边界
    project_structure.md         本文件
  methods/
    baseline_features.md         基础27列定义
    history_feature_optimization.md 历史表示与目标分解
    model_comparison.md          模型搜索配置与协议
  archive/                       中期、整合与旧工作流
  maintenance/                   清理规则和变更摘要
data/                            原始比赛输入，不改写
  README.md                      九份CSV、目标字段与信息边界
requirements/                    同一主环境的可选扩展
  README.md                      按任务选择安装入口
  optional_deep_models.txt       TabM / RealMLP
  optional_pretrained_transformers.txt TabDPT / Nori
tests/                           自动检查
  README.md                      测试分类、执行命令与依赖边界
output/                          运行产物，Git忽略
  experiment_runs/               所有逐次配置、指标、选择记录、原始报告
  models/                        模型、融合权重与对齐预测
    current_catalog.json         当前清单，关联用途与指标来源
    index.json                   保留的早期索引快照
  submissions/
    final/submission.csv         当前final原样副本
    candidates/                  三份及其他候选CSV
    components/
      history_macro_ensemble/    历史与宏观版全量组件
      round_08_ft_seed_average/   第八轮种子平均
      round_09_grouped_blend/     第九轮融合权重
      round_10_final_structure/   第十轮最终结构
    baselines/
      baseline_27_features/      早期27列参照
  checks/
    runs/                        实际运行核查与日志
    records/                     按日期的机器核查JSON及来源清单
    reviews/                     审核结论、差异补丁与执行状态
  archive/                       此前归档的运行副本
.cache/                          特征缓存、隔离依赖、作者代码与公开权重
.integration-work/               其他独立worktree与临时核查
```

## 命名规则

`baseline_*`表示基础参照，`features_*`表示特征实验，`comparison_*`表示模型比较，
`round_01_*`至`round_10_*`表示按时间推进的性能探索。
报告阅读目录按叙事顺序排列；运行和模型目录则保留轮次，方便逐一对应。
问题尝试保留在 `archive_weight_scale_debug`，不参与正式排名。

Python脚本保留原有模块命令，例如 `analysis.experiments.tenth_wave_structures`。
源码物理位置已经分组，索引JSON中的 `source`可定位文件。
保存模型需要的模块名仍可解析，不因目录名改变而重新训练或修改预测。

## 阅读文件和运行文件的区别

`experiment/`只放精选正文、汇总表和关键图；`output/experiment_runs/`保留全部可追溯明细，
不会为了简化视图而删除失败试验。
`python -m analysis.main experiment-refresh`从本地已有结果更新25份阅读文件，不训练，
手写的各栏目导航不会被覆盖。

来源清单与本次迁移核查通过[维护目录](../maintenance/README.md)索引，机器明细保存在`output/checks/records/`。
此前维护记录中的旧路径代表当时布局；当前路径以本次迁移映射为准。
`.cache/`中的依赖和权重删除后需要重新准备，不能把整个目录都视为无用文件。
