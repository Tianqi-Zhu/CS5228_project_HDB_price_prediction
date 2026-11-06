# main 目录与产物保留规则

| 目录 | 角色 | 保留原则 |
|---|---|---|
| `analysis/pipeline/` | 当前共享数据、特征和学习器实现 | 保留；只通过相应实验入口运行 |
| `analysis/experiments/` | 按用途和轮次分组的主线实现与依赖 | 旧模块名用于导入及检查点兼容，不按“是否直接进入论文”删除 |
| `analysis/validation/` | main 的协议、特征、来源与复现核验 | 保留；主要入口为 `python -m analysis.main planned` |
| `analysis/eda/`、`analysis/studies/` | EDA 与补充研究实现 | 新生成结果默认分别进入 `output/eda/`、`output/studies/` |
| `data/` | 九份原始数据文件 | 保留原字节；不搬移、缩减或清洗覆盖 |
| `experiment/main/M00`—`M09` | 已冻结实验材料、主要结果、配套表图 | 保留来源与可核验材料；不是每次运行的临时输出目录 |
| `reproduction/f1/` | F1 固定配方、原始提交、组件预测及新 Y/J 生成证据 | 保留；独立冻结运行时承担原模型复现职责 |
| `reproduction/concat_transformer/` | 已合并的 Concat 架构、权重、预处理和重放输入 | 保留；大型 `bank.npz` 用于离线重放，不能当作无用缓存删除 |
| `workflows/tabpfn/` | 隔离的 TabPFN 工作流与依赖说明 | 保留；生成工作目录放到 `output/` |
| `docs/` | 运行、方法、规划与当前状态 | 从根 README 进入；`docs/reviews/` 的旧计划文件只作跳转 |
| `requirements/` 与根环境文件 | 核心及可选模型环境 | 保留，不按本机已安装情况删减 |
| `review/` | 文件/来源清单与各阶段验收记录 | 保留；旧日志记录不同提交和运行阶段，不冒充最新结果 |
| `output/` | 新模型、预测、特征缓存、图表、临时导出及本地备份 | Git 忽略；确认来源和用途后按具体路径清理 |
| `.cache/` | 本机依赖、作者代码、固定权重与部分可重建缓存 | 分别判断；依赖包、固定模型和来源快照不是普通临时文件 |
| `.integration-work/` | 其他分支工作目录 | 独立于当前 main；其中有 appendix 和其他分支，不能整体删除 |

## 当前入口

```powershell
python -m analysis.main --help
python -m analysis.main planned
python -m analysis.main eda
python -m analysis.validation.repository_audit
```

EDA 默认写入 `output/eda/`；所有层的默认路径一致。`python -m analysis.eda.pdf_report` 从冻结的 M01 EDA 生成旧中期演示稿到 `output/reports/progress_report.pdf`；使用新分析结果则加 `--eda-dir output/eda`。不把中期 PDF 当作最终科学论文。

`python -m analysis.report_bundle` 是旧阅读包辅助入口，依赖已有历史实验输出，默认写入 `output/reading_bundle/`。它不是 main 的验收命令，也不再向 `experiment/01_final` 等过时目录复制中间文件。旧 `project_review`、`artifact_catalog` 等工具仍有相互导入和本地历史产物用途，暂不按入口列表删掉。

## 本轮已清理

| 内容 | 处理 | 恢复方式 |
|---|---|---|
| M01 的旧 `progress_report.pdf` | 从 Git main 删除，保留本地 SHA256 一致备份；约 1 MB | Git 历史、本地 `output/archive/main-structure-cleanup/`，或由保留脚本生成 |
| 8 个旧导出目录、6 个对应压缩包 | 按对应 Git 提交验证全部受跟踪文件后删除；导出目录同时检查不存在额外用户文件 | 由记录中的提交执行 `git archive`，再按运行指南重建临时输出 |
| 拆分过程的 pip 下载缓存 | 删除；没有删除已安装依赖目录 | 安装所需依赖时可重新生成下载缓存 |
| 旧测试文档中的不存在入口、错误 EDA 输出说明 | 更新为当前 main 的实际文件和命令 | 以本文、测试 README 及根 README 为准 |

本地共移除 15 个临时目标，约 **4.90 GiB**。逐项身份、大小与执行记录见 [清理记录](../../review/checks/structure-cleanup/cleanup.json)。Git 历史的磁盘占用不会因移除 PDF 自动缩小；本轮没有改写历史。

## 未当作垃圾删除

原始数据、F1/Concat 的最终结果与检查点、F1 Y/J 新训练模型、融合所需预测、未完成的 F2 材料、六份未跟踪审查草稿、其他分支工作目录和本机依赖全部保留。重复输入明细用于 EDA 的真实样本冲突检查；源分支的表格和配置用于证据追溯。这些文件即使不放正文，也不等于没有用途。

完整研究状态仍见[main 执行表](../planning/main-execution-status.md)。F2 缺口不会因目录清理而变为完成。
