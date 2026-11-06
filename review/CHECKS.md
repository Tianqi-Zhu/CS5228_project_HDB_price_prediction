# 分支运行核验

## 当前 main 更新后的核验

基于 `4b91fad` 整理后的本机记录见 [main-refresh/summary.json](checks/main-refresh/summary.json)。85 项 main 测试、32 项独立 MLP 测试、38 项隔离 TabPFN 测试与集成 `planned` 通过；MLP 和 TabPFN 分别复用已有隔离依赖，未把两套运行时合入 F1 环境。

本轮新增 `analysis.main final-pair`：F2 的 50,000 行候选可按两路保存组件重组，最大差约 `5.0e-7` SGD，符合六位小数的舍入容差；近期 28,995 行 F1/F2 配对完成。年度 F2、完整有序字典、信息政策完全等价和历史截图关联仍未验收，main 全部 P0 尚未完成。

下文是不同提交、机器和环境下的历史记录；其中“F2 生成器缺失”和 Python 3.13 字节重组失败不能代替本机当前 Python 3.11 的核验状态。F1 固定生成链在本机已通过；本轮未新跑完整研究模型训练或 TabPFN checkpoint 推理。

本分支保留原项目架构、三个环境文件、requirements/ 和全部九份原始 CSV。数据与原始提交按 SHA256 核对，未重新生成或缩减。

初次拆分记录在 [checks/summary.json](checks/summary.json) 和 [checks/export_check.json](checks/export_check.json)：选定的 53 项测试中 52 项通过、1 项因未附带历史缓存而跳过；原提交 CPU 重组在 50,000 行上的预测差为零。这些记录不代表完成了所有历史研究实验。

推送前的最新运行范围、完整测试结果及干净导出验证见 [checks/push-validation/](checks/push-validation/)。训练环境使用本机 cs5228 环境；额外模型依赖来自本机第三方包目录，不带入原工作区的训练检查点或特征缓存。未验证全新 Conda 环境安装。

| 入口 | 目的 |
|---|---|
| `python review.py verify` | 核对本分支文件和数据哈希 |
| `python reproduction/f1/verify.py` | 核对冻结配方、组件和原提交 |
| `python reproduction/f1/reproduce.py` | CPU 重组原提交，不训练 |
| `python reproduction/f1/train.py` | 从原始数据重训固定配方 |
| `python reproduction/f1/check_trained.py` | 重新加载新训练检查点并核对预测 |

历史 Kaggle 分数只对应原提交文件；新训练输出未经 Kaggle 评分。F2 的生成链仍待补充，本次拆分及运行核验不会将其标成已完成。

TabPFN integration checks are recorded separately in [checks/tabpfn-split/](checks/tabpfn-split/). These do not replace the earlier F1 full-training evidence. F2 remains pending.

Latest main requirement-card execution and whole-tree verification: [checks/main-final/](checks/main-final/). New scientific results are in [experiment/main/](../experiment/main/README.md). This supplements the earlier F1 full-training and TabPFN-isolation logs.

## 五项补齐的本次验收

当前材料与入口见[五项优化](../docs/planning/main-five-item-completion.md)。日志位于 [checks/five-item-validation/](checks/five-item-validation/)。

| 范围 | 已观察结果 |
|---|---|
| main 测试 | 73 项通过，无跳过 |
| 隔离 TabPFN 原测试 | 30 项通过，无跳过；不调用 API、不下载模型 |
| F1 Y/J 新生成 | 每窗 15 个基础模型训练及重载通过；39 份训练源文件逐窗按实际提交核验；新融合 CSV 逐字节重组通过 |
| F1 原提交 | 集成 `planned` 已执行 verify 和 50,000 行 CPU 重组，保留原 CSV |
| Concat | 28 份文件核验与 50,000 行 CPU 重放通过；最大差 0.105321 SGD，在原 `0.10 + 1e-4 × abs(value)` 容差内 |
| 集成审计 | 原 26 组预测、144 行误差分组、特征清单、场景、空间/清洗来源与可用逐行预测核验通过 |
| 全树静态核验 | 625 份注册文件、200 个 Python 文件及当前导航链接通过 |

提交 `8ebf6eb` 的干净导出也已通过：625 份注册文件检查、73 项 main 测试、30 项隔离 TabPFN 测试、集成 `planned`、F1 新入口帮助及 Concat 50,000 行 CPU 重放。导出未带入本地缓存或既有训练输出；本机第三方依赖按环境隔离复用。详见 [summary.json](checks/five-item-validation/summary.json) 和各 `clean-*.log`。

本轮新训练限于 F1 固定配方 Y/J，未重跑全部历史搜索、完整 Concat 八轮重训或新的独立留出测试。大型新检查点留在本地忽略目录，随分支保存紧凑预测与生成记录。F2 仍待实际生成材料；main 全部 P0 未完成。

## 结构与中间产物清理

规则及保留理由见[目录说明](../docs/guides/repository_layout.md)，逐项操作见 [cleanup.json](checks/structure-cleanup/cleanup.json)。删除前核验了旧导出目录及压缩包的 Git 对象身份，未删除分支工作目录、模型输出或已安装依赖。旧阶段 PDF 另有字节一致的本地备份。

本次 78 项 main 测试通过；默认 EDA 的三个层次实际运行完成，18 张表与冻结 M01 结果一致，11 张主图生成成功，原始 CSV 未改变。集成 `planned`、F1 生成链、Concat/TabPFN 文件核验和新帮助入口通过。大模型训练配方、最终结果与冻结运行时均未改动，也没有重新训练研究模型。运行记录位于 [checks/structure-cleanup/](checks/structure-cleanup/)。

提交 `c854542` 的干净导出再次通过完整 78 项测试、626 份文件静态核验、集成 `planned` 与默认 main EDA 运行。未携带本地缓存和训练输出；验证后移除该临时导出，保留 [committed-export.json](checks/structure-cleanup/committed-export.json) 及日志。


## Latest anniversary branch merge

Current integration evidence is in [anniversary-merge/summary.json](checks/anniversary-merge/summary.json). The unchanged main package passes 78 tests; the isolated latest workflow passes 38 tests without skips. The original F1 50,000-row numerical replay passes. Source mapping and conflict choices are recorded separately from scientific model results.

The strict fresh-refit serialized CSV SHA check in `analysis.main planned` fails in this runtime. Separate Y/J numerical replay reproduces RMSE/MAE exactly and bias within 4e-15 SGD; this does not establish byte-identical reproduction. The checker and frozen scientific inputs are unchanged. The candidate F2 export/source is present, while its association with the historical displayed score remains unverified.
