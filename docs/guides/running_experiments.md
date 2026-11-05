# 本地实验工作流

适用于当前完整项目分支 `zch-complete-proj`（原 `feature/midterm-eda-models`），从仓库根目录执行。
阅读入口在[报告导航](../../experiment/README.md)，逐轮汇总在[优化历程](../../experiment/04_optimization_history/README.md)，代码导航在[实验入口](../../analysis/experiments/README.md)。

## 先选任务，再运行命令

| 任务 | 入口 | 会发生什么 |
| --- | --- | --- |
| 查看最终结论 | 打开[FINAL_MODEL](../../experiment/01_final/README.md) | 只读，无需模型依赖 |
| 找脚本 | `analysis.main experiment-list` | AST列出入口，不导入实验 |
| 更新阅读报告 | `analysis.main experiment-refresh` | 从已有结果更新精选正文与表格，不训练 |
| 比较同输入模型 | `analysis.main common-models` | 训练，需较早搜索记录 |
| 解释特征增量 | `analysis.main feature-evidence` | 删除列／组并重训 |
| 重训27列参考管线 | `analysis.main canonical-submit` | 生成早期LightGBM预测，非最新final |
| 重建基础阶段报告 | `analysis.main project-review` | 读取已有结果并重写阶段报告、字段字典 |
| 运行后续性能探索 | `analysis.experiments.<轮次模块>` | 按各自参数训练或拟合融合 |
| 历史／宏观CSV导出 | `analysis.experiments.best_ensemble_submission` | 复用已核验组件，缺失时训练 |
| 第十轮核查与导出 | `analysis.experiments.tenth_wave_report` | 重载模型、重算、写报告，可能全量训练与导出 |
| 最终选择归档 | `analysis.experiments.final_submission_selection` | 核对CSV、复制final并重写选择文件与结论 |

这些提交命令只写本地文件，不上传Kaggle。查看结果不需要执行报告生成器。

实验代码已经按用途和轮次分组；原有模块命令保留，索引JSON中的`source`显示真实文件位置。
训练与阶段生成器把完整记录写入`output/experiment_runs/`，随后执行`experiment-refresh`更新阅读副本。

## 环境与索引

运行前可先做只读资源预检：

```powershell
python -m analysis.main preflight --profile core
python -m analysis.main preflight --profile reports
python -m analysis.main preflight --profile final
python -m analysis.main preflight --profile wave06
python -m analysis.main preflight --profile wave07
```

`core`检查九份比赛CSV、目标字段边界与核心包版本；`reports`另查精选报告的原始来源；
`final`另查15个已保存权重文件、特征包、12路融合预测、顺序及记录的输入／预测哈希；
`wave06`和`wave07`另查对应可选包、作者源码或公开权重。`all`合并这些检查，`--json`输出完整清单。
缺失或不匹配时退出码为1，并列出修复入口，不自动下载、安装、反序列化模型或训练。

预检使用标准库，也可独立执行 `python -m analysis.resource_preflight --profile final --json`。
包版本来自当前环境和已有隔离目录的元数据，每项显示实际提供位置；专项模型仍须通过其作者加载器启用相应路径。
通过预检表示静态资源条件满足，不保证所有传递依赖、CUDA推理或新环境安装可用；实际运行核查仍使用 `verify` 和测试集合。

```powershell
conda env create -f environment.yml
conda activate cs5228
python -m analysis.main --help
python -m analysis.main experiment-list --group wave10
python -m analysis.main experiment-list --group final --json
```

仅查看索引可以不用Conda建模依赖：

```powershell
python -m analysis.experiment_catalog --group wave06
```

完整分组包括 `baseline`、`features`、`models`、`report`、`frontier`、`wave02`至`wave10`、`final`、`other`。
没有 `__main__` 入口的模块是支持代码，不列为可直接执行的脚本。
索引展示的是命令前缀，运行前仍须查看所需参数。不要对没有参数解析器的旧脚本直接追加`--help`，
这类脚本可能忽略参数并开始工作。

GPU实验与分层依赖见[requirements说明](../../requirements/README.md)。基础Conda文件不是所有探索依赖的锁文件。
第六轮还需 `requirements/optional_deep_models.txt` 与作者代码，第七轮还需
`requirements/optional_pretrained_transformers.txt` 与对应公开权重；其下载器和版本记录见各轮README。
`.cache/`不随Git分发，包含依赖和权重的子目录缺失时须重新准备。

## 基础特征与模型对照

查看当前已保存模型、融合参数和指标来源，无需执行训练入口：

```powershell
python -m analysis.main artifact-list
python -m analysis.main artifact-list --usage final
python -m analysis.main artifact-list --write --hashes
```

最后一条计算哈希并写 `output/models/current_catalog.json`，文件多时需要等待。
`--json`输出逐文件条目；`--usage`可选择 `final`、`development`、`submission_component` 或 `discarded`。
筛选不能与 `--write`同时使用，避免把局部清单保存成全量索引。
清单保留原始指标的字段路径和窗口上下文，不将不同验证期的RMSE自动合并排名，
也不将全量融合的成绩误记为每个基础模型的成绩。模型数值不会因清单生成而改变。

下列命令会训练，按需要分别执行：

```powershell
python -m analysis.main feature-select
python -m analysis.main feature-optimize --stage all
python -m analysis.main feature-evidence
python -m analysis.main common-models --device cuda
python -m analysis.main canonical-submit
```

`common-models`复用`model-search`较早窗口记录；新机器的准备顺序见[模型对照说明](../methods/model_comparison.md)。
报告筛选中的19／27／47／62列阶梯有自己的固定配置，不能与另一轮27列模型表混算特征贡献。
阶梯训练命令为：

```powershell
python -m analysis.experiments.report_selection_trials ridge
python -m analysis.experiments.report_selection_trials ladder --stage outer
python -m analysis.experiments.report_selection_trials ladder --stage june
```

阶段文档生成器包括 `analysis.main project-review`、`report_selection_summary`、`report_best_scheme`。
它们会覆盖自己的Markdown输出，需明确想重建的阶段；无需为了阅读当前结果重新生成。

## 后续探索的依赖关系

```text
早期搜索／特征与逐行预测
  → 第一轮frontier
  → 第二轮同期历史与树融合
  → 第三轮FT／XGBoost及442.90历史主融合
  → 第四轮宏观等探索 → 历史／宏观提交导出

前轮对齐预测与模型 → 第五至七轮的特征／深度／Transformer对照
历史主融合 → 第八轮FT删减与四种子平均
前轮预测 → 第九轮分组融合
第九轮 + 第十轮新模型 → 结构融合及全量提交
三份已导出CSV + 截图记录 → 最终选择归档
```

这说明产物依赖，不表示每条支路都必须加入最终模型。最终结构版不包含宏观分支；
第五至七轮的全部候选也没有全部进入final。
完整历史重跑需要依照各轮README恢复模型、对齐预测和选择记录，不能只执行最后一个脚本。

代表性训练入口如下；每行是一个阶段示例，并非完整重跑脚本：

```powershell
python -m analysis.experiments.frontier_search screen
python -m analysis.experiments.second_wave_anniversary screen
python -m analysis.experiments.third_wave_models screen
python -m analysis.experiments.fifth_wave_features screen
python -m analysis.experiments.sixth_wave_deep screen
python -m analysis.experiments.seventh_wave_transformers screen
python -m analysis.experiments.eighth_wave_refinement ft_bag
python -m analysis.experiments.tenth_wave_structures screen
python -m analysis.experiments.tenth_wave_structures outer
python -m analysis.experiments.tenth_wave_structures june
```

第二至七轮常见 `screen` 是较早选择阶段，`outer`／`june`是年度／近期开发窗口；
融合脚本常用 `select`而非`screen`，以各自参数为准。比如：

```powershell
python -m analysis.experiments.third_wave_ensemble select
python -m analysis.experiments.third_wave_ensemble outer
python -m analysis.experiments.third_wave_ensemble june
```

第九轮 `ninth_wave_blending` 会拟合、比较融合规则；它没有上述位置参数。
当前开发窗口曾被反复查看，不能因命令名为outer而宣称是新独立测试。

## 三份提交的本地产物

| 文件 | 生成来源 | 前置条件 |
| --- | --- | --- |
| `kaggle_best_history.csv` | `best_ensemble_submission`的历史版导出，候选副本由后续归档维护 | 原10成员配方、选择记录、原始数据 |
| `kaggle_best_macro.csv` | 同上，加比赛所给宏观成员 | 历史版及宏观方案记录 |
| `kaggle_structure_blend.csv` | `tenth_wave_report`的结构融合导出 | 第九轮全量组件、第八轮FT种子、第十轮结构与融合记录 |

候选目录为 `output/submissions/candidates/`。
最终原样副本为 `output/submissions/final/submission.csv`，哈希与分数见
[FINAL_SELECTION.json](../../experiment/01_final/submission_selection.json)。

三份CSV和前置产物已经存在、且确实需要重建最终归档时，执行：

```powershell
python -m analysis.experiments.final_submission_selection
```

该命令含复制和报告重写，不是只读检查。截图分数是已记录的人工证据，命令不会访问Kaggle查询新分数。
重建阶段报告后应最后归档最终选择，核对生成文本是否仍与截图、文件哈希和本地指标相符。

## 检查与测试

本次新增索引的轻量检查：

```powershell
python -m unittest discover -s tests -p test_experiment_catalog.py -v
python -m analysis.main experiment-list --group wave10 --json
```

模型或融合代码变更后，可运行相应测试；下面只覆盖第九／十轮，不会重训所有历史实验：

```powershell
python -m unittest discover -s tests -p test_ninth_wave.py -v
python -m unittest discover -s tests -p test_tenth_wave.py -v
```

`python -m analysis.main verify`核查其实现覆盖的已有环境和结果，会读取本地产物；
它不等同于十轮从零重训。完整单元测试集合可用`python -m unittest discover -s tests -v`，
环境及历史产物要求见各项测试。本次整理的实际执行范围见[维护记录](../maintenance/README.md)。
