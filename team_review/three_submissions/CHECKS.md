# 独立工作区运行检查（2026-11-05）

以下为实际执行结果，不把提供环境文件等同于已在另一台机器安装验证。

| 检查 | 结果 |
|---|---|
| 三份原CSV的哈希、50,000行、ID顺序和有限值 | 通过 |
| 36份相关代码快照哈希 | 通过 |
| CPU重新组合三份预测 | 通过；与原提交逐行最大差异均0 |
| 模型结构及融合测试 | 9项通过 |
| Conda三个YAML文件语法 | 通过 |
| 独立目录全量基础模型重训 | 16个模型均完成，约553秒 |
| 重训产生三份50,000行预测 | 通过；本机数值与原提交逐行差异均0 |
| 新训练检查点重新加载 | 16份全部通过，预测差异均0 |

实际使用现有Conda环境 `cs5228`：Python 3.11.15、NumPy 2.4.6、Pandas 3.0.5、PyTorch 2.11.0+cu128、CUDA 12.8，GPU为RTX 4060 Laptop。未新建Conda环境；根目录 `environment.yml`/`environment.review.yml` 提供CPU审阅环境，`environment.training.yml` 提供训练环境。

RTDL数值嵌入、FT骨干及XGBoost在本次运行中从此独立工作区自己的隔离依赖目录加载，而非原工作区缓存；新环境按训练YAML安装同版本包即可。原始数据来自该分支已有 `data/`，与记录的SHA-256一致；重训没有读取原工作区的模型权重。

检查证据：

- [预测重组](checks/verification_run.json)
- [16模型全量训练记录](checks/training_run.json)
- [16模型检查点重载](checks/checkpoint_replay.json)
- [重训与原提交逐行比较](checks/retrain_comparison.json)
- [环境与隔离依赖](checks/environment_checked.json)
- [测试摘要](checks/test_summary.json)

大型新训练权重和运行输出位于此工作区的 `output/`，未纳入Git提交。三份真实提交及小型重组输入纳入分支，供队友不运行训练也可核查。

## 上传前的额外干净环境检查

从已提交Git内容导出全新目录，确认无原工作区缓存和模型权重。新建Python 3.11.15虚拟环境，仅从官方PyPI安装NumPy 2.4.6，三份预测核验与重组通过；最大数值差异约9.1e-13 SGD，低于1e-8的检查容差。新环境没有安装Pandas或PyTorch，CPU审阅不依赖训练环境。

另对 `environment.training.yml` 中全部pip依赖执行官方PyPI/PyTorch索引的安装解析dry-run，通过；该步骤没有安装CUDA训练环境。完整训练可运行的依据仍为上面在现有Conda环境中实际完成的16模型全量拟合和重载记录。

证据：[干净环境上传前核查](checks/pre_push_check.json)、[训练环境依赖解析](checks/training_dependencies_resolved.json)。
