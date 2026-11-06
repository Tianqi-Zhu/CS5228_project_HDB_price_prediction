# 环境与可选扩展

核心依赖统一维护在根目录 `environment.yml`，覆盖数据分析、常规模型、报告工具和最终树／Transformer融合。

## 主环境

在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda activate cs5228
```

该文件采用当前已验证的Windows／NVIDIA配置，固定PyTorch 2.11.0+cu128、RTDL和数值嵌入等训练版本；绘图、PDF、Excel等工具保留适度版本范围。
版本来自本机已运行的环境及隔离依赖，见审核记录；本次不会自动安装或更新当前环境。
其他硬件使用者需要选择兼容的PyTorch构建。

可执行 `python -m analysis.main preflight --profile core` 查看核心版本和元数据所在位置；
`--profile wave06`、`--profile wave07`检查可选路线。预检不安装或导入模型，完整使用边界见[运行指南](../docs/guides/running_experiments.md)。

## 可选扩展

| 文件 | 用途 | 安装位置 |
| --- | --- | --- |
| [optional_deep_models.txt](optional_deep_models.txt) | 第六轮TabM与RealMLP；数值嵌入已进入主环境 | `.cache/deep-deps/` |
| [optional_pretrained_transformers.txt](optional_pretrained_transformers.txt) | 第七轮TabDPT、Nori与相关支持包 | `.cache/transformer-deps/` |

仅复现相应路线时，在主环境激活后执行：

```powershell
python -m pip install --no-deps --target .cache/deep-deps -r requirements/optional_deep_models.txt
python -m pip install --no-deps --target .cache/transformer-deps -r requirements/optional_pretrained_transformers.txt
```

可选实验继续沿用作者源码、公开权重及既有 `.cache/foundation-deps/` 共享支持包的准备流程，
见[运行指南](../docs/guides/running_experiments.md)和[逐轮实验说明](../experiment/04_optimization_history/README.md)。
可选TXT用于专项隔离安装；源码、权重和共享支持包仍是相应路线的复现前置条件。

旧的profiles/extras分层清单与根requirements.txt已按审核稿移除，
核心版本以environment.yml为准，避免多个基础安装入口各自漂移。
