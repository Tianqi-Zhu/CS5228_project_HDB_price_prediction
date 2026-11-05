# TabPFN 拆分与执行

| 入口 | 用途 |
|---|---|
| `python review_tabpfn.py list` | 查看当前分支包含的原始入口与 F2 状态 |
| `python review_tabpfn.py verify` | 按哈希核对原始代码、结果和派生数据，不训练 |
| `python review_tabpfn.py metrics` | 从月度汇总重算已有整体 RMSE、MAE、bias；不执行模型推理 |
| `python review_tabpfn.py preflight` | 列出本机包、checkpoint、缺失资源和历史哈希差异，不下载或认证 |
| `python review_tabpfn.py test` | 在独立工作目录运行当前分支所附的原始测试 |
| `python review_tabpfn.py prepare` | 复制原代码、共享数据和结果到 `output/tabpfn-workspace/` |
| `python review_tabpfn.py run aligned -- --help` | 查看原始参数，不训练 |

分支分类见[拆分决策](../planning/tabpfn-branch-split.md)和[逐目录决策表](../planning/tabpfn-classification.csv)。本次实际核验记录位于 `review/checks/tabpfn-split/`。

## 环境与架构

`workflows/tabpfn/` 保存 TabPFN 原始 `analysis/`、`tests/`、`environment.yml` 与 `requirements-*.txt`。`environment.anniversary-original.yml` 保存合并分支的环境，根目录已有环境文件继续用于 F1。原始 README 和 API 文档作为来源材料保留，原命令应在组装后的独立工作目录中理解。

```powershell
conda env create -f workflows/tabpfn/environment.yml
conda activate cs5228-hdb
python -m pip install -r workflows/tabpfn/requirements-tabpfn.txt
```

CUDA 构建按原文件说明准备。本轮未验证全新环境安装；原记录与本机的 GPU、NumPy/Pandas 版本存在差异，完整推理需另行核验。API 路线依赖 `requirements-tabpfn-api.txt`，本次不调用它。

原来的同名模块保留在独立工作流内；`review_tabpfn.py` 将其放入 `output/`，并以单独进程运行。根目录 `analysis/`、F1 固定包和原始数据不被替换。训练输出与缓存写到该工作目录。需要新目录时，使用 `--workspace output/tabpfn-new-run`；现有工作目录与新快照不一致时入口会拒绝复用。

## 数据和结果

| 内容 | 处理 |
|---|---|
| 九份比赛原始 CSV | 继续使用根目录 `data/`，组装时复制到工作目录；保留 150,000 条原训练样本 |
| 清洗、去重等派生数据 | 保存在 appendix 的 `data/processed/`，明确作为替代数据口径 |
| 原结果目录 | 按分类保存；组装后还原为 `reports/<原目录名>/`，使原相对路径可追溯 |
| 原代码与来源哈希 | `workflows/tabpfn/manifest.json`；不修改历史结果中的 source/code 哈希以伪装成新运行 |
| 最终提交 CSV 与成员预测 | 来源 Git 未包含的文件继续标记缺失；不从汇总表反造预测 |

`metrics` 仅检查汇总一致性。组件月表未提供行数时，按原始训练集的对应月份人数加权；较早的抽样验证不混入完整 holdout 的汇总检查。原始逐行预测缺失时，该检查不能证明模型推理、样本行对齐或无泄漏。

## 运行边界

`run` 要求显式传入原始参数；无参数不会触发原脚本的默认训练。完整推理需要相应包、指定 checkpoint、原特征/适配缓存及其哈希。`preflight` 会显示当前缺口，补齐后再执行原配方。main 只包含必要入口；完整探索与历史提交入口位于 appendix。

`submission_zch_tabpfn87_equal_150k.csv` 仍待提供实际生成提交、schema、权重、CSV 与预测。现有 ABCDE/ABDE 配方及开发集结果是不同候选，不能替代该 F2。

## Validation dependency snapshot

`workflows/tabpfn/requirements-validation-extra-lock.txt` records the repository-local supplemental packages used for original test validation. It complements the existing cs5228 numerical/PyTorch environment; it is not a complete Conda lock. Original requirement files remain unchanged. The tests are run offline.

One appendix integration test needs two untracked archived school HTML inputs (2020/2023). `test-resources.json` declares their exact paths. The test wrapper skips only that named test when the files are absent, reports the skip, and leaves every original test file unchanged. Supplying the real cached files enables the original assertion.

Model checkpoints and legacy caches belong inside the generated workspace, e.g. `output/tabpfn-workspace/.cache/tabpfn/`. `preflight --workspace output/another-run` inspects that exact execution directory. It does not silently borrow the root F1 model cache.

On Windows, original experiment paths can exceed the default Git path limit. This repository enables `git config --local core.longpaths true`. When cloning into a deep directory, use `git -c core.longpaths=true clone <repository-url>` so the archived paths are preserved.
