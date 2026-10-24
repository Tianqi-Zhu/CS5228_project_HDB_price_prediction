# HDB 公开租金记录匹配

本分支只包含公开租金记录匹配方法、所需四份 CSV、检查代码及复现说明。
不依赖其他分支的 `analysis/`、模型权重、特征缓存、GPU 或 Kaggle 凭据。
完整思路和逐步代码解读见 [docs/method.md](docs/method.md)。
独立环境与干净克隆的实测复现记录见 [docs/reproduction.md](docs/reproduction.md)。

## 快速复现

使用 Python **3.11**。在此分支根目录打开终端，创建独立环境并运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m public_rent_lookup.run
```

macOS/Linux 将上述 `.venv\Scripts\python.exe` 换成 `.venv/bin/python`。
安装依赖需要能访问 PyPI；四份输入 CSV 已包含，计算阶段只读本地文件，无需联网下载数据。

脚本会自行创建 `outputs/`，输出 50,000 行的 `outputs/submission.csv`（`Id,Predicted`），
并打印训练集重建误差与测试匹配统计。没有预先生成的预测文件，也可以完整运行。

## 预期结果

在包含的固定数据快照及固定依赖下：

| 项目 | 数值及含义 |
| --- | --- |
| 全训练集公开目标重建 RMSE | **179.9471744827 SGD**，150,000 条已知标签实测 |
| 2024-04 至 2025-03 重建 RMSE | **181.2533674383 SGD**，35,158 条已知标签实测 |
| 测试匹配覆盖率 | **50,000 / 50,000** |
| 测试单一租金行 | **36,487（72.974%）** |
| 测试多候选租金行 | **13,513（27.026%）** |
| 测试误差量级估算 | **173.4121857270 SGD**，基于公开组内方差及等概率候选假设 |
| 实际 Kaggle 测试分数 | **未知，本分支不声称已经测得 173.41** |

这是原始目标信息的重建实验，没有模型训练。已知标签重建 RMSE 不能视作历史信息预测未来的泛化成绩。

## 文件与输出

```text
README.md
requirements.txt
.gitignore
.gitattributes
public_rent_lookup/
  __init__.py
  run.py                     # 完整本地匹配、核验和导出流程
  download_public.py         # 可选刷新公开数据
tests/
  test_lookup.py
  test_download.py
data/
  README.md
  train.csv
  test.csv
  example-submission.csv
source/
  hdb_public_rents.csv        # 固定公开快照，212,597 行
  hdb_public_rents.json       # 来源、下载时间、哈希
results/
  summary.json               # 固定快照的参考运行统计
  reproduction_check.json    # 干净克隆、输入输出哈希一致的检查记录
docs/
  method.md                  # 数据来源、思路、示例、公式、代码步骤与边界
  reproduction.md            # 实测独立复现的过程及结果
```

`outputs/` 是重新生成的结果目录，被 Git 忽略：

| 生成文件 | 内容 |
| --- | --- |
| `submission.csv` | 测试集租金结果，保持原始 ID 和行序 |
| `test_matches.csv` | 每条测试记录的匹配键、候选数量、均值、范围和歧义类型 |
| `train_reconstruction.csv` | 训练行对应公开均值、真实租金及误差 |
| `ambiguous_test_groups.csv` | 多候选测试分组及其测试行数、候选统计 |
| `monthly_metrics.csv` | 各月份的覆盖率、歧义数量和误差统计 |
| `summary.json` | 本次统计、输入输出哈希、代码哈希和环境版本 |

可指定输出目录，例如 `python -m public_rent_lookup.run --output-dir outputs/check`。
路径默认根据脚本所在目录定位，不依赖作者机器的绝对路径。

## 可选刷新公开数据

默认复现直接使用已包含的快照，不需要此步骤。若希望另存最新官方数据：

```powershell
python -m public_rent_lookup.download_public
python -m public_rent_lookup.run --public-data source/hdb_public_rents_latest.csv --output-dir outputs/latest
```

下载工具通过官方公开 API 分页读取并检查数量、重复 ID 和租金有效性。
默认另存 `source/hdb_public_rents_latest.csv` 及元数据，不替换固定快照；输出已存在时需要
另选路径或显式 `--force`。官方服务不可用时，已包含的固定快照仍可用于离线计算。
最新数据可能修订历史记录，因此刷新后的误差不保证与表中数值完全相同。

## 组员阅读顺序

1. 阅读 [完整方法说明](docs/method.md)，先了解低误差来自哪些信息。
   第 5.1 节用测试第 0 行演示完整过程：原始地址 → 规范化匹配键 → 五条公开候选租金 → 均值 3100 → 提交 CSV。
2. 安装依赖并运行检查，再执行一遍 `public_rent_lookup.run`。
3. 比较新生成的 `outputs/summary.json` 与 `results/summary.json` 的指标和输入哈希。
4. 打开 `outputs/test_matches.csv`，分别找一条 `single_rent` 与 `multiple_rents_mean` 记录，
   按规范化键在公开数据中核对候选租金。

代码不会自动提交 Kaggle，也不会进行登录或发送消息。使用公开目标记录是否符合课程要求，
应依据课程及比赛规则；该实验不证明任何其他队伍采用了相同方法。
