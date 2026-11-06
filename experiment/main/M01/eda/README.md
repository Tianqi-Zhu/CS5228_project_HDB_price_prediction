# EDA 目录说明

本目录保存已冻结的探索性数据证据。主线应优先使用
[`report.md`](report.md)、`figures/`、`tables/` 和 [`summary.json`](summary.json)。

## 目录结构

```text
experiment/main/M01/eda/
├── report.md                 # 中期报告可直接采用的综合 EDA 结论
├── notes.md                  # 更详细的中文分析记录
├── summary.json              # 综合 EDA 的机器可读摘要
├── figures/                  # 综合 EDA 的 11 张主图
├── tables/                   # 综合 EDA 的 18 张统计表
└── supplementary/
    ├── train_test/           # 主数据专项检查、5 张简图和摘要
    └── auxiliary/            # 六个辅助文件的专项报告、3 张图和摘要
```

## 两种统计口径

综合 EDA 与补充分析回答的问题不同，因此保留两套口径并明确标注：

- 综合 EDA 将设施距离连接回每条租赁记录，用于分析距离与月租之间的描述性关系；小学仅保留 `MAINLEVEL_CODE` 以 `primary` 开头的记录。
- Auxiliary 补充分析按唯一 HDB 楼栋统计距离，用于审计设施覆盖；小学同时包含 `primary` 和 `p1` 混合学段。

因此，两个摘要中的学校数量和距离分布不能直接互换。主报告统一使用综合 EDA
口径；楼栋覆盖和逐文件数据质量可引用补充报告。

## 重新生成

```powershell
conda activate cs5228
python -m analysis.eda.run
```

也可以通过 `python -m analysis.main eda` 运行同一入口。脚本读取 `data/` 中的
原始 CSV，并在运行前后检查源文件；新输出默认写入忽略的 `output/eda/`，不会覆盖本目录的冻结证据。

旧 `progress_report.pdf` 是由同一套图表生成的 15 页中期演示稿，已从 main 移除。需要时运行 `python -m analysis.eda.pdf_report`，从本目录生成 `output/reports/progress_report.pdf`。若使用新 EDA 数据，指定 `--eda-dir output/eda`。它不是最终报告的 8 页科学论文。
