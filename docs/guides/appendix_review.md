# Appendix：解释性分析的运行与阅读

本分支补充 main 之外的假设检验、替代方案和取舍依据。阅读 [实验索引](../../experiment/appendix/README.md) 与 [状态/缺口](../planning/appendix-execution-status.md)，无需按 Git 分支历史逐个猜测实验含义。

## 复算本轮解释实验（不训练）

需要核心环境中的 numpy、pandas、scikit-learn、matplotlib。仓库仍保留原环境文件与九份原始 CSV；可选深度学习包不影响下列纯预测分析。

```powershell
python -m analysis.appendix_review verify
python -m analysis.appendix_review run --output-dir output/appendix-review/fresh-1
```

输出路径必须是仓库 `output/` 下的新目录，不覆盖旧结果。一次运行生成 X08/X09/X11/X12/X13 子目录和 `execution.json`，无新拟合、无 API、无 Kaggle 上传。归档结果位于各卡的 `evaluation/`，普通运行不改写归档。

| 输出 | 解释用途 |
|---|---|
| X11/member_ablation.csv | 七树、TabPFN、固定等权 F2 与 F1 同行比较 |
| X11/error_moments.json | MSE 交叉矩恒等式；实际损失与误差相关的区别 |
| X11/group_metrics.csv | 逐月与24/36月支持组；不能把 F1 说成 F2 成员 |
| X12/paired_intervals.csv | F2−F1 的 RMSE/MAE 描述性区间 |
| X12/resampling_spec.json | seed42、各3,000次、3个月非循环区块、楼栋簇及未包含的不确定性 |
| X08/location_flags.csv / feature_sensitivity.csv | 来源状态问题与特征改变量；不是性能改善 |
| X09/query_horizons.csv | 380共同月、1–10月、11–16月查询分布 |
| X13/component_replay.json | 冻结 F2 数值重组；不代表在本机新跑 GPU 推理 |

X12 先把同一月份或楼栋中的 n、两模型 SSE/绝对误差求和，再按相同抽样重数重算 pooled RMSE 与 MAE。月份从 0..T−3 中有放回抽三月连续块，拼接后截到 T 个月；不循环到开头。楼栋按规范化 Town/Block/Street 聚类，有放回抽同数量簇。不会去重丢失抽样重数，也不对逐行误差做 i.i.d. bootstrap 冒充时间稳健性。J 只有10个月，且已被反复用于开发；区间不覆盖训练/选模不确定性。

## 输入来源

`reproduction/appendix_diagnostics/source_manifest.json` 固定 main `d075c81` 的证据身份。约2.31 MB 的压缩输入是原保存数组的无损数值子集：F1 Y/J、F2 J 两组件、覆盖标志和全量 F2 组件。原 CSV、NPZ、提交及代码文件的来源提交和 SHA256 均有记录。它不需要旁边存在一个 main 工作区，也不读取本地未提交报告。

`python -m analysis.appendix_imports` 是维护者的导入工具，需要本地 Git 对象库含固定来源提交及列出的来源分支；普通克隆只复算/核验时**无需运行导入**。`X05/external_studies/` 和 `X01/spatial_controls/` 保留原来源路径和字节。其中代码属于历史源快照，不承诺只拷这几个文件便可独立重训；完整原始实现应通过清单里的提交查看。旧报告内部链接可能指向未选入的历史文件，不作为当前导航使用。

## 独立复现层次

```powershell
python review.py verify
python reproduction/three_submissions/verify.py
python reproduction/three_submissions/reproduce.py
python reproduction/concat_explorations/verify.py
python reproduction/mlp_explain/run.py verify
python reproduction/mlp_explain/run.py diagnose --output-dir output/mlp_explain/new-diagnostics
python -m unittest discover -s tests -p test_appendix_statistics.py -v
```

MLP 完整重训需另读[MLP原协议](../../experiment/appendix/X06/mlp_explain/README.md)。TabPFN 隔离原测试使用 `python review_tabpfn.py test --workspace output/tabpfn-fresh-tests`，首次会准备源码/数据副本；模型依赖不齐时明确失败/跳过原档案声明的外部资源，不能当作完成 GPU 推理。

X07/X09 和严格历史设施重训的资源缺口在状态表中保留。本阶段不重启历轮大搜索，也不修改 main 或报告。
