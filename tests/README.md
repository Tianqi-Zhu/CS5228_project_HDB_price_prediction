# main 测试与运行核验

在仓库根目录、安装对应环境后运行：

```powershell
python -m unittest discover -s tests -v
python -m analysis.validation.repository_audit
python -m analysis.main planned
```

| 范围 | 当前实际存在的测试文件 |
|---|---|
| 数据、共享管线、公共模块与保存结果 | `test_pipeline.py` |
| 空间辅助特征及特征筛选 | `test_auxiliary_features.py`、`test_feature_selection.py`、`test_feature_evidence.py` |
| 历史、周年窗口与融合边界 | `test_second_wave.py`、`test_third_wave.py`、`test_ninth_wave.py`、`test_tenth_wave.py` |
| 行 ID、标签、分组误差与指标 | `test_main_validation.py` |
| F2 组件顺序、数值有效性及换行身份 | `test_final_pair.py` |
| MLP 隔离、数据身份与四份提交 | `test_mlp_integration.py`；独立运行时测试通过 `mlp test` 执行 |
| 特征元数据修正与清洗匹配对照 | `test_main_evidence_completion.py` |
| F1 固定权重、成员顺序与支持组 | `test_f1_validation.py` |
| CLI 帮助及参数转发 | `test_main_cli.py` |
| Concat 集成、梯度及 Windows/POSIX 运行兼容 | `test_concat_integration.py`、`test_concat_portability.py` |
| 可生成产物与冻结证据目录隔离 | `test_output_layout.py` |

专项运行可用 `python -m unittest discover -s tests -p test_output_layout.py -v`。旧完整仓库中其他实验的测试不属于当前 main 测试清单，不能据旧文档把它们计入本分支覆盖范围。

TabPFN 使用独立工作目录，避免覆盖 main 的 `analysis` 包：

```powershell
python review_tabpfn.py prepare --workspace output/tabpfn-tests
python review_tabpfn.py test --workspace output/tabpfn-tests
```

原 TabPFN 测试需要该工作流的依赖；缺失源分支外部资源的测试由其声明文件控制并显式报告。不会下载权重或调用 API。main 的临时第三方依赖路径与 TabPFN SDK 路径应分开，见[运行指南](../docs/guides/running_experiments.md)。

测试主要检查实现与信息边界，不代替模型泛化评价或完整重训。`review.py verify` 和 `repository_audit` 检查文件身份、语法及导航；`planned` 重算冻结预测，并核验已完成的新 F1 生成链。全量训练、Y/J 固定配方训练、原提交重放和大型模型核验见[复现入口](../reproduction/f1/README.md)。

数据、模型结果和最终结论分别由对应实验材料支撑；F2 未完成部分仍保留待补。历史验收记录见 [review/CHECKS.md](../review/CHECKS.md)。
