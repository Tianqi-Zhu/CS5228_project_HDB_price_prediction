# F1最终方案复现入口

| 项目 | 值 |
|---|---|
| 来源 | results/hdb-rental-final-ensemble @ 6f97e107ed2db72347c3d7d1f85b63999b3fcbbb |
| 范围 | 仅结构融合F1；历史/宏观两份见appendix分支 |
| 标准库核查 | `python reproduction/f1/verify.py` |
| CPU预测重组 | `python reproduction/f1/reproduce.py`（需要NumPy） |
| 全量重训 | `python reproduction/f1/train.py`（CUDA、完整训练环境；既有完整训练验收见运行指南） |
| 重载新检查点 | `python reproduction/f1/check_trained.py` |
| 数量 | 12路预测、15个基础预测器、1份CSV |
| 数据 | 直接使用本分支根目录data/；不复制第二份数据 |
| 代码隔离 | runtime/保存原flat模块；config只调整项目根路径；不与根目录分组模块混用 |
| 已知限制 | 原分数是用户截图显示值、公私榜未知；重组不是新训练；新重训CSV无已知Kaggle成绩 |

## 固定配方 Y/J 验证生成链

`validate.py` 从本分支 `data/` 独立重建特征，按 [validation_recipe.json](validation_recipe.json) 固定的原点前融合权重和 [training_recipes.json](training_recipes.json) 训练轮数，各训练 15 个基础模型。所有拟合调用中的查询标签置零；不使用查询标签停止训练，不新增搜索或拟合融合权重。每个模型保存后立即重载并比对预测。

```powershell
python reproduction/f1/validate.py --window Y --output-dir output/f1-validation/new-Y
python reproduction/f1/validate.py --window J --output-dir output/f1-validation/new-J
python -m analysis.main validation-refits
```

前两条需要 CUDA 与训练环境，输出目录必须不存在。第三条只用 CPU 核验随分支保存的组件预测、源代码/数据哈希、精确生成提交与融合结果，不需要新训练或大型检查点。

| 归档 | 内容 |
|---|---|
| [Y](validation_evidence/Y/validation_run.json)、[J](validation_evidence/J/validation_run.json) | 生成提交、30 次训练及重载记录、输入列、固定轮数、数据和源码哈希 |
| [汇总核验](validation_evidence/audit.json) | 两窗 ID/标签、原始数据、提交代码与生成 CSV 的重算 |
| 各窗 `components.npz` | 12 路逻辑预测及支持组；供离线重组，避免重复提交完整模型与特征缓存 |

大型新检查点留在本地 `output/five_item_validation/`；重新训练可由前两条命令完成。新生成链与旧预测档案分别标识：不会据此声称已恢复旧缓存的原始生成提交。Y/J 仍为已查看的开发集，预测情景见 [边界声明](../../docs/methods/prediction-scenario.md)。
