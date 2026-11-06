# 分支运行核验

## 当前 appendix 增量完善

当前范围见[执行表](../docs/planning/appendix-execution-status.md)与[解释分析入口](../docs/guides/appendix_review.md)。基于远端 `444811a` 保留 MLP 原解释材料，补入固定来源的 RNN/LSTM、动态租金面、空间交互；新增 X11/X12 的 J 分析及 X08/X09 来源/跨度审计。本轮新增科学模型拟合数为0。

运行证据位于 [appendix-refresh](checks/appendix-refresh/)。F2 已可核验 J 预测与50,000行保存组件重组；年度、完整schema/信息政策、历史截图关联仍待补。下面旧记录中的“F2仍缺生成链”属于当时提交的状态，不代表当前缺少全部F2材料。CPU重放、旧源GPU证据、全新环境与本轮完整重训分开登记。

```powershell
python -m analysis.appendix_review verify
python -m analysis.appendix_catalog audit
python -m unittest discover -s tests -p test_appendix_statistics.py -v
```

## 历史验证记录

本分支保留原项目架构、三个环境文件、requirements/ 和全部九份原始 CSV。数据与原始提交按 SHA256 核对，未重新生成或缩减。

初次拆分记录在 [checks/summary.json](checks/summary.json) 和 [checks/export_check.json](checks/export_check.json)：选定的 53 项测试中 52 项通过、1 项因未附带历史缓存而跳过；原提交 CPU 重组在 50,000 行上的预测差为零。这些记录不代表完成了所有历史研究实验。

推送前的最新运行范围、完整测试结果及干净导出验证见 [checks/push-validation/](checks/push-validation/)。训练环境使用本机 cs5228 环境；额外模型依赖来自本机第三方包目录，不带入原工作区的训练检查点或特征缓存。未验证全新 Conda 环境安装。

| 入口 | 目的 |
|---|---|
| `python review.py verify` | 核对本分支文件和数据哈希 |
| `python reproduction/three_submissions/verify.py` | 核对冻结配方、组件和原提交 |
| `python reproduction/three_submissions/reproduce.py` | CPU 重组原提交，不训练 |
| `python reproduction/three_submissions/train.py` | 从原始数据重训固定配方 |
| `python reproduction/three_submissions/check_trained.py` | 重新加载新训练检查点并核对预测 |

历史 Kaggle 分数只对应原提交文件；新训练输出未经 Kaggle 评分。F2 的生成链仍待补充，本次拆分及运行核验不会将其标成已完成。

TabPFN integration checks are recorded separately in [checks/tabpfn-split/](checks/tabpfn-split/). These do not replace the earlier F1 full-training evidence. F2 remains pending.
