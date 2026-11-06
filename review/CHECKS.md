# 分支运行核验

## 当前 appendix 增量完善

当前范围见[执行表](../docs/planning/appendix-execution-status.md)与[解释分析入口](../docs/guides/appendix_review.md)。基于远端 `444811a` 保留 MLP 原解释材料，补入固定来源的 RNN/LSTM、动态租金面、空间交互；新增 X11/X12 的 J 分析及 X08/X09 来源/跨度审计。本轮新增科学模型拟合数为0。

运行证据位于 [appendix-refresh](checks/appendix-refresh/)。F2 已可核验 J 预测与50,000行保存组件重组；年度、完整schema/信息政策、历史截图关联仍待补。下面旧记录中的“F2仍缺生成链”属于当时提交的状态，不代表当前缺少全部F2材料。CPU重放、旧源GPU证据、全新环境与本轮完整重训分开登记。

```powershell
python -m analysis.appendix_review verify
python -m analysis.appendix_catalog audit
python -m unittest discover -s tests -p test_appendix_statistics.py -v
```

## 最终验收

当前最终验收：[validation.json](checks/appendix-refresh/validation.json)、[干净导出](checks/appendix-refresh/clean-export.json)。提交 `9dd06ad` 的干净导出通过4,423份文件校验、845份Python语法和116处活动链接检查；在无main工作区/无历史训练缓存的导出中独立复算X08/X09/X11/X12/X13，数字表与归档一致。新环境仅复用了本机现有Python和显式依赖路径，没有声称新建Conda环境。

| 验收项 | 结果与边界 |
|---|---|
| 干净导出核心测试 | 140项运行，136通过、4项可选作者包/源码缺失跳过；无错误/失败 |
| 新统计契约测试 | 7项通过，包含在核心集合内，不重复计数 |
| MLP隔离测试 | 37项通过，无跳过 |
| TabPFN隔离原测试 | 280项运行，279通过、1项因原档案未附带2020/2023学校HTML快照而声明跳过 |
| 三份原提交 | 每份50,000行CPU组件重放通过 |
| F2组件 | 50,000行保存预测按等权重组，在原六位小数舍入容差内；不是新GPU推理 |
| 历史Concat源码 | 481文件、303份Python语法通过；Windows清单读取显式UTF-8 |

首次尝试的来源分区哈希登记和Windows编码失败、修正后的回归均在记录中保留；SDK测试复用既有RTDL依赖后通过，没有安装或升级依赖。

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
