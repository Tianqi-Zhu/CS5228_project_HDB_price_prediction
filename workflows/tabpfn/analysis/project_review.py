"""Build the current project report from saved, comparable experiment evidence.

This command only reads result artifacts; it does not retrain models, rewrite
source data, or mix public-label reconstruction into forecasting rankings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.experiments.feature_evidence import GROUPS
from analysis.pipeline.config import AUX, DATA, EXPERIMENT_REPORTS, ROOT
from analysis.pipeline.data import sha256
from analysis.pipeline.features import FEATURE_CANDIDATES


OUT = ROOT / "output/experiment_runs/foundation_feature_review"
NAMES = {
    "property": "房屋属性", "time": "时间", "location": "区位", "mrt": "MRT",
    "mall": "商场", "primary_school": "小学", "block_history": "楼栋×户型历史",
    "area_history": "规划区×户型历史", "all_amenities": "全部设施",
    "all_history": "全部历史", "time_and_history": "时间＋全部历史",
    "lightgbm": "LightGBM", "catboost": "CatBoost", "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGradientBoosting", "extra_trees": "ExtraTrees",
    "embedding_mlp": "Embedding MLP", "embedding_resnet": "Embedding ResNet",
    "ft_transformer": "FT-Transformer 96", "ft_transformer_default": "FT-Transformer 192",
    "tabnet": "TabNet", "inner_blend": "内部选权融合",
}
FEATURE_NOTES = {
    "FLAT_TYPE_CLEAN": ("统一空格和连字符的户型类别", "不同户型可能代表不同居住容量", "main"),
    "FLAT_MODEL_CLEAN": ("统一大小写与空格的房屋模型", "用设计类型描述房屋差异", "main"),
    "FLOOR_AREA_SQM": ("原始面积，平方米", "描述房屋规模，不与户型的价值混为一谈", "main"),
    "FLAT_AGE_YEARS": ("审批年份减租约起始年份；负值置缺失", "作为房龄代理，不能当作装修年份", "main"),
    "MAX_FLOOR": ("楼栋表最高楼层，非法占位值置缺失", "描述楼栋形态；并不是出租单元的实际楼层", "block"),
    "TOWN_CLEAN": ("规范化市镇类别", "表示较粗的区位差异", "main"),
    "PLANNING_AREA_CLEAN": ("通过规范化地址匹配规划区", "补充市镇边界之外的区位划分", "block"),
    "LATITUDE": ("楼栋纬度", "提供连续空间定位，允许模型学习地理变化", "block"),
    "LONGITUDE": ("楼栋经度", "与纬度共同表示位置，可与距离代理共享信息", "block"),
    "CBD_DIST_KM": ("楼栋到代码中 CBD 参考点的球面距离，km", "作为中心性代理；不是通勤时间", "block"),
    "APPROVAL_MONTH_INDEX": ("自2021-01起的月份序号", "表示市场时间位置；树模型不保证外推趋势", "main"),
    "APPROVAL_MONTH_SIN": ("sin(2π×审批月份/12)", "周期编码避免12月与1月数值相距过远", "main"),
    "APPROVAL_MONTH_COS": ("cos(2π×审批月份/12)", "与sin共同标识年内月份；季节性需实测", "main"),
}


def markdown_table(headers, rows):
    def escape(value):
        return str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(escape(v) for v in row) + " |" for row in rows])


def feature_dictionary():
    rows = []
    spec = FEATURE_CANDIDATES["compact_history_27"]
    membership = {column: group for group, columns in GROUPS.items() for column in columns}
    for column in spec.all_columns:
        group = membership[column]
        if column in FEATURE_NOTES:
            definition, rationale, source = FEATURE_NOTES[column]
        elif group in ("mrt", "mall", "primary_school"):
            definition = "去重设施坐标后的最近球面距离（km）" if column.endswith("DIST_KM") else "球面距离1km内去重设施坐标数量"
            rationale = "最近距离描述可达性代理，计数描述附近选择数量；二者需联合验证"
            source = "block + facility"
        else:
            statistic = next(s for s in ("RECENCY_MONTHS", "LAST_MEAN", "COUNT", "MEAN") if column.endswith(s))
            definition = {"MEAN": "严格过去月份的全部记录均价", "LAST_MEAN": "严格过去最近有记录月份的组均价",
                "COUNT": "严格过去记录数；未知组为0", "RECENCY_MONTHS": "距过去最近有记录月份的月数"}[statistic]
            rationale = {"MEAN": "表示已观察的租金水平", "LAST_MEAN": "描述最近水平与长期水平的差异",
                "COUNT": "描述统计支持量，不把小样本均值视作同等可靠", "RECENCY_MONTHS": "描述历史陈旧程度；不能用未来标签更新"}[statistic]
            source = "fitting rents only"
        availability = ("只用拟合集且严格早于该行审批月的标签；验证/测试期间冻结"
            if source == "fitting rents only" else "所给静态设施快照；历史开通日期不完整"
            if source == "block + facility" else "主表审批月已知字段" if source == "main"
            else "所给楼栋静态快照；不作历史时点完整性承诺")
        rows.append({"feature": column, "group": NAMES[group],
            "type": "category" if column in spec.categorical else "numeric", "source": source,
            "definition": definition, "reason_to_test": rationale, "availability_and_limit": availability})
    return pd.DataFrame(rows)


def check_alignment():
    effects = pd.read_csv(EXPERIMENT_REPORTS / "features_ablation_evidence/fold_effects.csv")
    if len(effects) != 156 or not effects.groupby("validation_start").size().eq(39).all():
        raise ValueError("Single/group deletion grid is incomplete")
    reference = pd.read_csv(EXPERIMENT_REPORTS / "features_history_optimization/refined/fold_metrics.csv")
    baseline = effects.loc[effects.variant.eq("baseline")].set_index("validation_start")
    expected = reference.loc[reference.variant.eq("deduplicated_27")].set_index("validation_start")
    if len(baseline) != 4 or not np.allclose(baseline.rmse_sgd, expected.loc[baseline.index].rmse_sgd, atol=1e-5, rtol=0):
        raise ValueError("Feature deletion baselines no longer reproduce earlier compact results")
    common = pd.read_csv(EXPERIMENT_REPORTS / "comparison_same_27_features/model_metrics.csv")
    if len(common) != 10 or set(common.n_features) != {27} or set(common.n_fit) != {114842} or set(common.n_validation) != {35158}:
        raise ValueError("Common-feature model comparison is incomplete or misaligned")
    for name, key in (("features_ablation_evidence", "source_sha256"), ("comparison_same_27_features", "source_sha256"),
                      ("baseline_27_feature_submission", "sources")):
        recorded = json.loads((EXPERIMENT_REPORTS / name / ("summary.json" if name == "baseline_27_feature_submission" else "results.json")).read_text(encoding="utf-8"))
        for path, digest in recorded[key].items():
            filename = str(path).replace("\\", "/").rsplit("/", 1)[-1]
            current = DATA / filename
            if not current.exists():
                current = AUX / filename
            if sha256(current) != digest:
                raise ValueError(f"Source bytes differ from {name}: {filename}")
    return effects, common


def run():
    effects, common = check_alignment()
    OUT.mkdir(parents=True, exist_ok=True)
    dictionary = feature_dictionary()
    dictionary.to_csv(OUT / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    feature_summary = pd.read_csv(EXPERIMENT_REPORTS / "features_ablation_evidence/feature_effects.csv")
    selection = pd.read_csv(EXPERIMENT_REPORTS / "features_selection/fold_metrics.csv")
    model_metrics = pd.read_csv(EXPERIMENT_REPORTS / "comparison_model_search/model_metrics.csv")
    diagnostic = pd.read_csv(EXPERIMENT_REPORTS / "features_ablation_evidence/error_diagnostics.csv")
    final = json.loads((EXPERIMENT_REPORTS / "baseline_27_feature_submission/summary.json").read_text(encoding="utf-8"))
    group_rows = []
    group_summary = feature_summary.loc[feature_summary.kind.isin(["group", "joint"])]
    for _, row in group_summary.iterrows():
        group_rows.append([NAMES[row.variant.split("__", 1)[1]], f"{row.delta_pooled_rmse_sgd:+.2f}",
            f"{row.latest_delta_rmse_sgd:+.2f}", f"{int(row.annual_folds_worse_without)}/3",
            f"{row.stress_16m_delta_rmse_sgd:+.2f}"])
    group_table = markdown_table(["删掉的模型输入组", "三年度汇总ΔRMSE", "最新年度ΔRMSE", "删除后变差年度数", "16个月ΔRMSE"], group_rows)
    single = feature_summary.loc[feature_summary.kind.eq("single")]
    single.to_csv(OUT / "single_feature_effects.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(OUT / "group_feature_effects.csv", index=False, encoding="utf-8-sig")
    common.to_csv(OUT / "common_model_comparison.csv", index=False, encoding="utf-8-sig")
    single_table = markdown_table(["删除单列", "汇总ΔRMSE", "最新ΔRMSE", "删除后变差年度数"], [
        [r.removed_columns, f"{r.delta_pooled_rmse_sgd:+.2f}", f"{r.latest_delta_rmse_sgd:+.2f}",
         f"{int(r.annual_folds_worse_without)}/3"] for _, r in single.head(8).iterrows()])
    common_table = markdown_table(["模型", "相同27列 RMSE", "MAE", "相对LGB ΔRMSE", "月份区块敏感性区间"], [
        [NAMES[r.family], f"{r.rmse_sgd:.2f}", f"{r.mae_sgd:.2f}", f"{r.delta_rmse_vs_lightgbm:+.2f}",
         f"[{r.month_block_p025:+.2f}, {r.month_block_p975:+.2f}]"] for _, r in common.iterrows()])
    ladder_rows = []
    for variant, label in [("main_c", "基础7列"), ("spatial_29", "空间29列"),
                           ("compact_static", "精简静态19列"), ("compact_history", "静态＋历史31列")]:
        part = selection.loc[selection.variant.eq(variant)]
        scores = {r.validation_start: r.rmse_sgd for _, r in part.iterrows()}
        ladder_rows.append([label] + [f"{scores[k]:.2f}" for k in ("2022-04", "2023-04", "2024-04", "2023-12")])
    ladder_rows.append(["删除重复历史后27列"] + [f"{effects.loc[effects.variant.eq('baseline') & effects.validation_start.eq(k), 'rmse_sgd'].iloc[0]:.2f}"
        for k in ("2022-04", "2023-04", "2024-04", "2023-12")])
    ladder = markdown_table(["特征方案", "2022年度窗", "2023年度窗", "2024年度窗", "16个月压力窗"], ladder_rows)
    latest_diagnostic = diagnostic.loc[diagnostic.validation_start.eq("2024-04")]
    bands = latest_diagnostic.loc[latest_diagnostic.dimension.eq("actual_rent_band")]
    band_table = markdown_table(["真实租金段（仅诊断）", "样本数", "RMSE", "预测减真实偏差", "该窗平方误差占比"], [
        [r.group, int(r.n), f"{r.rmse_sgd:.2f}", f"{r.bias_sgd:+.2f}", f"{100*r.share_of_fold_squared_error:.1f}%"]
        for _, r in bands.iterrows()])
    low = bands.loc[bands.group.eq("<=2000")].iloc[0]
    low_note = (f"最新窗口中，真实租金≤2000的{int(low.n):,}条记录（占{100*low.n/35158:.1f}%）"
                f"贡献{100*low.share_of_fold_squared_error:.1f}%的平方误差，"
                "因此低租金记录的系统高估比继续堆设施字段更值得作为下一步研究假设。\n")
    top_model = model_metrics.loc[model_metrics.validation_start.eq("2024-04")].sort_values("rmse_sgd").iloc[0]
    header = """# HDB 月租预测：实验与结果报告（当前工作稿）

本报告围绕三个问题组织：已有字段怎样转成合理输入、每列/每组带来多少预测增量、
同一信息条件下传统 ML 与深度学习有何差别。主线使用比赛训练标签及所给辅助表，
不使用官方公开源中目标月份的租金。公开目标匹配作为独立的数据重合审计附录，不参与主线排名。

## 1. 问题、数据与验证

预测单位是单条租赁记录，目标为 SGD 月租金，主指标 RMSE，辅以 MAE 和预测减真实的平均偏差。
训练集 150,000 行（2021-01..2025-03），测试集 50,000 行（2025-03..2026-07）。
所给六类辅助数据包括楼栋、MRT、商场、学校、COE、股票；拥有数据不代表必须全部作为输入。
保留重复记录，常量 FURNISHED/FEE 不入模，不凭租金大小删除异常行。

三个年度选择窗为 2022-04..2023-03、2023-04..2024-03、2024-04..2025-03；
分别只用窗口起点之前的数据拟合。另报告 2023-12..2025-03 的16个月压力窗，
它与年度窗重叠，不计入三年度汇总。汇总 RMSE 由各折平方误差按样本量合并后开平方，
不直接平均 RMSE。历史统计在拟合截止冻结，验证中不接收真实租金更新。

全部标签月份此前都已经看过，这些是方案选择证据，不是新的独立测试。
静态设施快照缺少完整开通日期，因此空间特征是比赛所给信息条件下的代理，
不能声称已实现所有字段的历史可获得性审计。

## 2. 特征工程的假设与取舍

默认输入为27列：房屋5、时间3、区位5、MRT/商场/小学各2、两种粒度的过去租金各4。
完整字段定义、来源、构造原因和时间边界见 [字段字典](output/experiment_runs/foundation_feature_review/feature_dictionary.csv)，
实现与用法见 [特征工程说明](docs/methods/baseline_features.md)。

房型统一写法；地址只用于楼栋连接与历史分组。楼栋最高楼层不等于出租单元楼层，
审批年减租约起始年只是楼龄代理。球面距离是可达性代理，不能解释成通勤时间或因果效应。
最近设施距离与1km数量分别描述最近选择和周边选择；是否有增量由成组删除实验判断。

历史租金按“楼栋×户型”和“规划区×户型”形成过去均价、记录数、最近均价和距最近记录月数。
每条拟合行只查询严格更早月份；验证行只查询拟合集历史。未知组为缺失/计数0，
模型可利用粗粒度历史。类别词表、数值变换仅在拟合集学习。

原31列中 SIGNATURE 历史与 BLOCK_FLAT 历史在当前数据上逐行完全重复，故删除4列。
这一结构性审计比按单次重要性排名删列更有依据，但换数据后需重新核对。

## 3. 特征方案从基础到历史

下面固定同一 LightGBM 参数及相同拟合/验证行，400棵树、31叶节点、学习率0.05、种子42。
相邻方案包含的变化不止一个字段，因此这里是方案比较，不把差异全部归因于某一个变量。

"""
    sections = [header, ladder, """

更丰富的40列静态和52列历史方案没有稳定胜过精简方案；变量更多并不保证收益。
进一步的历史窗口、衰减、平滑、多半径、地址类别和市场基准分解已做100次固定模型拟合，
见 [深入特征实验](docs/methods/history_feature_optimization.md)。溢价特征仅有小幅不稳定收益，
市场趋势残差分解改善部分上涨窗口，但在转折期可能变差，保留为独立研究路线。

COE/股票已有四窗对照，组合在该历史月度更新协议下4/4变差。
这是不同协议下的负结果，不能和本报告固定预测起点的分数直接混排；未来各月的真实宏观观测
在批量预测起点尚未知，故不放进默认输入。见 [经济消融](output/experiment_runs/features_economic_ablation/report.md)。

## 4. 每个输入/每组的 RMSE 增量

新增156次训练：完整27列＋逐列删除27个＋成组删除8个＋联合删除3个，共39配置×4窗。
每次删除都重新训练，参数、行、时间切点固定。定义：

```text
ΔRMSE = 删除指定输入并重训后的 RMSE − 完整27列模型的 RMSE
正值：删除后变差；负值：删除后反而变好。
```

单列删除衡量“其他输入仍在时的边际贡献”。保留的历史编码仍可能携带房型/位置/时间信息，
因此删除面积、位置或户型列，不代表移除了关于房屋规模或区位的全部信息。
贡献不是可相加的解释份额，也不是变量的因果效应；原始源级干预需要重新设计派生特征。

### 成组和联合删除

""", group_table, """

### 单列删除中汇总影响最大的八列

""", single_table, """

全部27列见 [逐列结果](output/experiment_runs/foundation_feature_review/single_feature_effects.csv)，含每列最新窗口与压力窗差值。
完整按折结果见 [fold_effects.csv](output/experiment_runs/features_ablation_evidence/fold_effects.csv)。
![成组删除](output/experiment_runs/features_ablation_evidence/group_deletion.png)

时间与历史特征可相互替代部分信号，应同时看联合删除而非只看单列排名。
某列删除后略好，不等于它在所有模型中无价值，也不据此把各单列的“改善”组合成新最优方案。
本次不自动改变27列默认方案，避免继续用已看过的外层标签逐步挑选。

最新窗附带配对3个月连续区块重采样，2000次，报告2.5/97.5百分位。
它仅反映固定预测下的月份敏感性，12个月的支持有限，不包括训练随机性或未来市场不确定性，
没有多重比较校正，不能当作正式显著性证明。

## 5. 模型层面：同信息比较与整条管线比较

选择这些模型是为检验不同建模假设，而不是凑架构数量：

| 对照 | 本项目要检验的问题 |
| --- | --- |
| LightGBM、CatBoost、XGBoost、HistGradientBoosting | 对混合数值/类别表格的非线性树集成，是主要预测基准 |
| ExtraTrees | 随机树集成能否与逐步提升树形成有效对照 |
| Embedding MLP / ResNet | 类别向量与连续输入联合训练，在相同信息下是否优于树 |
| FT-Transformer 96/192 | 特征token的注意力交互以及更大宽度是否提供额外收益 |
| TabNet | 顺序注意力式特征选择，在本表格与有限预算下是否有效 |

方法出处：[LightGBM原论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)、
[FT-Transformer与表格深度基线](https://arxiv.org/abs/2106.11959)、
[TabNet原论文](https://arxiv.org/abs/1908.07442)。本项目的数值结果来自下面的本地实验，
不以论文在其他数据集上的排名推断本数据的排名。

### 同27列、同114,842条拟合行、同35,158条验证行

只从原搜索中 compact 配置的内部2023-10..2024-03分数选择参数和轮数；
再用2024-04之前全部记录重训。没有残差偏移或历史预热删行。
完全一致的已有外层拟合可复用，其他模型补训。预处理按模型适配，但原始27个输入相同。

""", common_table, """

这一表更接近架构对照，仍有搜索预算不等、预处理不同、只跑一个种子的限制。
本表LightGBM按内部验证选择参数与轮数，和第4节固定400树的468.49基线配置不同；
同样27列并不代表超参数相同，因此两处LightGBM分数不同不构成结果冲突。
小差异不能作为模型家族普遍优劣的证明。配对区块区间的含义与上节相同。

### 各模型允许选择自身特征/目标表示的完整管线

原模型搜索包含30个内部候选配置及10种模型规格，允许精简、溢价、地址和市场残差版本。
""", f"最新年度观察最小 RMSE 为 **{top_model.rmse_sgd:.2f}**（{NAMES[top_model.family]}）。\n",
        """这张表比较完整管线，不能将收益全归因于架构。详见 [原模型比较](docs/methods/model_comparison.md)。
早期及16个月复核仅覆盖部分树模型，不把神经网络的最新一年结果泛化为全时期结论。

## 6. 误差从哪里来

下面使用固定27列 LightGBM 的最新窗诊断，与第4节删除基线一致。
偏差为预测减真实；按真实租金分组是事后诊断，不能在预测时以未知租金决定模型或修正量。

""", band_table, low_note, """

低租金组高估、高租金组低估的现象要结合回归到均值、缺失房屋细节和真实市场变化分析，
不能直接推成因果机制，也不能只根据目标分组误差做泄漏式校准。
进一步的按月份、预测跨度、Town、户型、历史支持量和历史陈旧程度见
[error_diagnostics.csv](output/experiment_runs/features_ablation_evidence/error_diagnostics.csv)。
表中同时记录样本数和平方误差占比，避免只关注小样本高RMSE组。
按月的观察用于定位问题，不新增任何未来标签特征。

## 7. 最终复现与信息边界

为了提供最简单可复现的正式预测路径，保留固定27列、400树 LightGBM 作为默认参考管线，
其最新窗 RMSE 为468.49，不将它称作465.66的搜索最优融合。
命令 `python -m analysis.main canonical-submit` 使用全部150,000条标签拟合，
保存包含词表和历史统计的完整模型，再重载核对，生成50,000行测试预测。
""", f"本次模型重载最大预测差为 {final['model_reload_max_difference']:.2g}；预测行数 {final['test_rows']:,}。\n",
        """输出在 `output/submissions/baselines/baseline_27_features/`，不自动上传；检查摘要在
[baseline_27_feature_submission/summary.json](output/experiment_runs/baseline_27_feature_submission/summary.json)。
测试2025-03的重叠月份历史排除同月标签，但最终基础模型仍以全部已知训练标签拟合，
因此不宣称这380条测试记录获得了完全事前的模型拟合。

公开原始租金匹配已独立放在 `experiment/public-rent-lookup` 分支。
其约180是公开目标重建误差、约173是候选方差估算，不能并入普通预测模型排名。
没有证据确定其他队伍使用了什么方法；是否允许目标恢复应依据课程/比赛要求。

## 8. 当前完成度与剩余工作

| 项目 | 当前状态 |
| --- | --- |
| 数据质量、连接率和主/辅助 EDA | 已有报告；原始数据哈希保护 |
| 明确的默认特征与每列构造理由 | 已完成27列字典和统一实现 |
| 特征组、历史表示与冗余审计 | 已有多窗口比较及重复列删除复验 |
| 每列/每组实际 RMSE 增量 | 本轮补齐156次重训与按折结果 |
| 同特征 ML/DL 比较 | 本轮补齐10规格共同输入对照 |
| 不同特征适配的模型搜索 | 已完成30个内部配置；单独解释协议 |
| 误差分析与模型复现导出 | 本轮补齐历史支持/陈旧度/跨度诊断及最终重载核对 |
| 神经网络多随机种子、更多前向窗口 | 尚未补齐，不夸大当前架构排名；可作为有限后续实验 |
| 所有设施历史开通时点 | 仍缺数据，保留限制或做无设施敏感性对照 |
| 真正未看过的独立测试/实际Kaggle成绩 | 尚无；当前外层不能重新称为未触碰测试 |
| 最终课程格式、页数、分工与参考文献排版 | 需按课程原始rubric定稿；本仓库未提供完整rubric |

不为增加复杂度额外引入潜变量模型。只有误差诊断形成新假设、能说明输入来源和验证方式，
才继续扩展。下一步最值得做的是围绕现有证据定稿报告与答辩，而非无限堆变量或架构。

## 9. 复跑入口

```powershell
# 每列/每组删除实验与误差诊断
python -m analysis.main feature-evidence
# 同27列模型对照：需先具备原model-search的内部trial结果
python -m analysis.main common-models
# 简单默认管线全量重训、模型重载与本地提交CSV
python -m analysis.main canonical-submit
# 只读取已有实验结果，重新生成本报告、字段字典和索引
python -m analysis.main project-review
```

环境与原模型搜索复跑见 [模型说明](docs/methods/model_comparison.md)。历史实验入口保留在
[运行指南](docs/guides/running_experiments.md)，不需要为读当前主线逐个运行。
机器可读项目缺口与证据路径见 [output/experiment_runs/foundation_feature_review/README.md](output/experiment_runs/foundation_feature_review/README.md)。
"""]
    from analysis.report_bundle import rewrite_links
    report_path = OUT / "BASELINE_REPORT.md"
    report_path.write_text(rewrite_links("\n".join(sections), ROOT / "project_report.md", report_path), encoding="utf-8")
    manifest_paths = [EXPERIMENT_REPORTS / p for p in (
        "features_ablation_evidence/results.json", "features_ablation_evidence/feature_effects.csv",
        "comparison_same_27_features/results.json", "comparison_same_27_features/model_metrics.csv",
        "baseline_27_feature_submission/summary.json", "comparison_model_search/model_metrics.csv")]
    manifest = {"report": report_path.relative_to(ROOT).as_posix(), "review_scope": "Normal past-information forecasting; public-label lookup excluded",
        "feature_deletion_fits": len(effects), "common_model_specs": len(common), "feature_columns": len(dictionary),
        "baseline_alignment_passed": True, "artifacts": {
            p.relative_to(ROOT).as_posix(): sha256(p) for p in manifest_paths},
        "remaining_limitations": ["One seed per DL model; limited repeated temporal DL comparison",
            "Previously examined outer labels", "Incomplete historical facility openings",
            "No independent test score or full course rubric"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index = """# 当前项目证据入口

先阅读 [当前实验报告](../../project_report.md)，再查看需要的证据。

| 问题 | 对应产物 |
| --- | --- |
| 每个变量为什么构造、来自哪里、何时可用？ | [feature_dictionary.csv](feature_dictionary.csv) |
| 删除单列后RMSE如何变化？ | [single_feature_effects.csv](single_feature_effects.csv) |
| 相关字段一起删除如何变化？ | [group_feature_effects.csv](group_feature_effects.csv) |
| 同27列下模型差别？ | [common_model_comparison.csv](common_model_comparison.csv) |
| 时间窗、参数、重训数和不确定性口径？ | [features_ablation_evidence/results.json](../experiments/features_ablation_evidence/results.json) |
| 每个时间窗的具体数值？ | [fold_effects.csv](../experiments/features_ablation_evidence/fold_effects.csv) |
| 哪些样本贡献了误差？ | [error_diagnostics.csv](../experiments/features_ablation_evidence/error_diagnostics.csv) |
| 同特征模型如何选参数和复用？ | [comparison_same_27_features/results.json](../experiments/comparison_same_27_features/results.json) |
| 默认管线能否重训、保存、预测？ | [baseline_27_feature_submission/summary.json](../experiments/baseline_27_feature_submission/summary.json) |
| 缺口与一致性检查？ | [manifest.json](manifest.json)，报告第8节 |

正ΔRMSE表示删除后变差；负值表示在该模型/配置下删除后变好。贡献不能相加，不能解释为因果。
月份区块百分位只反映固定预测的月份敏感性，不是完整模型不确定性。
同输入对比与各自特征适配的模型搜索是两种问题，不能把其结果拼成一个架构排名。

这里的CSV和报告由 `python -m analysis.main project-review` 读取已有结果生成，不会重新训练。
原始输入不变；公开目标匹配与主预测流程分别解释。历史报告继续保留，不作为当前协议的混合榜单。
"""
    (OUT / "README.md").write_text(rewrite_links(index, ROOT / "reports/foundation_feature_review/README.md", OUT / "README.md"), encoding="utf-8")
    print("Updated output/experiment_runs/foundation_feature_review/ from verified saved results. Run experiment-refresh to update reading copies.", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return run()


if __name__ == "__main__":
    main()
