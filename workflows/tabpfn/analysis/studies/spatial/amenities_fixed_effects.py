"""Descriptive, building-clustered amenity contrasts on one common sample."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.studies.spatial.mrt_fixed_effects import fit, t_critical, test as test_fit
from analysis.studies.spatial.common import OUTPUT, SOURCE, load_spatial_training
from analysis.pipeline.data import sha256


OUT = OUTPUT / "amenities_adjusted"
LABELS = ["[0,0.3]", "(0.3,0.6]", "(0.6,1.0]", "(1.0,1.5]", "(1.5,2.0]", "(2.0,inf)"]
AMENITIES = {
    "open_mrt": ("NEAREST_OPEN_MRT_KM", "已开放 MRT"),
    "mall": ("NEAREST_MALL_KM", "商场"),
    "primary_school": ("NEAREST_PRIMARY_SCHOOL_KM", "小学"),
}
NUMERIC = ["MONTHLY_RENT", "FLOOR_AREA_SQM", "BUILDING_AGE_YEARS", "MAX_FLOOR", "DISTANCE_TO_CBD_KM"] + [v[0] for v in AMENITIES.values()]
REQUIRED = NUMERIC + ["APPROVAL_MONTH_INDEX", "FLAT_TYPE_CLEAN", "TOWN_CLEAN", "STREET_CLEAN", "BLOCK_CLEAN"]


def distance_bins(values):
    """Six right-closed distance bins, including zero in the first bin."""
    return pd.cut(values, [0, .3, .6, 1, 1.5, 2, np.inf],
                  right=True, include_lowest=True, labels=LABELS)


def zscore(frame, column):
    values = frame[column].to_numpy(float)
    scale = values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError(f"Cannot standardize {column}")
    return (values - values.mean()) / scale


def test():
    test_fit()
    observed = distance_bins(np.array([0, .3, .300001, .6, 1, 1.5, 2, 2.000001]))
    assert observed.astype(str).tolist() == [
        LABELS[0], LABELS[0], LABELS[1], LABELS[1],
        LABELS[2], LABELS[3], LABELS[4], LABELS[5],
    ]
    assert pd.isna(distance_bins(np.array([-0.01]))[0])
    dummy = pd.get_dummies(distance_bins(np.array([.1, .4, .8, 1.2, 1.8, 2.2])), dtype=float)
    assert list(dummy.columns) == LABELS
    assert dummy.iloc[:, 1:].shape == (6, 5)
    assert np.array_equal(dummy.iloc[0, 1:].to_numpy(), np.zeros(5))
    print("Amenity bin boundary and baseline indexing tests passed")


def main(frame=None):
    source_hash_before = sha256(SOURCE)
    original = load_spatial_training() if frame is None else frame
    missing = {col: int(original[col].isna().sum()) for col in REQUIRED}
    mask = original[REQUIRED].notna().all(axis=1)
    mask &= np.isfinite(original[NUMERIC].to_numpy(float)).all(axis=1)
    data = original.loc[mask].copy()
    if len(data) == 0:
        raise ValueError("No common complete rows")
    bins = {name: distance_bins(data[column]) for name, (column, _) in AMENITIES.items()}
    if any(b.isna().any() for b in bins.values()):
        raise ValueError("Negative or invalid amenity distance")
    month_type = data.APPROVAL_MONTH_INDEX.astype(str) + "|" + data.FLAT_TYPE_CLEAN
    groups = [pd.factorize(month_type)[0], pd.factorize(data.TOWN_CLEAN)[0]]
    building = data.TOWN_CLEAN + "|" + data.STREET_CLEAN + "|" + data.BLOCK_CLEAN
    cluster = pd.factorize(building)[0]
    y = data.MONTHLY_RENT.to_numpy(float)
    distances = {name: pd.get_dummies(b, dtype=float).iloc[:, 1:].to_numpy() for name, b in bins.items()}
    area = zscore(data, "FLOOR_AREA_SQM")
    age = zscore(data, "BUILDING_AGE_YEARS")
    controls = np.column_stack([area, area ** 2, age, age ** 2, zscore(data, "MAX_FLOOR")])
    cbd = zscore(data, "DISTANCE_TO_CBD_KM")
    specifications = {}
    for name in AMENITIES:
        specifications[f"separate_{name}"] = (np.column_stack([distances[name], controls]), {name: 0})
    joint = np.column_stack([*(distances[name] for name in AMENITIES), controls])
    offsets = {name: 5 * i for i, name in enumerate(AMENITIES)}
    specifications["joint"] = (joint, offsets)
    specifications["joint_cbd"] = (np.column_stack([joint, cbd]), offsets)

    results = {}
    for spec, (x, index) in specifications.items():
        beta, covariance, info = fit(y, x, groups, cluster)
        if info["slope_rank"] != x.shape[1]:
            raise AssertionError(f"Rank deficiency in {spec}")
        results[spec] = (beta, covariance, info, index)
    critical = t_critical(len(np.unique(cluster)) - 1)
    records = []
    for amenity in AMENITIES:
        b = bins[amenity]
        raw_baseline = y[np.asarray(b == LABELS[0])].mean()
        for spec in [f"separate_{amenity}", "joint", "joint_cbd"]:
            beta, covariance, _, index = results[spec]
            start = index[amenity]
            for i, label in enumerate(LABELS):
                pick = np.asarray(b == label)
                n_buildings = int(building[pick].nunique())
                effect = 0.0 if i == 0 else float(beta[start + i - 1])
                se = 0.0 if i == 0 else float(np.sqrt(covariance[start + i - 1, start + i - 1]))
                records.append({
                    "amenity": amenity, "specification": spec, "distance_bin_km": label,
                    "n_rows": int(pick.sum()), "n_buildings": n_buildings,
                    "sparse_buildings_lt30": n_buildings < 30,
                    "raw_mean_rent_sgd": float(y[pick].mean()),
                    "raw_mean_difference_sgd": float(y[pick].mean() - raw_baseline),
                    "adjusted_difference_sgd": effect, "cluster_se_sgd": se,
                    "ci95_low_sgd": effect - critical * se,
                    "ci95_high_sgd": effect + critical * se,
                })
    table = pd.DataFrame(records)
    source_hash_after = sha256(SOURCE)
    if source_hash_after != source_hash_before:
        raise RuntimeError("Source file changed during analysis")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "amenity_bin_effects.csv", index=False)
    manifest = {
        "source": "data/train.csv", "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after, "source_rows": len(original),
        "common_complete_rows": len(data), "excluded_rows": int((~mask).sum()),
        "missing_by_required_column": missing, "unique_buildings": int(building.nunique()),
        "month_type_levels": int(month_type.nunique()), "town_levels": int(data.TOWN_CLEAN.nunique()),
        "models": {key: value[2] for key, value in results.items()},
        "bin_boundaries": "right-closed: [0,0.3], (0.3,0.6], (0.6,1], (1,1.5], (1.5,2], (2,infinity)",
        "reference_group": "0–0.3 km within each amenity",
        "uncertainty": "95% pointwise t interval; df = independent building clusters - 1; t critical via third-order large-df Cornish-Fisher approximation; building-cluster sandwich with finite-sample correction",
        "sparse_flag": "fewer than 30 distinct town|street|block building keys",
        "control_columns": ["approval month × flat type fixed effects", "town fixed effects", "area z and squared", "building age z and squared", "max building floor z"],
        "cbd_sensitivity": "joint_cbd adds standardized haversine distance to approximate Raffles Place",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
    english_titles = {"open_mrt": "Open MRT", "mall": "Shopping mall", "primary_school": "Primary school"}
    for ax, (name, _) in zip(axes, AMENITIES.items()):
        subset = table[(table.amenity == name) & (table.specification == "joint_cbd")]
        x = np.arange(6)
        raw = subset.raw_mean_difference_sgd.to_numpy().copy()
        adjusted = subset.adjusted_difference_sgd.to_numpy().copy()
        error_low = (subset.adjusted_difference_sgd - subset.ci95_low_sgd).to_numpy().copy()
        error_high = (subset.ci95_high_sgd - subset.adjusted_difference_sgd).to_numpy().copy()
        # This school bin is only three buildings; omit its extreme bars so
        # the supported school bins remain readable. The table retains it.
        if name == "primary_school" and subset.iloc[-1].sparse_buildings_lt30:
            raw[-1] = adjusted[-1] = error_low[-1] = error_high[-1] = np.nan
        ax.bar(x - .19, raw, width=.36, label="Raw mean")
        ax.bar(x + .19, adjusted, width=.36,
               yerr=[error_low, error_high],
               capsize=3, label="Joint + CBD adjusted")
        for i, sparse in enumerate(subset.sparse_buildings_lt30):
            if sparse:
                ax.text(i, 0, "3 buildings;\nsee table", ha="center", va="bottom", fontsize=8, color="darkred")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks(x, LABELS, rotation=35, ha="right")
        ax.set_title(english_titles[name])
        ax.set_xlabel("Nearest distance (km)")
        ax.set_ylabel("Monthly rent difference vs 0–0.3 km (SGD)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Amenity distance and rent: raw versus joint + CBD adjusted (95% building-cluster CI)")
    fig.tight_layout()
    fig.savefig(OUT / "amenity_raw_vs_adjusted.png", dpi=170)
    plt.close(fig)

    def markdown_subset(name):
        wide = table[table.amenity == name].pivot(index="distance_bin_km", columns="specification",
                                                      values="adjusted_difference_sgd").reindex(LABELS)
        joint_rows = table[(table.amenity == name) & (table.specification == "joint_cbd")].set_index("distance_bin_km").reindex(LABELS)
        lines = ["| 距离 km | 行/楼栋 | 原始均值差 | 单独调整 | 联合调整 | 联合+CBD [95%区间] |",
                 "|---|---:|---:|---:|---:|---:|"]
        for label in LABELS:
            row = joint_rows.loc[label]
            marker = "*" if row.sparse_buildings_lt30 else ""
            ci = f"{row.adjusted_difference_sgd:+.0f} [{row.ci95_low_sgd:+.0f}, {row.ci95_high_sgd:+.0f}]"
            lines.append(f"| {label}{marker} | {int(row.n_rows):,}/{int(row.n_buildings):,} | {row.raw_mean_difference_sgd:+.0f} | {wide.loc[label, 'separate_'+name]:+.0f} | {wide.loc[label, 'joint']:+.0f} | {ci} |")
        return "\n".join(lines)

    report = f"""# 商场、MRT、小学距离与 HDB 月租：控制变量分析

共同完整样本 {len(data):,} 笔、{building.nunique():,} 栋，来自 {len(original):,} 笔训练数据；排除 {(~mask).sum():,} 笔。三类设施与所有模型都使用同一批记录，因此模型间系数变化不是样本更换造成。源数据 SHA-256、各列缺失、固定效应和设计秩见 [分析清单](analysis_manifest.json)。下表金额均为 SGD/月，相对于同一设施 0–0.3 km 组；“原始均值差”是均值差，不是中位数。

控制：日历月份 × 规范化户型固定效应、town 固定效应、面积（标准化及平方）、楼龄（标准化及平方）、楼栋最高楼层。“单独调整”每次只加入一种设施距离；“联合调整”同时加入三种；“联合+CBD”再加入至 CBD 参考点的直线距离。模型在固定效应投影后，所有斜率列均满秩。95% 区间以 town+street+block 楼栋聚类，使用有限样本修正及楼栋数减一的 t 临界值（大自由度三阶近似）；区间是逐项区间，不能把多个组同时显著当作整体检验。

**概括。** 在联合控制三类设施和 CBD 距离后，离已开放 MRT 0.6–1 km 的楼栋比 0–0.3 km 组平均低约 S$86/月（95% 区间 −100 至 −72），1–1.5 km 组低约 S$139/月。商场 0.6–1 km 组低约 S$40/月（−56 至 −24），1–1.5 km 组低约 S$87/月。小学 0.3–0.6 km 组约低 S$4/月（−13 至 +5），与基准组未能区分；0.6–1 km 组约低 S$20/月（−34 至 −6），幅度明显小于相应 MRT、商场组。MRT 和商场的距离梯度较清楚，小学梯度不规则，不能概括为“所有设施越近租金越高”。

单独调整与联合调整的差距也有信息：例如 1–1.5 km，MRT 从 −168 变成 −137，商场从 −126 变成 −89，小学从 −37 变成 −21 SGD/月；加入 CBD 后分别为 −139、−87、−26。三种距离共享部分区位信息，所以单独模型不能简单把全部差异归给那一种设施。这些模型只是条件均值回归，并未要求每个距离组合在所有 town、月份、户型中都有充分重叠；稀疏组合的外推风险仍在。

## 已开放 MRT

{markdown_subset('open_mrt')}

## 商场

{markdown_subset('mall')}

## 小学

{markdown_subset('primary_school')}

![设施距离的原始均值差和联合控制差异](amenity_raw_vs_adjusted.png)

图中小学 2 km 以上组因仅 3 栋而省略柱子，完整数值保留在表中。

表格中 * 表示该距离组少于 30 栋楼；尤其小学 2 km 以上组只有 57 笔、3 栋，+425 的系数及其形式上的置信区间都高度依赖这 3 栋，不能把区间很窄误认成可靠证据，更不能据此推断一般趋势。小学 1.5–2 km 组也只有 43 栋，解读须谨慎。控制后若系数方向或大小改变，说明三类设施距离、CBD 中心性和住房条件的分布相互关联。区间包含 0 的组，在当前模型和标准误下没有足够证据区分于 0–0.3 km 基准组；区间不包含 0 仅说明该特定对比在该模型下有统计差异。

此处“学校”只指辅助表识别的最近小学（含 p1 标记的混合学校），没有涵盖所有教育设施，也没有衡量学校质量；距离是楼栋至设施的球面直线距离，不是步行路程。设施清单是当前快照，没有按租赁月份还原开业或启用历史。控制了 town 与上述房屋因素后，town 内微区位、装修、交通服务质量及其他未观测因素仍可能影响租金与设施距离；楼栋聚类只修正重复记录的不确定性。结果是条件关联，不能解释为设施导致的租金变化，也不代表加入预测模型后一定提高预测准确率。

复现：`python -m analysis.main spatial-controls`。完整逐行结果见 [CSV](amenity_bin_effects.csv)。
"""
    (OUT / "amenities_adjusted_summary_zh.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({"common_complete_rows": len(data), "unique_buildings": int(building.nunique()), "source_hash_unchanged": source_hash_before == source_hash_after, "models": manifest["models"]}, indent=2))


if __name__ == "__main__":
    import sys
    test() if "--test" in sys.argv else main()
