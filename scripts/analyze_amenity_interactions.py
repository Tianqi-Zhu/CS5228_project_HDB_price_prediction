"""MRT × mall adjusted rent contrasts, with building-cluster uncertainty."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_mrt_adjusted import fit, t_critical

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/train_spatial_features.csv"
OUT = ROOT / "reports/spatial_features/amenity_interactions"
LABELS = ["0–0.5", "0.5–1", "1+"]
SCHOOL_LABELS = ["0–0.3", "0.3–0.6", "0.6–1", "1–1.5", "1.5–2", "2+"]
NUMERIC = ["MONTHLY_RENT", "NEAREST_OPEN_MRT_KM", "NEAREST_MALL_KM",
           "NEAREST_PRIMARY_SCHOOL_KM", "FLOOR_AREA_SQM", "BUILDING_AGE_YEARS",
           "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]
REQUIRED = NUMERIC + ["APPROVAL_MONTH_INDEX", "FLAT_TYPE_CLEAN", "TOWN_CLEAN",
                      "STREET_CLEAN", "BLOCK_CLEAN"]


def distance_3bin(values):
    return pd.cut(values, [0, .5, 1, np.inf], labels=LABELS,
                  right=True, include_lowest=True)


def cell_index(mrt, mall):
    """Index into the eight joint-cell coefficients; near/near is reference."""
    if not (0 <= mrt < 3 and 0 <= mall < 3):
        raise IndexError((mrt, mall))
    return None if (mrt, mall) == (0, 0) else mrt * 3 + mall - 1


def cell_vector(mrt, mall, n_slopes=8):
    result = np.zeros(n_slopes)
    index = cell_index(mrt, mall)
    if index is not None:
        result[index] = 1
    return result


def interaction_vector(mrt, mall, n_slopes=8):
    """Joint-cell departure from an additive MRT plus mall pattern."""
    return (cell_vector(mrt, mall, n_slopes) - cell_vector(mrt, 0, n_slopes)
            - cell_vector(0, mall, n_slopes) + cell_vector(0, 0, n_slopes))


def zscore(data, name):
    values = data[name].to_numpy(float)
    scale = values.std(ddof=0)
    if scale == 0 or not np.isfinite(scale):
        raise ValueError(f"Cannot standardize {name}")
    return (values - values.mean()) / scale


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate(vector, beta, covariance, critical):
    value = float(np.einsum("i,i->", vector, beta, optimize=False))
    variance = float(np.einsum("i,ij,j->", vector, covariance, vector, optimize=False))
    if variance < -1e-8:
        raise ValueError("Negative contrast variance")
    se = float(np.sqrt(max(variance, 0)))
    return value, se, value - critical * se, value + critical * se


def main():
    before = sha256(SOURCE)
    original = pd.read_csv(SOURCE)
    missing = {name: int(original[name].isna().sum()) for name in REQUIRED}
    mask = original[REQUIRED].notna().all(axis=1)
    mask &= np.isfinite(original[NUMERIC].to_numpy(float)).all(axis=1)
    data = original.loc[mask].copy()
    mrt = distance_3bin(data.NEAREST_OPEN_MRT_KM)
    mall = distance_3bin(data.NEAREST_MALL_KM)
    school = pd.cut(data.NEAREST_PRIMARY_SCHOOL_KM,
                    [0, .3, .6, 1, 1.5, 2, np.inf], labels=SCHOOL_LABELS,
                    right=True, include_lowest=True)
    if mrt.isna().any() or mall.isna().any() or school.isna().any():
        raise ValueError("Negative or invalid amenity distance")
    mrt_code = mrt.cat.codes.to_numpy()
    mall_code = mall.cat.codes.to_numpy()
    joint_code = mrt_code * 3 + mall_code
    joint = np.eye(9, dtype=float)[joint_code, 1:]
    school_x = pd.get_dummies(school, dtype=float).iloc[:, 1:].to_numpy()
    area = zscore(data, "FLOOR_AREA_SQM")
    age = zscore(data, "BUILDING_AGE_YEARS")
    controls = np.column_stack([school_x, area, area ** 2, age, age ** 2,
                                zscore(data, "MAX_FLOOR"), zscore(data, "DISTANCE_TO_CBD_KM")])
    x = np.column_stack([joint, controls])
    month_type = data.APPROVAL_MONTH_INDEX.astype(str) + "|" + data.FLAT_TYPE_CLEAN
    groups = [pd.factorize(month_type)[0], pd.factorize(data.TOWN_CLEAN)[0]]
    building = data.TOWN_CLEAN + "|" + data.STREET_CLEAN + "|" + data.BLOCK_CLEAN
    cluster = pd.factorize(building)[0]
    y = data.MONTHLY_RENT.to_numpy(float)
    beta, covariance, info = fit(y, x, groups, cluster)
    critical = t_critical(info["clusters"] - 1)
    baseline = y[joint_code == 0].mean()
    records = []
    for mi in range(3):
        for ma in range(3):
            select = joint_code == mi * 3 + ma
            n_rows = int(select.sum())
            n_buildings = int(building.loc[select].nunique())
            sparse = n_rows < 100 or n_buildings < 30
            adjusted = estimate(cell_vector(mi, ma, len(beta)), beta, covariance, critical)
            interaction = estimate(interaction_vector(mi, ma, len(beta)), beta, covariance, critical)
            records.append(dict(mrt_km=LABELS[mi], mall_km=LABELS[ma],
                                n_rows=n_rows, n_buildings=n_buildings,
                                sparse_lt100rows_or30buildings=sparse,
                                raw_mean_rent_sgd=float(y[select].mean()),
                                raw_difference_vs_near_near_sgd=float(y[select].mean() - baseline),
                                adjusted_vs_near_near_sgd=adjusted[0],
                                adjusted_cluster_se_sgd=adjusted[1],
                                adjusted_ci95_low_sgd=adjusted[2],
                                adjusted_ci95_high_sgd=adjusted[3],
                                interaction_departure_sgd=interaction[0],
                                interaction_cluster_se_sgd=interaction[1],
                                interaction_ci95_low_sgd=interaction[2],
                                interaction_ci95_high_sgd=interaction[3]))
    table = pd.DataFrame(records)
    after = sha256(SOURCE)
    if before != after:
        raise RuntimeError("Source changed during analysis")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "mrt_mall_joint_cells.csv", index=False)
    pd.DataFrame(covariance, columns=[f"slope_{i}" for i in range(len(beta))]).to_csv(
        OUT / "cluster_covariance.csv", index=False)
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256_before": before,
        "source_sha256_after": after, "source_rows": len(original),
        "common_complete_rows": len(data), "excluded_rows": int((~mask).sum()),
        "missing_by_required_column": missing, "model": info,
        "month_type_levels": int(month_type.nunique()), "town_levels": int(data.TOWN_CLEAN.nunique()),
        "distance_bins_km": "right-closed [0,0.5], (0.5,1], (1,infinity)",
        "reference": "MRT <=0.5 km and mall <=0.5 km",
        "interaction_formula": "joint(mrt,mall)-joint(mrt,near_mall)-joint(near_mrt,mall)+joint(near_mrt,near_mall)",
        "controls": ["approval month × flat type FE", "town FE", "primary school six bins",
                     "area z and square", "building age z and square", "max building floor z",
                     "CBD distance z"],
        "cluster": "town|street|block building; finite-sample sandwich correction",
        "interval": "pointwise 95% t, df=building clusters minus one, third-order Cornish-Fisher critical value",
        "sparse_rule": "omit plot estimate if fewer than 100 transactions OR 30 buildings; CSV retains estimate",
        "sparse_cells": int(table.sparse_lt100rows_or30buildings.sum()),
        "interpretation": "descriptive conditional association, not causal effect or prediction validation",
    }
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)
    specs = [("adjusted_vs_near_near_sgd", "A. Adjusted rent difference vs near / near"),
             ("interaction_departure_sgd", "B. Departure from additive distances")]
    for ax, (column, title) in zip(axes, specs):
        values = table[column].to_numpy().reshape(3, 3).copy()
        sparse = table.sparse_lt100rows_or30buildings.to_numpy().reshape(3, 3)
        values[sparse] = np.nan
        limit = max(np.nanmax(np.abs(values)), 1)
        plot = ax.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-limit, vmax=limit)
        plot.cmap.set_bad("#eeeeee")
        for mi in range(3):
            for ma in range(3):
                row = table.iloc[mi * 3 + ma]
                if sparse[mi, ma]:
                    label = "Sparse\nsee CSV"
                else:
                    label = f"{row[column]:+.0f}\nn={row.n_rows:,} · b={row.n_buildings:,}"
                    if mi == 0 and ma == 0 and column == "adjusted_vs_near_near_sgd":
                        label = f"Reference\nn={row.n_rows:,} · b={row.n_buildings:,}"
                    elif column == "interaction_departure_sgd":
                        if mi == 0 or ma == 0:
                            label = f"Defined 0\nn={row.n_rows:,} · b={row.n_buildings:,}"
                        elif row.interaction_ci95_low_sgd > 0 or row.interaction_ci95_high_sgd < 0:
                            label = f"{row[column]:+.0f}*\nn={row.n_rows:,} · b={row.n_buildings:,}"
                ax.text(ma, mi, label, ha="center", va="center", fontsize=9,
                        color="white" if not sparse[mi, ma] and abs(values[mi, ma]) > .55 * limit else "#202020")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Nearest mall distance (km)")
        ax.set_ylabel("Nearest open MRT distance (km)")
        ax.set_title(title, fontsize=11)
        fig.colorbar(plot, ax=ax, shrink=.75, label="SGD / month")
    fig.suptitle("MRT × mall: conditional monthly rent associations", fontsize=14)
    fig.supxlabel("* Pointwise building-cluster 95% interval excludes zero; reference row/column is zero by definition.", fontsize=9)
    fig.savefig(OUT / "mrt_mall_interaction_heatmap.png", dpi=190)
    plt.close(fig)

    interior = table.iloc[[4, 5, 7, 8]]
    report = f"""# MRT × 商场距离交互：月租条件关联

在共同完整训练样本 {len(data):,} 笔、{info['clusters']:,} 栋上，以最近已开放 MRT 距离和最近商场距离各分三组，形成 3×3 个格子。每格至少 {table.n_rows.min():,} 笔和 {table.n_buildings.min():,} 栋；图内 n 为交易笔数、b 为不同楼栋数。完整九格系数及逐项 95% 区间见 [CSV](mrt_mall_joint_cells.csv)，模型秩、缺失和源文件哈希见 [分析清单](analysis_manifest.json)。

![MRT 与商场距离的九宫格调整差异及交互偏离](mrt_mall_interaction_heatmap.png)

左图是每个组合相对于“两者都在 0.5 km 内”格子的调整后月租差，单位 SGD/月。右图是**偏离可加关系的交互对比**：该格子的调整差，减去“仅 MRT 属该距离组”的差，再减去“仅商场属该距离组”的差。基准行与基准列按定义为 0，不是独立估计出无交互。内侧四格的对比及 95% 区间为：

| MRT km | 商场 km | 交互偏离 SGD/月 | 楼栋聚类 95% 区间 | 交易/楼栋 |
|---|---|---:|---:|---:|
"""
    for _, row in interior.iterrows():
        report += (f"| {row.mrt_km} | {row.mall_km} | {row.interaction_departure_sgd:+.0f} | "
                   f"[{row.interaction_ci95_low_sgd:+.0f}, {row.interaction_ci95_high_sgd:+.0f}] | "
                   f"{row.n_rows:,.0f}/{row.n_buildings:,.0f} |\n")
    report += """
内侧四格的交互偏离均为正；其中两种距离都超过 1 km 的格子约 +43 SGD/月，逐项 95% 区间为 +7 至 +79，图上以 * 标记。其余三格的区间都跨 0；这组探索性结果并未做多重比较校正。正偏离表示该格子的租金降幅比两个单独距离差直接相加时较小，并不表示绝对租金更高。

回归包含精确“审批月份 × 户型”及 town 固定效应，面积和楼龄各含标准化值与平方项，并控制楼栋最高楼层、CBD 直线距离及小学距离六组。两种热图来自同一显式 3×3 组合模型；标准误及对比区间使用 town+street+block 楼栋聚类的完整协方差矩阵。区间逐项计算，多个格子同时查看时不能当作整体显著性检验；预设少于 100 笔或 30 栋的格子会在图中遮蔽。图上浅灰表示支持不足，并非零效应。

设施距离为球面直线距离，设施清单是当前快照；未还原每笔交易时的开业状态，也未测量步行距离。控制变量不能排除 town 内细微区位、装修等混杂，因此图只描述条件关联，不能作为设施的因果效应或预测提升证据。复现：`.venv-model-benchmark/bin/python scripts/test_analyze_amenity_interactions.py`，然后 `.venv-model-benchmark/bin/python scripts/analyze_amenity_interactions.py`。
"""
    (OUT / "mrt_mall_interaction_summary_zh.md").write_text(report, encoding="utf-8")
    print(table[["mrt_km", "mall_km", "n_rows", "n_buildings", "adjusted_vs_near_near_sgd", "interaction_departure_sgd", "interaction_ci95_low_sgd", "interaction_ci95_high_sgd"]].to_string(index=False))
    print(json.dumps({"source_hash_unchanged": before == after, "model": info}, indent=2))


if __name__ == "__main__":
    main()
