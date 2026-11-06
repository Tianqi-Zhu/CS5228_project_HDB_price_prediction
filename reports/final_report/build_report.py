"""Regenerate the paper's figures, tables, and evidence audit without fitting models.

Run from the repository root with the project Python environment.
The IEEEtran manuscript can then be compiled with Tectonic or latexmk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
TAB = HERE / "tables"
BLUE, ORANGE, GREEN = "#235789", "#CB793A", "#36796D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42})

SOURCES = [
    "data/train.csv", "data/test.csv", "reports/train_test_eda/summary.json",
    "reports/auxiliary_data_eda/summary.json",
    "experiment/03_model_comparison/recommended_models.csv",
    "experiment/03_model_comparison/same_input_models.csv",
    "experiment/02_feature_engineering/controlled_feature_ladder.csv",
    "experiment/02_feature_engineering/group_feature_effects.csv",
    "experiment/01_final/submission_selection.json",
    "experiment/04_optimization_history/09_grouped_blending.md",
    "experiment/04_optimization_history/10_plr_transformer_final.md",
    "reports/tabpfn_anniversary_history_80_20/report.md",
    "reports/zch_anniversary_tabpfn_blend/predictions_june.csv",
    "reports/half_median_zch_tabpfn_ensemble/scores.csv",
    "reports/half_median_zch_tabpfn_ensemble/subgroup_scores.csv",
    "reports/flat_model_grouping_ensemble/scores.csv",
    "reports/tabpfn_anniversary_history_80_20/scores.csv",
    "reports/tabpfn_anniversary_history_80_20/predictions_baseline77.csv",
    "reports/tabpfn_anniversary_history_80_20/predictions_anniversary87.csv",
    "reports/tabpfn_anniversary_history_80_20/predictions_anniversary92.csv",
    "reports/tabpfn_feature_ensemble_final_submission/report.md",
    "reports/tabpfn_anniversary_history_80_20/coverage.csv",
    "reports/half_median_zch_tabpfn_ensemble/comparison_predictions.csv",
    "reports/flat_model_grouping_ensemble/comparison_predictions.csv",
    "reports/final_report/reference_pool.bib",
]


def save(name):
    plt.savefig(FIG / (name + ".pdf"), bbox_inches="tight", pad_inches=.025)
    plt.savefig(FIG / (name + ".png"), dpi=190, bbox_inches="tight", pad_inches=.025)
    plt.close()


def table(name, headers, rows, alignment):
    text = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
            " & ".join(headers) + r" \\", r"\midrule"]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    (TAB / (name + ".tex")).write_text("\n".join(text) + "\n", encoding="utf-8")


def appendix_outputs(train, pred):
    """Describe fixed saved predictions; do not fit models or select variants."""
    deletion = pd.read_csv(ROOT / "experiment/02_feature_engineering/group_feature_effects.csv").set_index("variant")
    coverage = pd.read_csv(ROOT / "reports/tabpfn_anniversary_history_80_20/coverage.csv")
    groups = ["joint__all_history", "group__block_history", "group__time", "group__property",
              "group__location", "joint__all_amenities", "group__area_history"]
    labels = ["All histories", "Block/type histories", "Calendar", "Property", "Location", "All amenities", "Area/type histories"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.0), layout="constrained")
    values = deletion.loc[groups, "delta_pooled_rmse_sgd"].to_numpy()
    axes[0].barh(range(len(groups)), values, color=[BLUE if v > 0 else ORANGE for v in values], height=.65)
    axes[0].set_yticks(range(len(groups)), labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color=".35", lw=.7)
    axes[0].set(xlim=(-.8, 9.2), xlabel="Deletion minus full-model RMSE (SGD)", title="(a) Feature-group deletion")
    for i, value in enumerate(values):
        axes[0].text(max(.05, value+.12), i, f"{value:+.2f}", va="center", fontsize=7)
    x = np.arange(3)
    for offset, column, label, color in ((-.18, "fitting_support_share", "Fitting records", BLUE),
                                        (.18, "query_support_share", "Validation records", ORANGE)):
        percentages = coverage[column].to_numpy() * 100
        bars = axes[1].bar(x+offset, percentages, width=.34, label=label, color=color)
        axes[1].bar_label(bars, labels=[f"{v:.1f}" for v in percentages], fontsize=7, padding=2)
    axes[1].set_xticks(x, ["12 months", "24 months", "36 months"])
    axes[1].set(ylim=(0, 88), ylabel="Records with cohort support (%)", title="(b) Anniversary-history availability")
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    save("appendix_feature_diagnostics")

    filtered = pd.read_csv(ROOT / "reports/half_median_zch_tabpfn_ensemble/comparison_predictions.csv")
    grouped = pd.read_csv(ROOT / "reports/flat_model_grouping_ensemble/comparison_predictions.csv")
    cohort = {k: pd.read_csv(ROOT / f"reports/tabpfn_anniversary_history_80_20/predictions_{k}.csv")
              for k in ("baseline77", "anniversary87", "anniversary92")}
    for frame in (filtered, grouped, *cohort.values()):
        np.testing.assert_array_equal(frame.row_id, pred.row_id)
        np.testing.assert_array_equal(frame.month, pred.month)
        np.testing.assert_array_equal(frame.actual, pred.actual)
    np.testing.assert_allclose(filtered.original_equal_blend, pred.equal_blend, atol=1e-6, rtol=0)
    np.testing.assert_allclose(grouped.blend_original, pred.equal_blend, atol=1e-6, rtol=0)

    actual = pred.actual.to_numpy()
    bins = pd.cut(actual, [-np.inf, 2000, 3000, 4000, np.inf], labels=["<=2,000", "2,000-3,000", "3,000-4,000", ">4,000"])
    diagnostic = pd.DataFrame({"band": bins, "squared_error": (pred.equal_blend-actual)**2,
                               "error": pred.equal_blend-actual}).groupby("band", observed=True).agg(
        rows=("error", "size"), sse=("squared_error", "sum"), bias=("error", "mean"))
    diagnostic["row_share_pct"] = diagnostic.rows / len(pred) * 100
    diagnostic["sse_share_pct"] = diagnostic.sse / diagnostic.sse.sum() * 100
    diagnostic["rmse_sgd"] = np.sqrt(diagnostic.sse / diagnostic.rows)
    diagnostic.to_csv(HERE / "supplementary_rent_band_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.85), layout="constrained")
    x = np.arange(4)
    axes[0].bar(x-.18, diagnostic.row_share_pct, width=.34, color=BLUE, label="Share of records")
    axes[0].bar(x+.18, diagnostic.sse_share_pct, width=.34, color=ORANGE, label="Share of squared error")
    axes[0].set_xticks(x, [f"{band}\nn={n:,}" for band,n in zip(diagnostic.index, diagnostic.rows)])
    axes[0].set(ylim=(0,65), ylabel="Share (%)", xlabel="Actual rent band (SGD); diagnostic only", title="(a) Concentration of prediction error")
    axes[0].legend(frameon=False, fontsize=6.5)
    months = np.sort(pred.month.unique())
    monthly_mask = np.stack([(pred.month == month).to_numpy() for month in months])
    monthly_counts = monthly_mask.sum(axis=1)
    old_sse = monthly_mask @ ((filtered.original_equal_blend-actual).to_numpy()**2)
    new_sse = monthly_mask @ ((filtered.filtered_equal_blend-actual).to_numpy()**2)
    monthly_delta = np.sqrt(new_sse/monthly_counts)-np.sqrt(old_sse/monthly_counts)
    bars = axes[1].bar(range(10), monthly_delta, color=[ORANGE if v>0 else GREEN for v in monthly_delta])
    axes[1].bar_label(bars, labels=[f"{v:+.1f}" for v in monthly_delta], fontsize=6, padding=2)
    axes[1].axhline(0, color=".35", lw=.7)
    axes[1].set_xticks(range(10), ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"], rotation=45)
    axes[1].set(ylim=(-3, 11), ylabel="Filtered minus original RMSE (SGD)", title="(b) Training-only filter by month")
    save("appendix_error_diagnostics")

    # Paired circular moving blocks preserve all rows in each sampled month.
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(months), size=(5000, 4))
    draws = ((starts[:, :, None] + np.arange(3)) % len(months)).reshape(5000,-1)[:, :len(months)]
    predictions = {"77": cohort["baseline77"].predicted.to_numpy(),
                   "87": cohort["anniversary87"].predicted.to_numpy(),
                   "92": cohort["anniversary92"].predicted.to_numpy(),
                   "equal": pred.equal_blend.to_numpy(), "locked": pred.locked_blend.to_numpy(),
                   "grouped": grouped.blend_grouped.to_numpy(),
                   "filtered": filtered.filtered_equal_blend.to_numpy()}
    comparisons = [("87 vs 77 inputs", "87", "77"), ("92 vs 87 inputs", "92", "87"),
                   ("Equal blend vs 87", "equal", "87"), ("Earlier-locked blend vs 87", "locked", "87"),
                   ("Coarser categories vs original", "grouped", "equal"),
                   ("Half-median filter vs original", "filtered", "equal")]
    records = []
    for label, candidate, reference in comparisons:
        sse_c = monthly_mask @ ((predictions[candidate]-actual)**2)
        sse_r = monthly_mask @ ((predictions[reference]-actual)**2)
        boot_c = np.sqrt(sse_c[draws].sum(axis=1)/monthly_counts[draws].sum(axis=1))
        boot_r = np.sqrt(sse_r[draws].sum(axis=1)/monthly_counts[draws].sum(axis=1))
        lower, upper = np.quantile(boot_c-boot_r, [.025,.975])
        delta = np.sqrt(sse_c.sum()/len(pred))-np.sqrt(sse_r.sum()/len(pred))
        records.append({"contrast": label, "delta_rmse_sgd": float(delta), "lower95_sgd": float(lower),
                        "upper95_sgd": float(upper), "months_improved": int(np.sum(sse_c < sse_r))})
    pd.DataFrame(records).to_csv(HERE / "supplementary_paired_intervals.csv", index=False)
    table("paired_intervals", ["Contrast", r"$\Delta$RMSE", "95\\% block interval", "Months improved"], [
        [r["contrast"], f"{r['delta_rmse_sgd']:+.3f}", f"[{r['lower95_sgd']:+.3f}, {r['upper95_sgd']:+.3f}]", f"{r['months_improved']}/10"] for r in records], "lrrr")
    return {"paired_saved_predictions_aligned": True, "paired_replicates": 5000,
            "block_months": 3, "circular_blocks": True, "seed": 42,
            "intervals": records, "rent_band_rows_total": int(diagnostic.rows.sum())}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    train, test = [pd.read_csv(ROOT / f"data/{p}.csv") for p in ("train", "test")]
    assert len(train) == 150000 and len(test) == 50000
    assert "MONTHLY_RENT" not in test
    assert train.duplicated().sum() == 425
    assert train.isna().sum().sum() == 0
    assert (train.RENT_APPROVAL_DATE < "2024-04").sum() == 114842
    assert (train.RENT_APPROVAL_DATE < "2024-06").sum() == 121005

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.05), layout="constrained")
    monthly = train.groupby("RENT_APPROVAL_DATE").MONTHLY_RENT.agg(["mean", "median"])
    dates = pd.to_datetime(monthly.index)
    axes[0].plot(dates, monthly["mean"], color=BLUE, label="Mean")
    axes[0].plot(dates, monthly["median"], color=GREEN, ls="--", label="Median")
    axes[0].axvline(pd.Timestamp("2024-04-01"), color=".4", lw=.8, ls=":")
    axes[0].set(ylabel="Monthly rent (SGD)", title="(a) Observed market shift")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].hist(train.MONTHLY_RENT, bins=np.arange(250, 7850, 150), color=BLUE, alpha=.9)
    axes[1].axvline(train.MONTHLY_RENT.median(), color=ORANGE, lw=1.2, label="Median = 2,700")
    axes[1].set(xlabel="Monthly rent (SGD)", ylabel="Transactions", title="(b) Rent distribution")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    axes[1].legend(frameon=False)
    save("eda")

    ladder = pd.read_csv(ROOT / SOURCES[6])
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.0), sharey=True, layout="constrained")
    for ax, stage, title in zip(axes, ("outer", "june"), ("(a) Annual: Apr 2024-Mar 2025", "(b) Recent: Jun 2024-Mar 2025")):
        for family, label, color in (("lgb", "LightGBM", BLUE), ("cat", "CatBoost", ORANGE)):
            a = ladder[(ladder.stage == stage) & (ladder.family == family)].sort_values("features")
            ax.plot(range(4), a.rmse_sgd, marker="o", ms=3.5, color=color, label=label)
        ax.set_xticks(range(4), ["19\nStatic", "27\n+ History", "47\n+ Ratios", "62\n+ Cohorts"])
        ax.set(title=title, ylabel="RMSE (SGD)")
        ax.grid(axis="y", color=".9", lw=.6)
        ax.legend(frameon=False)
    save("feature_ladder")

    models = pd.read_csv(ROOT / SOURCES[4])
    order = ["ridge", "extra_trees", "lightgbm", "xgboost", "catboost", "embedding_mlp", "ft_transformer"]
    indexed = models.set_index("family")
    labels = {"ridge":"Ridge", "extra_trees":"ExtraTrees", "lightgbm":"LightGBM", "xgboost":"XGBoost",
              "catboost":"CatBoost", "embedding_mlp":"Embedding MLP", "ft_transformer":"FT-Transformer"}
    table("common_models", ["Model", "RMSE", "MAE", "Bias"], [
        [labels[k], f"{indexed.loc[k].rmse_sgd:.2f}", f"{indexed.loc[k].mae_sgd:.2f}", f"{indexed.loc[k].bias_sgd:+.2f}"] for k in order], "lrrr")
    table("feature_ladder", ["Model / window", "19", "27", "47", "62"], [
        [label + " / " + window] + [f"{ladder[(ladder.family == family) & (ladder.stage == stage) & (ladder.features == n)].iloc[0].rmse_sgd:.2f}" for n in (19,27,47,62)]
        for family,label in (("lgb","LightGBM"),("cat","CatBoost")) for stage,window in (("outer","annual"),("june","recent"))], "lrrrr")

    selection = json.loads((ROOT / SOURCES[8]).read_text(encoding="utf-8"))
    table("submissions", ["Pipeline", "Annual RMSE", "Recent RMSE"], [
        [label, f"{c['local_annual_rmse']:.4f}", f"{c['local_recent_rmse']:.4f}"]
        for c,label in zip(selection["candidates"], ["History ensemble", "Macro-enhanced ensemble", "Grouped ensemble + PLR-FT"])], "lrr")

    pred = pd.read_csv(ROOT / SOURCES[12])
    assert len(pred) == 28995 and pred.row_id.is_unique
    np.testing.assert_array_equal(pred.actual, train.loc[pred.row_id, "MONTHLY_RENT"])
    np.testing.assert_array_equal(pred.month, train.loc[pred.row_id, "RENT_APPROVAL_DATE"])
    pred["equal_blend"] = (pred.zch_original + pred.tabpfn87) / 2
    metrics = {}
    for name in ("zch_original", "tabpfn87", "equal_blend", "locked_blend"):
        error = pred[name] - pred.actual
        metrics[name] = {"rows": len(error), "rmse": float(np.sqrt(np.mean(error**2))),
                         "mae": float(np.abs(error).mean()), "bias": float(error.mean())}
    assert abs(metrics["equal_blend"]["rmse"] - 440.9519575034) < 1e-7
    assert abs(metrics["tabpfn87"]["rmse"] - 441.9592693871) < 1e-7
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.1), layout="constrained")
    month = pred.groupby("month")
    for name, label, color in (("zch_original", "Seven-tree ensemble", BLUE), ("tabpfn87", "TabPFN + cohorts", ORANGE), ("equal_blend", "Equal blend", GREEN)):
        rmses = month.apply(lambda p: np.sqrt(np.mean((p[name]-p.actual)**2)), include_groups=False)
        axes[0].plot(range(10), rmses, marker="o", ms=2.5, color=color, label=label)
    axes[0].set_xticks(range(10), ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"], rotation=45)
    axes[0].set(ylabel="RMSE (SGD)", title="(a) Recent development months")
    axes[0].legend(frameon=False, fontsize=6.5)
    density = axes[1].hexbin(pred.actual, pred.equal_blend, gridsize=45, mincnt=1, bins="log", cmap="Blues", rasterized=True)
    fig.colorbar(density, ax=axes[1], label="Transactions / hexagon", fraction=.045, pad=.025)
    axes[1].plot([0,8000],[0,8000], color=ORANGE, ls="--", lw=.9)
    axes[1].set(xlim=(0,8000), ylim=(0,8000), xlabel="Actual rent (SGD)", ylabel="Predicted rent (SGD)", title="(b) Predicted versus observed rent")
    save("error_analysis")

    filtering = pd.read_csv(ROOT / SOURCES[13]).set_index("model")
    assert abs(filtering.loc["original_equal_blend", "rmse_sgd"] - metrics["equal_blend"]["rmse"]) < 1e-7
    table("filtering", ["Model", "RMSE", "MAE", "Bias"], [
        [label] + [f"{filtering.loc[k,c]:.2f}" for c in ("rmse_sgd","mae_sgd","bias_sgd")]
        for k,label in (("original_equal_blend","All fitting rows"),("filtered_equal_blend","Half-median filter"))], "lrrr")
    cohort_metrics = {}
    for name in ("baseline77", "anniversary87", "anniversary92"):
        frame = pd.read_csv(ROOT / f"reports/tabpfn_anniversary_history_80_20/predictions_{name}.csv")
        np.testing.assert_array_equal(frame.row_id, pred.row_id)
        np.testing.assert_array_equal(frame.actual, pred.actual)
        error = frame.predicted - frame.actual
        cohort_metrics[name] = {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.abs(error).mean())}
    cohort_rows = [[label, count, f"{cohort_metrics[k]['rmse']:.3f}", f"{cohort_metrics[k]['mae']:.3f}"]
        for k,label,count in (("baseline77", "Frozen Fast control", "77"),
        ("anniversary87", "+ 12/24-month cohorts", "87"),
        ("anniversary92", "+ 12/24/36-month cohorts", "92"))]
    table("tabpfn", ["Model", "Inputs", "RMSE", "MAE"], cohort_rows + [
        ["Seven-tree ensemble", "varied", "443.863", "322.919"],
        ["Earlier-locked blend", "varied", f"{metrics['locked_blend']['rmse']:.3f}", f"{metrics['locked_blend']['mae']:.3f}"],
        ["Equal blend", "varied", "440.952", "318.888"]], "llrr")

    supplementary = appendix_outputs(train, pred)

    audit = {"checks": {"raw_row_counts": True, "annual_and_recent_split_sizes": True,
        "duplicate_count": True, "recent_prediction_ids_months_labels": True,
        "latest_tree_tabpfn_metrics_recomputed": True,
        "all_three_tabpfn_cohort_ablation_metrics_recomputed": True,
        "filter_control_reproduces_unfiltered_blend": True},
        "data": {"train": len(train), "test": len(test), "exact_duplicates": int(train.duplicated().sum()),
        "monthly_mean_first": float(monthly['mean'].iloc[0]), "monthly_mean_last": float(monthly['mean'].iloc[-1])},
        "recomputed_recent_metrics": metrics,
        "recomputed_cohort_metrics": cohort_metrics,
        "supplementary_diagnostics": supplementary,
        "final_selection": selection["recommended_final"],
        "limitations": ["Archived final-structure scores and Kaggle value are read from the saved selection record; the screenshot was not supplied in this turn.",
            "Root output/ model binaries and final submission CSV are absent from this checkout; no full final-model replay is claimed.",
            "The report does not train models or upload predictions.",
            "Names, student IDs, confirmed team name and actual workload ownership require the team."],
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}}
    (HERE / "evidence_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_checks": audit["checks"], "metrics": metrics,
        "figures": len(list(FIG.glob('*.pdf'))), "tables": len(list(TAB.glob('*.tex')))}, indent=2))


if __name__ == "__main__":
    main()
