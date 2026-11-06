"""Summarize and plot the inference-only context sampling ablation."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from analysis.compare_tabpfn_diagscale_80_20 import BASE
from analysis.config import REPORTS
from analysis.evaluation import metrics


RUN = REPORTS / "tabpfn_context_sampling/phase_a_outer_budget10000_queriesfull_seed42"
LABELS = {"S0": "Full context", "S1": "Uniform", "S2": "Recent", "S3": "Local", "S4": "Mixture"}


def main():
    scores = pd.read_csv(RUN / "scores.csv")
    base = joblib.load(BASE)
    q = base["X_query"]
    x = base["X_train"]
    monthly, towns, distributions, coverage = [], [], [], []
    qtown = q.TOWN_CLEAN.astype(str)
    qcoords = np.deg2rad(q[["LATITUDE", "LONGITUDE"]].to_numpy(float))
    context_origin = pd.Period("2024-06", "M").ordinal
    for strategy in LABELS:
        predicted = pd.read_csv(RUN / f"predictions_{strategy}.csv")
        np.testing.assert_array_equal(predicted.row_id.to_numpy(), q.index.to_numpy())
        predicted["town"] = qtown.to_numpy()
        for month, part in predicted.groupby("month", sort=True):
            monthly.append({"strategy": strategy, "month": month, "n": len(part),
                            **metrics(part.actual, part.predicted)})
        for town, part in predicted.groupby("town", sort=True):
            towns.append({"strategy": strategy, "town": town, "n": len(part),
                          **metrics(part.actual, part.predicted)})
        ids = np.load(RUN / f"context_ids_{strategy}.npy")
        selected = x.loc[ids]
        age = context_origin - np.asarray(
            [pd.Period("2021-01", "M").ordinal + int(v)
             for v in selected.APPROVAL_MONTH_INDEX], dtype=int)
        context_tree = BallTree(
            np.deg2rad(selected[["LATITUDE", "LONGITUDE"]].to_numpy(float)),
            metric="haversine")
        distance, _ = context_tree.query(qcoords, k=1)
        km = distance[:, 0] * 6371.0088
        distributions.append(pd.DataFrame({"strategy": strategy, "age_months": age}))
        coverage.append(pd.DataFrame({"strategy": strategy, "nearest_context_km": km}))
    monthly_df = pd.DataFrame(monthly)
    town_df = pd.DataFrame(towns)
    distribution_df = pd.concat(distributions, ignore_index=True)
    coverage_df = pd.concat(coverage, ignore_index=True)
    monthly_df.to_csv(RUN / "monthly_scores.csv", index=False)
    town_df.to_csv(RUN / "town_scores.csv", index=False)
    distribution_df.groupby("strategy")[["age_months"]].agg(
        ["median", "mean", "std"]).to_csv(RUN / "context_distribution_summary.csv")
    coverage_df.groupby("strategy")[["nearest_context_km"]].agg(
        ["median", "mean", "std"]).to_csv(RUN / "query_coverage_summary.csv")

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ordered = scores.set_index("sampler_inference").loc[list(LABELS)]
    bars = ax.bar([LABELS[k] for k in LABELS], ordered.rmse_sgd,
                  color=["#2b6cb0", "#9aa4af", "#d69e2e", "#38a169", "#805ad5"])
    ax.set_ylim(450, max(ordered.rmse_sgd) + 3)
    ax.set_ylabel("Validation RMSE (S$)")
    ax.set_title("Same TabPFN adapter; different labeled context selection")
    for bar, value in zip(bars, ordered.rmse_sgd, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .15, f"{value:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(RUN / "outer_rmse.png", dpi=160)
    plt.close(fig)

    pivot = monthly_df.pivot(index="strategy", columns="month", values="rmse_sgd").loc[list(LABELS)]
    difference = pivot.sub(pivot.loc["S1"], axis=1)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(difference.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-18, vmax=18)
    ax.set_yticks(range(len(LABELS)), [LABELS[key] for key in LABELS])
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_title("Monthly RMSE difference versus uniform 10k context (negative is better)")
    fig.colorbar(im, ax=ax, label="RMSE difference (S$)")
    fig.tight_layout()
    fig.savefig(RUN / "monthly_rmse_difference.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for key in ("S1", "S2", "S3", "S4"):
        subset = distribution_df[distribution_df.strategy.eq(key)]
        axes[0].hist(subset.age_months, bins=np.arange(0, 44, 3), density=True,
                     alpha=.4, label=LABELS[key])
        nearest = coverage_df[coverage_df.strategy.eq(key)].nearest_context_km
        axes[1].hist(np.clip(nearest, 0, 1), bins=np.linspace(0, 1, 30),
                     density=True, alpha=.4, label=LABELS[key])
    axes[0].set(xlabel="Months before June 2024", ylabel="Density", title="Context age")
    axes[1].set(xlabel="Validation property to nearest context (km; clipped at 1)",
                ylabel="Density", title="Spatial coverage")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RUN / "context_age_distance.png", dpi=160)
    plt.close(fig)

    timing = ordered.inference_seconds.copy()
    timing.loc["S0"] = 1185.423050100042  # saved matched full-context adapter timing
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for key in LABELS:
        ax.scatter(timing.loc[key], ordered.loc[key, "rmse_sgd"], s=55)
        ax.annotate(LABELS[key], (timing.loc[key], ordered.loc[key, "rmse_sgd"]),
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set(xlabel="Saved fit + prediction time (seconds; log scale)",
           ylabel="Validation RMSE (S$)", title="Context size and accuracy tradeoff")
    fig.tight_layout()
    fig.savefig(RUN / "cost_accuracy.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
