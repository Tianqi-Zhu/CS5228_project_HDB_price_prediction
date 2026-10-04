"""Build reproducible validation summaries and simple prediction ensembles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analysis.pipeline.config import DATE, EXPERIMENT_REPORTS
from analysis.pipeline.data import check_output_path


DEEP_MEMBERS = [
    "aux_transformer_te18_seed42",
    "aux_transformer_te18_seed77",
    "aux_resnet_te18_seed77",
]
HYBRID_MEMBERS = DEEP_MEMBERS + ["aux_lightgbm_l31"]
REFERENCE_RMSE = 485.09281617236115


def metrics(actual, prediction):
    return {
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "mae": float(mean_absolute_error(actual, prediction)),
        "bias": float(np.mean(prediction - actual)),
    }


def load_members(root: Path, names: list[str]):
    frames = []
    for name in names:
        path = root / name / "predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if frames:
            for column in ("index", DATE, "actual"):
                aligned = (
                    np.allclose(frame[column], frames[0][column])
                    if column == "actual" else frame[column].equals(frames[0][column])
                )
                if not aligned:
                    raise ValueError(f"{name}: prediction rows do not align on {column}")
        frames.append(frame)
    return frames


def run(root=EXPERIMENT_REPORTS / "deep_learning", output=None):
    root = Path(root)
    output = check_output_path(Path(output) if output else root / "summary")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for result_path in sorted(root.glob("*/results.json")):
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        if "metrics" in artifact:
            config = artifact.get("config", {})
            rows.append({
                "run": result_path.parent.name,
                "architecture": config.get("architecture", artifact.get("model", "")),
                "seed": config.get("seed", artifact.get("seed", "")),
                "fit_window_months": config.get("fit_window_months", ""),
                "recency_half_life": config.get("recency_half_life", ""),
                "target_encoding": config.get("target_encoding", artifact.get("target_encoding", "")),
                **artifact["metrics"],
            })
    pd.DataFrame(rows).sort_values("rmse").to_csv(output / "model_metrics.csv", index=False)

    summary = {"reference_lightgbm_rmse": REFERENCE_RMSE, "ensembles": {}}
    monthly_frames = {}
    for label, names in (("deep_equal_weight", DEEP_MEMBERS), ("hybrid_equal_weight", HYBRID_MEMBERS)):
        frames = load_members(root, names)
        prediction = np.mean([frame["prediction"].to_numpy(float) for frame in frames], axis=0)
        actual = frames[0]["actual"].to_numpy(float)
        result = metrics(actual, prediction)
        result["members"] = names
        result["rmse_improvement_vs_reference"] = REFERENCE_RMSE - result["rmse"]
        summary["ensembles"][label] = result
        predictions = frames[0][["index", DATE, "actual"]].copy()
        predictions["prediction"] = prediction
        predictions.to_csv(output / f"{label}_predictions.csv", index=False)
        monthly = predictions.groupby(DATE).apply(
            lambda group: pd.Series({
                "rows": len(group),
                "rmse": mean_squared_error(group["actual"], group["prediction"]) ** 0.5,
                "mae": mean_absolute_error(group["actual"], group["prediction"]),
                "bias": np.mean(group["prediction"] - group["actual"]),
            }), include_groups=False,
        ).reset_index()
        monthly.to_csv(output / f"{label}_monthly_errors.csv", index=False)
        monthly_frames[label] = monthly

    # A fixed, interpretable blend chosen from the temporal holdout.  The
    # historical prior and residual learner add signals that the direct models
    # do not capture, while retaining half of the robust hybrid ensemble.
    blend_sources = {
        "hybrid_equal_weight": (output / "hybrid_equal_weight_predictions.csv", 0.50),
        "historical_prior_48m": (root / "historical_priors" / "predictions.csv", 0.24),
        "residual_transformer_seed42": (
            root / "aux_transformer_residual_te18_seed42" / "predictions.csv", 0.26,
        ),
    }
    blend_frames = []
    for name, (path, weight) in blend_sources.items():
        frame = pd.read_csv(path)
        if blend_frames:
            for column in ("index", DATE, "actual"):
                aligned = (
                    np.allclose(frame[column], blend_frames[0][1][column])
                    if column == "actual" else frame[column].equals(blend_frames[0][1][column])
                )
                if not aligned:
                    raise ValueError(f"{name}: prediction rows do not align on {column}")
        blend_frames.append((name, frame, weight))
    prediction = sum(
        weight * frame["prediction"].to_numpy(float)
        for _, frame, weight in blend_frames
    )
    actual = blend_frames[0][1]["actual"].to_numpy(float)
    label = "historical_residual_blend"
    result = metrics(actual, prediction)
    result["weights"] = {name: weight for name, _, weight in blend_frames}
    result["rmse_improvement_vs_reference"] = REFERENCE_RMSE - result["rmse"]
    summary["ensembles"][label] = result
    predictions = blend_frames[0][1][["index", DATE, "actual"]].copy()
    predictions["prediction"] = prediction
    predictions.to_csv(output / f"{label}_predictions.csv", index=False)
    monthly = predictions.groupby(DATE).apply(
        lambda group: pd.Series({
            "rows": len(group),
            "rmse": mean_squared_error(group["actual"], group["prediction"]) ** 0.5,
            "mae": mean_absolute_error(group["actual"], group["prediction"]),
            "bias": np.mean(group["prediction"] - group["actual"]),
        }), include_groups=False,
    ).reset_index()
    monthly.to_csv(output / f"{label}_monthly_errors.csv", index=False)
    monthly_frames[label] = monthly
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        ("Previous\nLightGBM", REFERENCE_RMSE),
        ("Best\nResNet", next(row["rmse"] for row in rows if row["run"] == "aux_resnet_te18_seed77")),
        ("Best\nTransformer", next(row["rmse"] for row in rows if row["run"] == "aux_transformer_te18_seed77")),
        ("Deep\nensemble", summary["ensembles"]["deep_equal_weight"]["rmse"]),
        ("Hybrid\nensemble", summary["ensembles"]["hybrid_equal_weight"]["rmse"]),
        ("Residual\nTransformer", next(
            row["rmse"] for row in rows
            if row["run"] == "aux_transformer_residual_te18_seed42"
        )),
        ("Historical +\nresidual blend", summary["ensembles"][label]["rmse"]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    labels, values = zip(*selected)
    bars = ax.bar(labels, values, color=[
        "#8797a5", "#42a5a5", "#238c8c", "#1f5f85", "#315879", "#a2673f", "#17324d",
    ])
    ax.set_ylabel("Temporal validation RMSE (SGD)")
    ax.set_ylim(min(values) - 4, max(values) + 4)
    ax.set_title("Auxiliary deep models improve the fixed 2024-04 holdout")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, monthly in monthly_frames.items():
        ax.plot(monthly[DATE], monthly["rmse"], marker="o", linewidth=2,
                label=label.replace("_", " "))
    ax.set_ylabel("Monthly RMSE (SGD)")
    ax.set_xlabel("Validation month")
    ax.set_title("Ensemble error across the validation horizon")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(output / "monthly_rmse.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    run()


if __name__ == "__main__":
    main()
