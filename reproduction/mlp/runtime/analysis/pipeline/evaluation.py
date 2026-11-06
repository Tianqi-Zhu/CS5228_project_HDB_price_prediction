"""Common rental prediction metrics and group-level error summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from analysis.pipeline.config import EXPERIMENT_REPORTS, TARGET


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError("Metrics require matching, nonempty one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite targets and predictions")
    error = predicted - actual
    return {
        "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
    }


def grouped_errors(frame: pd.DataFrame, by: str, model: str) -> pd.DataFrame:
    rows = []
    for group, part in frame.groupby(by, sort=True):
        row = {"model": model, "group": str(group), "n": len(part)}
        row.update(metrics(part["actual"].to_numpy(), part["predicted"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


class HDBEvaluator:
    """Evaluate a fitted HDB regression model."""

    def evaluate(self, model, X_test, y_test) -> dict:
        predictions = model.predict(X_test)
        values = metrics(y_test, predictions)
        return {
            "mae": values["mae_sgd"],
            "rmse": values["rmse_sgd"],
            "bias": values["bias_sgd"],
            "r2": r2_score(y_test, predictions),
        }

    @staticmethod
    def prediction_frame(model, X_test, y_test) -> pd.DataFrame:
        predictions = model.predict(X_test)
        return pd.DataFrame(
            {
                "actual": y_test.to_numpy(),
                "predicted": predictions,
                "residual": y_test.to_numpy() - predictions,
            },
            index=y_test.index,
        )

    @staticmethod
    def print_report(values: dict) -> None:
        print("Evaluation Metrics")
        print("-" * 30)
        for name, value in values.items():
            print(f"{name.upper():<8}: {value:,.4f}")


class HDBVisualizer:
    """Small reusable plot set for exploratory model inspection."""

    def __init__(self, output_dir: str | Path = EXPERIMENT_REPORTS / "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_target_distribution(self, df, target_col: str = TARGET, save: bool = True):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[target_col].dropna(), bins=30)
        ax.set(title="Monthly Rent Distribution", xlabel="Monthly Rent", ylabel="Frequency")
        if save:
            fig.savefig(self.output_dir / "target_distribution.png", bbox_inches="tight")
        return fig, ax

    def plot_numeric_scatter(self, df, x_col: str, y_col: str = TARGET, save: bool = True):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[x_col], df[y_col], alpha=0.4)
        ax.set(title=f"{y_col} vs {x_col}", xlabel=x_col, ylabel=y_col)
        if save:
            fig.savefig(self.output_dir / f"{y_col}_vs_{x_col}.png", bbox_inches="tight")
        return fig, ax

    def plot_grouped_rent(self, df, group_col: str, target_col: str = TARGET,
                          top_n: int = 20, save: bool = True):
        grouped = (df.groupby(group_col, dropna=False)[target_col].mean()
                   .sort_values(ascending=False).head(top_n))
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.sort_values().plot(kind="barh", ax=ax)
        ax.set(title=f"Average {target_col} by {group_col}", xlabel=f"Average {target_col}")
        if save:
            fig.savefig(self.output_dir / f"avg_{target_col}_by_{group_col}.png", bbox_inches="tight")
        return fig, ax

    def generate_default_eda_plots(self, df, target_col: str = TARGET) -> None:
        self.plot_target_distribution(df, target_col=target_col)
        numeric = [
            column for column in df.select_dtypes(include="number").columns
            if column != target_col
        ]
        for column in numeric[:5]:
            self.plot_numeric_scatter(df, x_col=column, y_col=target_col)


def make_monthly_error_figure(monthly: pd.DataFrame, model: str,
                              output_dir=EXPERIMENT_REPORTS / "model_baseline") -> None:
    part = monthly[monthly["model"] == model].sort_values("group")
    months = part["group"].tolist()
    x = np.arange(len(months))
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    display_name = {
        "mean_constant": "mean predictor", "A_property": "A: property",
        "B_property_time": "B: property + time",
        "C_property_time_town": "C: property + time + town",
    }.get(model, model)
    fig.suptitle(f"Validation error by month: {display_name}", fontsize=20, weight="bold")
    ax_rmse.plot(x, part["rmse_sgd"], marker="o", linewidth=2.5, color="#147D91")
    ax_rmse.set_ylabel("RMSE (SGD)")
    ax_rmse.grid(axis="y", alpha=0.25)
    ax_bias.bar(x, part["bias_sgd"], color="#D78036", width=0.7)
    ax_bias.axhline(0, color="#193043", linewidth=1)
    ax_bias.set_ylabel("Mean prediction - actual (SGD)")
    ax_bias.set_xlabel("Rent approval month")
    ax_bias.set_xticks(x, months, rotation=40, ha="right")
    ax_bias.grid(axis="y", alpha=0.25)
    fig.savefig(Path(output_dir) / "monthly_validation_error.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot_model_comparison(monthly: pd.DataFrame,
                          output_dir=EXPERIMENT_REPORTS / "model_families") -> None:
    labels = ["ridge", "random_forest", "lightgbm", "C_property_time_town"]
    display = {
        "ridge": "Ridge", "random_forest": "Random forest",
        "lightgbm": "LightGBM", "C_property_time_town": "CatBoost C",
    }
    colors = {
        "ridge": "#D78036", "random_forest": "#5B7894",
        "lightgbm": "#8F5BB5", "C_property_time_town": "#147D91",
    }
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_rmse, ax_bias) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    fig.suptitle("Model comparison on the future holdout", fontsize=20, weight="bold")
    months = sorted(monthly["group"].unique())
    x = np.arange(len(months))
    for label in labels:
        part = monthly[monthly["model"] == label].set_index("group").loc[months]
        ax_rmse.plot(x, part["rmse_sgd"], marker="o", linewidth=2.2,
                     color=colors[label], label=display[label])
        ax_bias.plot(x, part["bias_sgd"], marker="o", linewidth=2.2, color=colors[label])
    ax_rmse.set_ylabel("RMSE (SGD)")
    ax_rmse.legend(loc="upper left")
    ax_rmse.grid(axis="y", alpha=0.25)
    ax_bias.axhline(0, color="#193043", linewidth=1)
    ax_bias.set_ylabel("Mean prediction - actual (SGD)")
    ax_bias.set_xlabel("Rent approval month")
    ax_bias.set_xticks(x, months, rotation=40, ha="right")
    ax_bias.grid(axis="y", alpha=0.25)
    fig.savefig(Path(output_dir) / "model_comparison_by_month.png", dpi=160, facecolor="white")
    plt.close(fig)
