from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class HDBVisualizer:
    """Visualization utilities for HDB rent analysis."""

    def __init__(self, output_dir: str | Path = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_target_distribution(
        self,
        df: pd.DataFrame,
        target_col: str = "monthly_rent",
        save: bool = True,
    ):
        """Plot rental target distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[target_col].dropna(), bins=30)
        ax.set_title("Monthly Rent Distribution")
        ax.set_xlabel("Monthly Rent")
        ax.set_ylabel("Frequency")

        if save:
            fig.savefig(
                self.output_dir / "target_distribution.png",
                bbox_inches="tight",
            )

        return fig, ax

    def plot_numeric_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str = "monthly_rent",
        save: bool = True,
    ):
        """Scatter plot for a numeric feature against rent."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[x_col], df[y_col], alpha=0.4)
        ax.set_title(f"{y_col} vs {x_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

        if save:
            fig.savefig(
                self.output_dir / f"{y_col}_vs_{x_col}.png",
                bbox_inches="tight",
            )

        return fig, ax

    def plot_grouped_rent(
        self,
        df: pd.DataFrame,
        group_col: str,
        target_col: str = "monthly_rent",
        top_n: int = 20,
        save: bool = True,
    ):
        """Plot mean rent grouped by a categorical feature."""
        grouped = (
            df.groupby(group_col, dropna=False)[target_col]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Average {target_col} by {group_col}")
        ax.set_xlabel(f"Average {target_col}")

        if save:
            fig.savefig(
                self.output_dir / f"avg_{target_col}_by_{group_col}.png",
                bbox_inches="tight",
            )

        return fig, ax

    def generate_default_eda_plots(
        self,
        df: pd.DataFrame,
        target_col: str = "monthly_rent",
    ):
        """Generate a minimal default EDA plot set."""
        self.plot_target_distribution(df, target_col=target_col)

        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != target_col
        ]

        for col in numeric_cols[:5]:
            self.plot_numeric_scatter(df, x_col=col, y_col=target_col)
