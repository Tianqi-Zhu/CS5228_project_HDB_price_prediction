"""Plot held-out rental surfaces and PCA factors from the full experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analysis.config import AUX, DATA
from analysis.experiments.factored_rental_surface import (
    HYBRID_COMPONENTS, OUT, TYPES, forecast_surfaces, make_panel,
)
from analysis.experiments.spatiotemporal_hybrid import spatial_frame


def main() -> None:
    panel = make_panel(
        spatial_frame(pd.read_csv(DATA / "train.csv")),
        pd.read_csv(AUX / "sg-hdb-block.csv"),
    )
    origin = int(np.flatnonzero(panel.months == pd.Period("2024-04", freq="M"))[0])
    pca = PCA(n_components=HYBRID_COMPONENTS, svd_solver="full")
    factors = pca.fit_transform(panel.values[:origin])
    cells = panel.cells
    n_cells = len(cells)
    type_offset = TYPES.index("4-room") * n_cells
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    weights = pca.components_[:3, type_offset:type_offset + n_cells]
    limit = np.max(np.abs(weights))
    for i, ax in enumerate(axes):
        dots = ax.scatter(cells["LONGITUDE"], cells["LATITUDE"], c=weights[i],
                          cmap="coolwarm", vmin=-limit, vmax=limit, s=34)
        ax.set(title=f"Factor {i + 1} ({pca.explained_variance_ratio_[i]:.1%})",
               xlabel="Longitude", ylabel="Latitude")
    fig.colorbar(dots, ax=axes, label="4-room PCA loading", shrink=0.8)
    fig.suptitle("PCA loadings on HDB-supported rental cells")
    fig.savefig(OUT / "pca_4room_component_maps.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), layout="constrained", sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(origin), factors[:, i], color=f"C{i}")
        ax.set(ylabel=f"Factor {i + 1} score")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Rental approval month")
    axes[-1].set_xticks(np.arange(0, origin, 4), panel.months[:origin:4].astype(str), rotation=30)
    fig.suptitle("Past-only PCA factor scores at the 2024-04 forecast origin")
    fig.savefig(OUT / "pca_factor_paths.png", dpi=160, facecolor="white")
    plt.close(fig)

    predicted, _, _ = forecast_surfaces(panel, origin, np.array([1]), HYBRID_COMPONENTS, "state_space")
    actual = panel.values[origin, type_offset:type_offset + n_cells]
    forecast = predicted[0, type_offset:type_offset + n_cells]
    lower, upper = np.quantile(np.r_[actual, forecast], [0.02, 0.98])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, values, title in zip(axes, (actual, forecast, forecast - actual),
                                 ("Observed smoothed surface", "Forecast before April", "Forecast minus observed")):
        if title.startswith("Forecast minus"):
            bound = np.max(np.abs(values))
            dots = ax.scatter(cells["LONGITUDE"], cells["LATITUDE"], c=values,
                              cmap="coolwarm", vmin=-bound, vmax=bound, s=34)
        else:
            dots = ax.scatter(cells["LONGITUDE"], cells["LATITUDE"], c=values,
                              cmap="viridis", vmin=lower, vmax=upper, s=34)
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
        fig.colorbar(dots, ax=ax, shrink=0.75, label="SGD")
    fig.suptitle("April 2024 4-room rental surface (observed used only for evaluation)")
    fig.savefig(OUT / "april_2024_surface_forecast.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved factor and surface figures in {OUT}")


if __name__ == "__main__":
    main()
