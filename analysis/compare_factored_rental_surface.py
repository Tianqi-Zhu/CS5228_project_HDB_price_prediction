"""Run the originally requested PCA surface and CatBoost hybrid experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.factored_rental_surface import evaluate


if __name__ == "__main__":
    evaluate()
