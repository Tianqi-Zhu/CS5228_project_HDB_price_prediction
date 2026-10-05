"""Run the one-resolution, pre-origin rental-surface proof of concept."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.coarse_rental_surface import evaluate


if __name__ == "__main__":
    evaluate()
