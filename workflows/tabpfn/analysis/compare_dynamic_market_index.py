"""Run the staged full-history tree and forecast-market-index comparison."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.dynamic_market_index import evaluation


if __name__ == "__main__":
    evaluation()
