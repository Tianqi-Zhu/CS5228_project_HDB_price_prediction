"""Run the recent-window history ablation after compare_dynamic_market_index."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.recent_history import evaluate


if __name__ == "__main__":
    evaluate()
