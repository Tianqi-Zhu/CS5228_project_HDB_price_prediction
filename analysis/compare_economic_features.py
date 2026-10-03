"""Run the E-baseline versus engineered economic-feature experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.economic_feature_ablation import main


if __name__ == "__main__":
    main()
