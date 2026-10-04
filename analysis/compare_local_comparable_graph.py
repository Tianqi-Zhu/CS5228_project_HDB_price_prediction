"""Run the block-level comparable-rent graph ablation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.local_comparable_graph import main


if __name__ == "__main__":
    main()
