"""Run the TabM-style neural challenger and tree/neural blend."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.neural_tabm import main


if __name__ == "__main__":
    main()
