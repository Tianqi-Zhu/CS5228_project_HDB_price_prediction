"""Evaluate an exactly local stock correction on saved TabPFN backtests."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.worksite_tabpfn_correction import main


if __name__ == "__main__":
    main()
