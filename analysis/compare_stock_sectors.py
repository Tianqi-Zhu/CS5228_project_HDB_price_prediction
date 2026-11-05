"""Run the stock-industry HDB rental forecasting experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.stock_sector_signals import main


if __name__ == "__main__":
    main()
