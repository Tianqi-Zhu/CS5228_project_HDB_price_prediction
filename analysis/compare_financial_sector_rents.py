"""Run the financial-sector stock versus national mean-rent experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.financial_sector_rents import main


if __name__ == "__main__":
    main()
