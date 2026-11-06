"""Run the company-stock / nearby HDB rent-change experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.company_worksite_stocks import main


if __name__ == "__main__":
    main()
