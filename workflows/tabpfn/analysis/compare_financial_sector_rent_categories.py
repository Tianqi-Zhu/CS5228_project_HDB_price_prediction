"""Run financial-stock versus mean HDB rent experiments by room type/model."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.financial_sector_rent_categories import main


if __name__ == "__main__":
    main()
