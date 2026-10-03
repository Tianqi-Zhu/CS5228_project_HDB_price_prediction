"""Run the COE/stock temporal-signal EDA from the repository root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eda.economic_signals import main


if __name__ == "__main__":
    main()
