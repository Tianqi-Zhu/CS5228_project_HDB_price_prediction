"""Calculate stock indices and compare national/regional HDB mean-rent growth."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.stock_index_regions import main


if __name__ == "__main__":
    main()
