"""Run the fixed-origin Bayesian amenity factor experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.bayesian_amenity_factors import main


if __name__ == "__main__":
    main()
