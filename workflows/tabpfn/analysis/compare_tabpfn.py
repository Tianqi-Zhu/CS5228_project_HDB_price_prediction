"""Run the fixed-origin TabPFN comparison from the repository root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.tabpfn_comparison import main


if __name__ == "__main__":
    main()
