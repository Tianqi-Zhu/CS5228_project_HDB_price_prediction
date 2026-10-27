"""Run origin-safe feature ablations and frozen TabPFN residual heads."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.experiments.tabpfn_adaptation import main

if __name__ == '__main__':
    main()
