"""Descriptive calibration audit of rank-13 Gaussian predictive scales."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/transformer_ranked_experiments/13_gaussian_nll"


def main():
    rows = []
    for seed in (42,43,44):
        frame = pd.read_csv(SOURCE / f"outer/gaussian/seed{seed}/predictions.csv")
        error = np.abs(frame.actual.to_numpy()-frame.predicted.to_numpy())
        sigma = frame.sigma_sgd.to_numpy()
        weight = frame.weight.to_numpy()
        rows.append({"seed": seed, "rows": len(frame),
            "mean_sigma_sgd": float(np.average(sigma, weights=weight)),
            "median_sigma_sgd": float(np.median(sigma)),
            "coverage_gaussian_80": float(np.average(error <= 1.281551565545*sigma, weights=weight)),
            "coverage_gaussian_90": float(np.average(error <= 1.644853626951*sigma, weights=weight)),
            "abs_error_sigma_correlation": float(np.corrcoef(error, sigma)[0,1])})
    output = pd.DataFrame(rows)
    output.to_csv(SOURCE / "uncertainty_audit.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
