"""Rank 5: causal rent-change innovations in the conditioned frozen-T0 head."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
import analysis.run_conditioned_rent_corrections as core


OUT = REPORTS / "transformer_ranked_experiments/05_rent_innovations"
CACHE = ROOT / ".cache/transformer_ranked_experiments/05_rent_innovations"
SOURCE_REPORT = REPORTS / "transformer_ranked_experiments/01_conditioned_correction"
SOURCE_CACHE = ROOT / ".cache/transformer_ranked_experiments/01_conditioned_correction"


def innovations(values):
    """A change minus the mean of its preceding three available changes."""
    output = values.copy()
    if output.ndim != 3 or output.shape[1:] != (12, 34):
        raise ValueError("Expected twelve monthly tokens with value and presence channels")
    for level, change in ((0, 1), (2, 3), (4, 5)):
        observed = values[:, :, 17+change] > 0
        for month in range(12):
            previous = slice(max(0, month-3), month)
            count = observed[:, previous].sum(1)
            total = (values[:, previous, change] * observed[:, previous]).sum(1)
            forecast = np.divide(total, count, out=np.zeros(len(values)), where=count > 0)
            output[:, month, change] = np.where(observed[:, month],
                                                values[:, month, change]-forecast, 0.)
        output[:, :, level] = 0.
        output[:, :, 17+level] = 0.
    return output


def innovation_data(_prepared, fold, seed, args):
    tag = f"{fold}_seed{seed}"
    path, manifest = SOURCE_CACHE / "features" / f"{tag}.joblib", SOURCE_REPORT / "features" / f"{tag}.json"
    if sha256(path) != json.loads(manifest.read_text())["sha256"]:
        raise ValueError("Original T0 representation cache changed")
    values = joblib.load(path)
    if args.smoke:
        values = {key: (value[:512] if key.startswith("train_") or key == "y_train" else
                        value[:64] if key.startswith("query_") else value)
                  for key, value in values.items()}
    for side in ("train", "query"):
        values[f"{side}_sequence"] = innovations(values[f"{side}_sequence"])
    return values


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=8.)
    parser.add_argument("--max-gpu-utilization", type=float, default=35.)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.)
    args = parser.parse_args()
    args.smoke = args.stage == "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    core.OUT, core.CACHE = OUT, CACHE
    core.DESIGNS = {"conditioned_innovations": (True, True)}
    core.base_data = innovation_data
    before = {str(path): sha256(path) for path in (Path(__file__),
              ROOT / "analysis/experiments/conditioned_rent_correction.py",
              ROOT / "analysis/run_conditioned_rent_corrections.py",
              SOURCE_REPORT / "settings.json",
              core.SOURCE_CACHE / "prepared.joblib")}
    settings = {"source_hashes": before, "change_estimator": "preceding_three_available_changes",
                "remove_three_level_channels": True, "retain_counts_ages": True,
                "conditioned_zero_initialized_head": True, "seeds": [42,43,44],
                "selection_label_max": "2024-04"}
    location = OUT / "settings.json"
    if location.exists() and json.loads(location.read_text()) != settings:
        raise ValueError("Experiment settings changed")
    core.write_json(location, settings)
    prepared = joblib.load(core.SOURCE_CACHE / "prepared.joblib")
    torch.set_num_threads(4)
    design = "conditioned_innovations"
    if args.smoke:
        core.fit(prepared, "early_2022", 42, design, args)
    else:
        prior = [core.fit(prepared, fold, 42, design, args) for fold in ("early_2022", "early_2023")]
        epochs = max(1, int(np.floor((prior[0]["best_epoch"]+prior[1]["best_epoch"])/2+.5)))
        core.write_json(OUT / "selection.json",
                        {"earlier_best_epochs": [x["best_epoch"] for x in prior],
                         "outer_epochs": epochs, "outer_labels_used": False})
        rows = [core.fit(prepared, "outer", seed, design, args, epochs=epochs)
                for seed in (42,43,44)]
        pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
        core.write_json(OUT / "worker_status.json", {"state": "completed", "stage": args.stage})
    if {str(path): sha256(Path(path)) for path in before} != before:
        raise ValueError("Original source changed")


if __name__ == "__main__":
    main()
