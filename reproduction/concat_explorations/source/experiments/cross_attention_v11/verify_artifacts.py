"""Independently validate the completed screening artifacts and historical guard."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/cross_attention_v11"


def main():
    if (OUT / "STATUS").read_text().strip() != "COMPLETED":
        raise RuntimeError("Training is not complete")
    guard = json.loads((OUT / "root_original_guard.json").read_text())
    changed = [name for name, digest in guard.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise AssertionError(f"Historical files changed: {changed}")
    preflight = json.loads((OUT / "preflight.json").read_text())
    inputs = json.loads((OUT / "source_input_hashes.json").read_text())
    for name, digest in {**inputs, **preflight["hashes"]}.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Registered source/input changed: {name}")
    reference = None
    results = {}
    for arm in ("concat", "summary", "sequence"):
        folder = OUT / arm
        summary = json.loads((folder / "summary.json").read_text())
        curve = json.loads((folder / "curve.json").read_text())
        replay = json.loads((folder / "independent_replay.json").read_text())
        with np.load(folder / "holdout_predictions.npz") as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        actual, predicted = arrays["actual_sgd"], arrays["predicted_sgd"]
        assert actual.shape == predicted.shape == (35882,)
        assert np.isfinite(actual).all() and np.isfinite(predicted).all()
        assert len(np.unique(arrays["row_id"])) == len(actual)
        if reference is None:
            reference = arrays
        else:
            for key in ("row_id", "origin", "horizon", "actual_sgd"):
                assert np.array_equal(reference[key], arrays[key]), (arm, key)
        error = predicted.astype(np.float64) - actual.astype(np.float64)
        metrics = {"rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
                   "mae_sgd": float(np.mean(np.abs(error))),
                   "bias_sgd": float(np.mean(error)), "n": len(actual)}
        for key, value in metrics.items():
            assert np.isclose(value, summary["overall"][key], rtol=0, atol=1e-7), (arm, key)
        selected = min((row for row in curve if row["epoch"] >= 5),
                       key=lambda row: row["rmse_sgd"])
        assert selected["epoch"] == summary["selected_epoch"] == replay["epoch"]
        assert replay["max_abs_delta_sgd"] < .01
        assert replay["rows"] == len(actual)
        assert summary["parameters"] == preflight["parameters"][arm] == 87249
        assert set(map(int, summary["horizon"])) == set(np.unique(arrays["horizon"]))
        for horizon, group in summary["horizon"].items():
            subset = error[arrays["horizon"] == int(horizon)]
            assert len(subset) == group["n"]
            assert np.isclose(np.sqrt(np.mean(subset ** 2)), group["rmse_sgd"], rtol=0, atol=1e-7)
        results[arm] = {"selected_epoch": selected["epoch"], **metrics,
                        "replay_max_abs_delta_sgd": replay["max_abs_delta_sgd"]}
    payload = {"status": "PASSED", "historical_files_unchanged": len(guard),
               "registered_hashes_match": True, "paired_rows": len(reference["row_id"]),
               "arms": results}
    (OUT / "artifact_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
