"""Synthetic history export for developing the adapter before the real file lands.

Writes the exact key set the producer's `DATA_READY.json` will describe, including
`history_observed`, so the loader is exercised against the real contract rather than a guess.
Missing cells follow the producer's policy: `count == 0` means no group history, the mean/std
family is NaN in `history_raw`, `recency` is the sentinel `-1`, and `history_observed` is 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from experiments.history_fusion_v9 import data as HD

ROOT = Path(__file__).resolve().parents[2]
V8_DATA = ROOT / "reports" / "target_asof_v8" / "data"


def build(out_dir: Path, data_dir_for_v8: Path = V8_DATA, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    from experiments.target_asof_fusion_v8 import run as V8R

    generator = np.random.default_rng(seed)
    schema = json.loads((ROOT / "reports/history_fusion_v9/data/schema.json").read_text()) \
        if (ROOT / "reports/history_fusion_v9/data/schema.json").exists() else None
    if schema is None:
        names = ([f"hist_{level}__{stat}" for level in HD.LEVELS for stat in HD.STATS]
                 + ["history_cutoff_age"])
        schema = {"dataset": "synthetic", "levels": list(HD.LEVELS),
                  "stats": list(HD.STATS), "feature_names": names, "n_features": 50}

    payload = {}
    for population, split_name in (("train", "train"), ("validation", "val")):
        base = V8R.load_split(split_name, data_dir_for_v8, "orig")
        rows = len(base["row_id"])
        raw = generator.normal(2000.0, 300.0, size=(rows, 50))
        observed = np.ones((rows, 50), dtype=bool)

        # every 7th row has no history at all: count 0, NaN stats, recency sentinel -1
        empty = np.zeros(rows, dtype=bool)
        empty[::7] = True
        for level in range(7):
            block = slice(level * 7, (level + 1) * 7)
            raw[empty, block.start] = 0.0                 # count
            raw[empty, block.start + 1:block.start + 6] = np.nan
            raw[empty, block.start + 6] = -1.0             # recency sentinel
            observed[empty, block] = False
        target_month = np.asarray(base["target_month"], dtype=np.int64)
        cutoff = np.where(target_month - 3 > 26, 26, target_month - 3)
        # training rows have age 3; validation rows grow past that once the fold cap binds
        raw[:, 49] = (target_month - cutoff).astype(np.float64)
        observed[:, 49] = True

        numeric = raw.copy()
        numeric = np.nan_to_num(numeric, nan=0.0)
        numeric[:, 0::7] = np.log1p(np.clip(numeric[:, 0::7], 0, None))
        numeric = ((numeric - numeric.mean(0)) / (numeric.std(0) + 1e-9)).astype(np.float32)
        # column 49 is the policy gap and is passed through unscaled, exactly as the producer
        # documents; the loader cross-checks it against history_cutoff_age
        numeric[:, 49] = raw[:, 49].astype(np.float32)

        payload[f"{population}__row_id"] = np.asarray(base["row_id"], dtype=np.int64)
        payload[f"{population}__target_month"] = target_month.astype(np.int32)
        payload[f"{population}__actual_sgd"] = np.asarray(base["actual_sgd"], dtype=np.float64)
        payload[f"{population}__history_raw"] = raw
        payload[f"{population}__history_numeric"] = numeric
        payload[f"{population}__history_observed"] = observed
        payload[f"{population}__cutoff_month"] = cutoff.astype(np.int32)
        payload[f"{population}__history_cutoff_age"] = (target_month - cutoff).astype(np.int32)

    np.savez_compressed(out_dir / "orig_history.npz", **payload)
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    (out_dir / "orig_manifest.json").write_text(json.dumps(
        {"fold": "orig", "synthetic": True,
         "note": "fixture only; never a result"}, indent=2))

    digest = HD.sha_of(out_dir / "orig_history.npz")
    (out_dir / "DATA_READY.json").write_text(json.dumps({
        "status": "ready",
        "synthetic": True,
        "files": {"orig_history.npz": {"sha256": digest}},
    }, indent=2))
    return {"rows": {k: int(v.shape[0]) for k, v in payload.items() if v.ndim >= 1
                     and k.endswith("__row_id")}, "sha256": digest}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/history_fusion_v9/data_synthetic")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    info = build(out)
    print(f"wrote synthetic history export to {out}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
