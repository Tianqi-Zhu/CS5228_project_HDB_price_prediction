"""Synthetic fixture matching the declared v8 data contract.

Exists so the model, runner and reload path can be developed and smoke-tested before the real
data package lands. The producer's files remain the only authority for real runs; this writes to
a scratch directory and marks every artefact as synthetic.

The fixture deliberately contains the three window shapes the market encoder must handle:

* rows whose window starts before the series exists (leading padding, `step_valid=0`);
* rows with an interior month that has no observation (`step_valid=1`, `observed=0`);
* rows with nothing observed at all (the learned missing vector).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

WINDOW_MONTHS = 12


def build_fixture(out_dir: Path, n_train: int = 768, n_val: int = 512,
                  n_numeric: int = 21, cardinalities=(27, 7, 19),
                  channels: int = 8, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)

    numeric_names = [f"num_{i:02d}" for i in range(n_numeric)]
    schema = {
        "synthetic": True,
        "house_numeric_order": numeric_names,
        "house_cardinality": list(cardinalities),
        "house_categorical_order": ["TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"],
        "calendar_features": ["target_month_sin", "target_month_cos"],
        "market_channels": [f"ch_{i:02d}" for i in range(channels)],
        "market_window_months": WINDOW_MONTHS,
        "target": "MONTHLY_RENT/1000",
        "date_assumption": ("SYNTHETIC: calendar months, oldest->newest, values usable from "
                           "the following month; no publication timing is modelled"),
        "no_rent_history": True,
        "no_anchor": True,
        "note": "fixture only; never a result",
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(
        {**schema, "purpose": "SYNTHETIC_FIXTURE", "n_train": n_train, "n_val": n_val},
        indent=2))

    def make_split(name: str, rows: int, windows: int, month_start: int) -> None:
        target_month = generator.integers(month_start, month_start + 24, size=rows)
        window_id = generator.integers(0, windows, size=rows)

        values = generator.normal(0.0, 1.0, size=(windows, WINDOW_MONTHS, channels)).astype(
            np.float32)
        observed = np.ones((windows, WINDOW_MONTHS, channels), dtype=np.float32)
        step_valid = np.ones((windows, WINDOW_MONTHS), dtype=np.float32)

        # window shapes: first quarter padded out, second quarter with an interior hole,
        # last quarter fully missing
        quarter = max(1, windows // 4)
        step_valid[:quarter, :4] = 0.0
        observed[:quarter, :4] = 0.0
        values[:quarter, :4] = 0.0
        interior = slice(quarter, 2 * quarter)
        observed[interior, 6] = 0.0
        values[interior, 6] = 0.0
        step_valid[interior, 6] = 1.0          # the month exists, it just has no value
        empty = slice(2 * quarter, 3 * quarter)
        observed[empty] = 0.0
        values[empty] = 0.0
        step_valid[empty] = 1.0

        house_numeric = generator.normal(size=(rows, n_numeric)).astype(np.float32)
        house_cat = np.stack([generator.integers(0, c, size=rows)
                              for c in cardinalities], axis=1).astype(np.int64)
        calendar = np.stack([np.sin(target_month / 12.0), np.cos(target_month / 12.0)],
                            axis=1).astype(np.float32)
        # target depends weakly on the house columns and on the window's newest values, so the
        # arms are not all predicting the same constant
        newest = values[window_id][:, -1].mean(axis=1)
        target_sgd = (1800.0
                      + 400.0 * house_numeric[:, 0]
                      + 150.0 * newest
                      + generator.normal(0.0, 50.0, size=rows))
        target_ksgd = (target_sgd / 1000.0).astype(np.float32)

        np.savez_compressed(
            out_dir / f"{name}.npz",
            house_numeric=house_numeric, house_cat=house_cat, calendar=calendar,
            target_ksgd=target_ksgd, actual_sgd=target_sgd.astype(np.float64),
            row_id=(np.arange(rows) + (0 if name == "train" else 10_000)).astype(np.int64),
            target_month=target_month.astype(np.int64),
            window_id=window_id.astype(np.int64))
        np.savez_compressed(
            out_dir / f"{name}__market.npz",
            market_values=values, market_observed=observed, market_step_valid=step_valid,
            market_month_index=np.tile(np.arange(WINDOW_MONTHS, dtype=np.int32),
                                       (windows, 1)))

    make_split("train", n_train, max(8, n_train // 24), month_start=0)
    make_split("val", n_val, max(8, n_val // 24), month_start=24)
    return schema


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/target_asof_v8/data_synthetic")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[2] / out
    schema = build_fixture(out)
    print(f"wrote synthetic fixture to {out} ({len(schema['house_numeric_order'])} numeric, "
          f"{schema['market_window_months']} months, {len(schema['market_channels'])} channels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
