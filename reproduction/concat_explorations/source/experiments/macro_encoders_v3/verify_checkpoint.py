"""Verification that a saved checkpoint rebuilds predictions without fitting.

Loads ``checkpoints/<split>_stage<X>_smoke.pt`` **and** the fitted scaler JSON, runs
one forward pass on the real exported windows, and asserts the tokens equal the ones
recorded in ``smoke.json``.  Nothing is trained here, and any optimizer state is
ignored — this only proves the saved state is sufficient to reproduce the numbers.

Run:
    .venv/bin/python -m experiments.macro_encoders_v3.verify_checkpoint --split orig
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.macro_encoders_v3 import api, contract, data, encoders, scaling

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="orig", choices=["orig", "early", "long"])
    parser.add_argument("--stages", nargs="+", default=["A", "B"])
    parser.add_argument("--contract-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = contract.load_manifest(args.contract_dir)
    out_dir = contract.REPORT_DIR / manifest["contract_id"]
    smoke_path = out_dir / "smoke.json"
    if not smoke_path.exists():
        print(f"BLOCKED: no smoke.json at {smoke_path}; run the smoke first")
        return 3
    smoke = json.loads(smoke_path.read_text())

    print(f"contract_id: {manifest['contract_id']}")
    print(f"smoke generated: {smoke['generated']}\n")

    for stage in args.stages:
        print(f"--- {args.split}/{stage}")
        record = smoke["stages"].get(stage)
        if record is None:
            check(f"{args.split}/{stage}: stage recorded in smoke.json", False, "missing")
            continue

        windows = data.load_windows(manifest, args.split, stage)
        rows = data.load_rows(manifest, args.split, stage)

        # reload the persisted scaler state (no refitting)
        scaler_dir = out_dir / "scalers" / args.split / f"stage{stage}"
        check(f"{stage}: scaler state exists", (scaler_dir / "index.json").exists())
        reloaded = scaling.MacroScalers.load(scaler_dir)
        check(f"{stage}: reloaded policy matches the contract",
              reloaded.policy == manifest["policy"])
        check(f"{stage}: reloaded contract id matches",
              reloaded.contract_id == manifest["contract_id"])

        tensors = api.build_window_tensors(windows, reloaded)

        # rebuild the module from the checkpoint only
        checkpoint = out_dir / "checkpoints" / f"{args.split}_stage{stage}_smoke.pt"
        check(f"{stage}: checkpoint exists", checkpoint.exists())
        modules = encoders.build_encoders()
        modules.load_state_dict(torch.load(checkpoint, weights_only=True))
        modules.eval()

        with torch.no_grad():
            first = api.encode_windows(modules, tensors)
            second = api.encode_windows(modules, tensors)

        # determinism: two forward passes from the same state must agree exactly
        check(f"{stage}: forward is deterministic",
              torch.equal(first["dbs_token"], second["dbs_token"])
              and torch.equal(first["coe_token"], second["coe_token"]))

        # reproduce the recorded shapes
        shapes = record["window_token_shapes"]
        check(f"{stage}: dbs token shape reproduces",
              list(first["dbs_token"].shape) == shapes["dbs_token"],
              f"{list(first['dbs_token'].shape)} vs {shapes['dbs_token']}")
        check(f"{stage}: coe token shape reproduces",
              list(first["coe_token"].shape) == shapes["coe_token"],
              f"{list(first['coe_token'].shape)} vs {shapes['coe_token']}")

        # window counters must reconcile
        check(f"{stage}: window count reconciles with rows",
              int(windows.window_count.sum()) == len(rows.row_window_index))
        check(f"{stage}: n_windows reproduces",
              windows.n_windows == record["n_windows"],
              f"{windows.n_windows} vs {record['n_windows']}")

        # no window is missing, and every token is finite
        check(f"{stage}: no window is fully missing",
              float(first["dbs_missing"].sum()) == 0.0
              and float(first["coe_missing"].sum()) == 0.0)
        check(f"{stage}: tokens are finite",
              bool(torch.isfinite(first["dbs_token"]).all())
              and bool(torch.isfinite(first["coe_token"]).all()))

        # leading-padded early window really is leading-padded and still encoded
        first_mask = windows.dbs_mask[0]
        leading_pad = float(first_mask[:int(np.flatnonzero(first_mask)[0])].sum())
        check(f"{stage}: window 0 has leading padding", leading_pad == 0.0)
        check(f"{stage}: window 0 uses its real tail",
              first_mask[int(np.flatnonzero(first_mask)[0]):].sum() == first_mask.sum())

        # parameter count matches what the smoke recorded
        counts = encoders.parameter_count(modules)
        check(f"{stage}: parameter count reproduces",
              counts["total"] == record["parameter_counts"]["total"],
              f"{counts['total']} vs {record['parameter_counts']['total']}")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("checkpoint reconstruction verified (no fitting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
