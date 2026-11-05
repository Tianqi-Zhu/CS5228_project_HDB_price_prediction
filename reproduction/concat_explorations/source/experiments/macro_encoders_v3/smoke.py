"""Real-data smoke run for the v3 macro encoders.

Encodes the actual exported `orig` stage-A and stage-B windows, prints shapes and
token statistics, checks both branches receive gradient, and saves the fitted
scalers plus a checkpoint under ``reports/macro_encoders_v3/<contract>/``.

This is a smoke test, not a rental-prediction score: the loss here is a toy
surrogate, so nothing in its output says whether macro features help.

Run:
    .venv/bin/python -m experiments.macro_encoders_v3.smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.macro_encoders_v3 import api, contract, data, encoders, scaling


def run_stage(manifest: dict, split_id: str, stage: str, out_dir: Path,
              seed: int = 42) -> dict:
    started = time.perf_counter()
    windows = data.load_windows(manifest, split_id, stage)
    rows = data.load_rows(manifest, split_id, stage)
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)

    torch.manual_seed(seed)
    modules = encoders.build_encoders()
    modules.train()
    counts = encoders.parameter_count(modules)

    # forward on the unique windows
    out = api.encode_windows(modules, tensors)
    # toy surrogate so both branches get a gradient
    loss = out["dbs_token"].pow(2).mean() + out["coe_token"].pow(2).mean()
    loss.backward()

    grad = {
        "dbs_lstm_weight_ih_l0_abs_sum": float(
            modules.dbs.lstm.weight_ih_l0.grad.abs().sum()),
        "dbs_projection_abs_sum": float(modules.dbs.projection.weight.grad.abs().sum()),
        "coe_gru_weight_ih_l0_abs_sum": float(
            modules.coe.gru.weight_ih_l0.grad.abs().sum()),
        "coe_projection_abs_sum": float(modules.coe.projection.weight.grad.abs().sum()),
    }

    # row-level gather over the whole stage, on the live graph
    modules.zero_grad()
    per_row = api.macro_tokens_for_rows(modules, tensors, rows.row_window_index, rows.row_id)
    row_shapes = {key: list(value.shape) for key, value in per_row.items()
                  if torch.is_tensor(value)}

    summary = api.summarize_tokens(per_row)

    # persist the fitted state and a checkpoint
    scaler_dir = out_dir / "scalers" / split_id / f"stage{stage}"
    scalers.save(scaler_dir)
    checkpoint = out_dir / "checkpoints" / f"{split_id}_stage{stage}_smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modules.state_dict(), checkpoint)

    # confirm the saved state reloads and reproduces the tokens exactly
    modules.eval()
    with torch.no_grad():
        before = api.encode_windows(modules, tensors)
    restored = encoders.build_encoders()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    with torch.no_grad():
        after = api.encode_windows(restored, tensors)
    reproduced = bool(torch.equal(before["dbs_token"], after["dbs_token"])
                      and torch.equal(before["coe_token"], after["coe_token"]))

    elapsed = time.perf_counter() - started
    return {
        "split_id": split_id,
        "stage": stage,
        "policy": windows.policy,
        "contract_id": windows.contract_id,
        "n_rows": int(len(rows.row_window_index)),
        "n_windows": windows.n_windows,
        "window_count_sum": int(windows.window_count.sum()),
        "dbs_seq_shape": list(tensors.dbs_seq.shape),
        "coe_seq_shape": list(tensors.coe_seq.shape),
        "parameter_counts": counts,
        "window_token_shapes": {
            "dbs_token": list(out["dbs_token"].shape),
            "coe_token": list(out["coe_token"].shape),
            "dbs_missing": list(out["dbs_missing"].shape),
            "coe_missing": list(out["coe_missing"].shape),
        },
        "row_token_shapes": row_shapes,
        "token_summary": summary,
        "gradient": grad,
        "gradients_nonzero": all(value > 0 for value in grad.values()),
        "checkpoint_reproduces_tokens_exactly": reproduced,
        "dbs_windows_with_history": int(windows.dbs_has_history().sum()),
        "coe_windows_with_history": int(windows.coe_has_history().sum()),
        "scaler_fit_scope": scalers.dbs.fitted_on,
        "seconds": round(elapsed, 2),
        "note": "toy surrogate loss for gradient checking; not a rental-prediction score",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="orig", choices=["orig", "early", "long"])
    parser.add_argument("--stages", nargs="+", default=["A", "B"])
    parser.add_argument("--contract-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = contract.load_manifest(args.contract_dir)
    out_dir = contract.REPORT_DIR / manifest["contract_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"contract_id: {manifest['contract_id']}")
    print(f"policy:      {manifest['policy']}\n")

    results = {}
    for stage in args.stages:
        print(f"--- {args.split}/{stage}")
        payload = run_stage(manifest, args.split, stage, out_dir)
        results[stage] = payload
        print(f"  rows {payload['n_rows']:,} -> windows {payload['n_windows']}")
        print(f"  dbs {payload['dbs_seq_shape']} -> token {payload['window_token_shapes']['dbs_token']}")
        print(f"  coe {payload['coe_seq_shape']} -> token {payload['window_token_shapes']['coe_token']}")
        print(f"  params {payload['parameter_counts']['total']:,} "
              f"({payload['parameter_counts']['by_submodule']})")
        print(f"  gradients non-zero: {payload['gradients_nonzero']}  {payload['gradient']}")
        print(f"  checkpoint reproduces exactly: {payload['checkpoint_reproduces_tokens_exactly']}")
        print(f"  {payload['seconds']}s", flush=True)

    def sha256_file(path: Path) -> str:
        import hashlib
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    report = {
        "generated": np.datetime64("now").astype("datetime64[s]").astype(str),
        "contract_id": manifest["contract_id"],
        "policy": manifest["policy"],
        "source_hashes": manifest["source_hashes"],
        # hashes of the exact code that produced these checkpoints, so a later
        # reader can prove which implementation wrote them
        "code_hashes": {path.name: sha256_file(path)
                        for path in sorted(contract.PACKAGE_DIR.glob("*.py"))},
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "stages": results,
    }
    path = out_dir / "smoke.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(contract.ROOT)}")

    failed = [stage for stage, payload in results.items()
              if not payload["gradients_nonzero"]
              or not payload["checkpoint_reproduces_tokens_exactly"]]
    if failed:
        print(f"FAILED stages: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
