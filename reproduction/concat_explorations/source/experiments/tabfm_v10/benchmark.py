"""TabFM anchor-residual benchmark on frozen orig/A only. No paid APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "reports/diagnostics_phase2_c/frozen"
BASELINE = ROOT / "reports/feature_transformer_v4/runs/v4-search-mps-s42/trial6/inner_predictions.npz"
SOURCE_COMMIT = "fbb665569425fd2f490c6576b3af967876fe11ff"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_populations(fit, hold):
    for name, data in (("fit", fit), ("hold", hold)):
        n = len(data["row_id"])
        if len(np.unique(data["row_id"])) != n:
            raise ValueError(f"{name}: duplicate IDs")
        if any(len(data[k]) != n for k in data):
            raise ValueError(f"{name}: mismatched row counts")
        expected = (data["target"].astype(float) + data["anchor_ksgd"]) * 1000
        if not np.allclose(expected, data["actual_sgd"], rtol=0, atol=.002):
            raise ValueError(f"{name}: broken anchor residual units")
        if not np.isfinite(data["numeric"]).all():
            raise ValueError(f"{name}: non-finite numeric features")
    if np.intersect1d(fit["row_id"], hold["row_id"]).size:
        raise ValueError("fit/hold ID overlap")
    # All context labels must have been known by the fixed validation origin.
    if np.max(fit["origin"] + fit["horizon"]) > np.min(hold["origin"]):
        raise ValueError("context labels cross the validation origin")


def load_frozen(directory=FROZEN):
    populations = []
    for name in ("afit", "ahold"):
        with np.load(Path(directory) / f"{name}.npz", allow_pickle=False) as arrays:
            populations.append({k: arrays[k] for k in ("numeric", "categorical", "anchor_ksgd",
                "target", "actual_sgd", "row_id", "horizon", "origin")})
    validate_populations(*populations)
    names = json.loads((Path(directory) / "afit_columns.json").read_text())["codes"]
    fit, hold = populations
    n_numeric = fit["numeric"].shape[1]
    if n_numeric != 56 or fit["categorical"].shape[1] != 4 or len(names) != 60:
        raise ValueError("expected frozen v4 56 numeric + 4 categorical schema")
    return fit, hold, names[:n_numeric], names[n_numeric:]


def sample_context(fit, n_rows, seed, policy):
    if n_rows < 1:
        raise ValueError("context_rows must be positive")
    n = len(fit["row_id"])
    if n_rows >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    weights = None
    if policy == "recent":
        # Prefer recent known target months without discarding less common flat types.
        month = fit["origin"] + fit["horizon"]
        flat_type = fit["categorical"][:, 1]
        _, inverse, counts = np.unique(flat_type, return_inverse=True, return_counts=True)
        weights = np.power(2., (month - month.max()) / 12.) / np.sqrt(counts[inverse])
        weights /= weights.sum()
    elif policy != "uniform":
        raise ValueError("context policy must be recent or uniform")
    return np.sort(rng.choice(n, min(n_rows, n), replace=False, p=weights))


def feature_frame(data, numeric_names, categorical_names):
    frame = pd.DataFrame(data["numeric"], columns=numeric_names)
    for i, name in enumerate(categorical_names):
        frame[name] = pd.Series([None if code == 0 else f"code_{int(code)}"
                                for code in data["categorical"][:, i]], dtype=object)
    return frame


def reconstruct(residual, anchor_ksgd):
    return (np.asarray(residual, dtype=float) + np.asarray(anchor_ksgd, dtype=float)) * 1000.


def aligned_baseline(path, hold):
    with np.load(path, allow_pickle=False) as arrays:
        ids = arrays["row_id"]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("baseline duplicate IDs")
        lookup = pd.Index(ids).get_indexer(hold["row_id"])
        if np.any(lookup < 0):
            raise ValueError("baseline missing validation IDs")
        if not np.array_equal(arrays["actual_sgd"][lookup], hold["actual_sgd"]):
            raise ValueError("baseline label mismatch")
        if not np.array_equal(arrays["horizon"][lookup], hold["horizon"]):
            raise ValueError("baseline horizon mismatch")
        return arrays["predicted_sgd"][lookup]


def check_quota(log_path, *, now=None, stop_percent=90., max_age_seconds=600):
    """Fail closed on stale/missing quota evidence; never enables credit spending."""
    now_dt = datetime.fromisoformat(now.replace("Z", "+00:00")) if now else datetime.now(timezone.utc)
    path = Path(log_path)
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 2_000_000))
        lines = stream.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        if '"rate_limits"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload", {})
        limits = payload.get("rate_limits")
        if payload.get("type") != "token_count" or not limits:
            continue
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        age = (now_dt - timestamp).total_seconds()
        if age > max_age_seconds or age < -60:
            raise RuntimeError("quota evidence stale; stop before further model work")
        used = {}
        for name in ("primary", "secondary"):
            value = (limits.get(name) or {}).get("used_percent")
            if value is None or not 0 <= value <= 100:
                raise RuntimeError("quota remaining allowance unavailable")
            used[f"{name}_used_percent"] = value
            if value >= stop_percent:
                raise RuntimeError(f"quota {name} used {value}%; stopping before credits")
        return {"timestamp": event["timestamp"], **used}
    raise RuntimeError("quota event unavailable; stop before model work")


def new_run_directory(path):
    Path(path).mkdir(parents=True, exist_ok=False)
    return Path(path)


def metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"n": len(error), "rmse_sgd": float(np.sqrt(np.mean(error**2))),
            "mae_sgd": float(np.abs(error).mean()), "bias_sgd": float(error.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--context-rows", type=int, default=512)
    parser.add_argument("--context-policy", choices=("recent", "uniform"), default="recent")
    parser.add_argument("--validation-rows", type=int, default=512, help="0 = all A validation rows")
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--predict-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--usage-log", required=True, type=Path)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "reports/tabfm_v10/runtime")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "reports/tabfm_v10/weights/regression")
    args = parser.parse_args(argv)
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id must be a simple directory name")
    if min(args.context_rows, args.estimators, args.predict_batch) < 1 or args.validation_rows < 0:
        raise ValueError("invalid inference budget")
    quota = check_quota(args.usage_log)
    fit, hold, numeric_names, categorical_names = load_frozen()
    context = sample_context(fit, args.context_rows, args.seed, args.context_policy)
    if args.validation_rows and args.validation_rows < len(hold["row_id"]):
        # Uniform without replacement; selection independent of all validation labels.
        selected = np.sort(np.random.default_rng(args.seed + 1).choice(
            len(hold["row_id"]), args.validation_rows, replace=False))
        hold = {k: v[selected] for k, v in hold.items()}
    baseline = aligned_baseline(BASELINE, hold)
    manifest = {
        "status": "PREPARED", "scope": "orig/A viewed development window only",
        "source_commit": SOURCE_COMMIT, "seed": args.seed, "quota_snapshot": quota,
        "context_rows": len(context), "available_fit_rows": len(fit["row_id"]),
        "validation_rows": len(hold["row_id"]), "context_policy": args.context_policy,
        "target": "actual_sgd/1000 - anchor_ksgd; add anchor back for SGD scoring",
        "features": numeric_names + categorical_names, "estimators": args.estimators,
        "predict_batch": args.predict_batch, "dtype": args.dtype, "device": args.device,
        "baseline_same_rows": metrics(hold["actual_sgd"], baseline),
        "input_hashes": {p.name: sha256(p) for p in (FROZEN / "afit.npz", FROZEN / "ahold.npz",
                                                    FROZEN / "afit_columns.json", BASELINE)},
        "runner_sha256": sha256(__file__), "no_paid_api": True,
        "selection_note": "fixed context/sample selected without validation labels; no internal NNLS tuning",
    }
    out = new_run_directory(ROOT / "reports/tabfm_v10/runs" / args.run_id)
    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save()
    np.savez_compressed(out / "selection.npz", context_row_id=fit["row_id"][context],
                        validation_row_id=hold["row_id"])
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    started = time.perf_counter()
    try:
        check_quota(args.usage_log)
        if not (args.weights_dir / "model.safetensors").is_file():
            raise FileNotFoundError("local regression model.safetensors missing; no automatic paid fallback")
        for sub in ("dependencies", "source"):
            sys.path.insert(0, str(args.runtime_dir / sub))
        import torch
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
        torch.set_num_threads(2)
        model = tabfm_v1_0_0_pytorch.load(model_type="regression", checkpoint_path=str(args.weights_dir),
            device=args.device, dtype=torch.bfloat16 if args.dtype == "bfloat16" else None)
        reg = TabFMRegressor(model=model, n_estimators=args.estimators, random_state=args.seed,
            max_num_rows=len(context), max_num_features=60, batch_size=1, enable_nnls=False,
            cache_context=True, maybe_quantize_kv_cache=False)
        fit_frame = feature_frame(fit, numeric_names, categorical_names)
        reg.fit(fit_frame.iloc[context].reset_index(drop=True), fit["target"][context].astype(float))
        query = feature_frame(hold, numeric_names, categorical_names)
        chunks = []
        manifest["status"] = "RUNNING"
        save()
        for start in range(0, len(query), args.predict_batch):
            check_quota(args.usage_log)
            predicted = reg.predict(query.iloc[start:start + args.predict_batch])
            if not np.isfinite(predicted).all():
                raise ValueError("TabFM returned non-finite predictions")
            chunks.append(predicted)
            print(f"predicted {min(start + args.predict_batch, len(query))}/{len(query)}", flush=True)
        predicted = reconstruct(np.concatenate(chunks), hold["anchor_ksgd"])
        np.savez_compressed(out / "predictions.npz", row_id=hold["row_id"], origin=hold["origin"],
            horizon=hold["horizon"], actual_sgd=hold["actual_sgd"], predicted_sgd=predicted,
            baseline_sgd=baseline)
        manifest.update(status="COMPLETED", elapsed_seconds=time.perf_counter() - started,
            metrics=metrics(hold["actual_sgd"], predicted),
            fixed_equal_blend=metrics(hold["actual_sgd"], (predicted + baseline) / 2),
            parameters=sum(p.numel() for p in model.parameters()),
            per_horizon={str(int(h)): metrics(hold["actual_sgd"][mask], predicted[mask])
                for h in np.unique(hold["horizon"]) for mask in [hold["horizon"] == h]})
        save()
        print(json.dumps(manifest, indent=2))
    except Exception as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - started)
        save()
        raise


if __name__ == "__main__":
    main()
