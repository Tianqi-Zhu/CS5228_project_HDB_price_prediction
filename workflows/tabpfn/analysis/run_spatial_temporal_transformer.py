"""Prepare, queue, train and audit randomly initialized rental transformers.

CPU-only setup: --stage prepare. GPU work always passes the capacity gate.
All output goes to new report/cache directories; no submissions are written.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from analysis.config import AUX, DATA, ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import (
    CONTROLS, EXPERIMENTS, PRIMARY, SENSITIVITIES, SEQUENCE_CHANNELS,
    FeaturePreprocessor, RentalTransformer, SequenceBank, SequencePreprocessor,
    adjusted_rents, episode_origins, forecast_context, macro_panel, rebuild_histories,
)
from analysis.experiments.spatiotemporal_hybrid import spatial_frame

OUT = REPORTS / "spatial_temporal_transformer"
CACHE = ROOT / ".cache/spatial_temporal_transformer"
BASE = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
FOLDS = {
    "early_2022": ("2022-09", "2023-06", "calendar_inner_2022-09.joblib"),
    "early_2023": ("2023-07", "2024-04", "calendar_inner_2023-07.joblib"),
    "outer": ("2024-06", "2025-03", None),
    "long16": ("2023-12", "2025-03", None),
}
SCHEMA_VERSION = 1
MAX_EPOCHS, PATIENCE = 30, 5


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Settings changed: use a new output directory ({path})")
    write_json(path, value)


def sources():
    return [DATA / "train.csv", DATA / "test.csv", AUX / "sg-coe-prices.csv",
            AUX / "sg-stock-prices.csv", AUX / "sg-hdb-block.csv",
            AUX / "sg-mrt-stations.csv", AUX / "sg-shopping-malls.csv", AUX / "sg-schools.csv",
            BASE, ROOT / ".cache/tabpfn_v3_features/calendar_inner_2022-09.joblib",
            ROOT / ".cache/tabpfn_v3_features/calendar_inner_2023-07.joblib",
            REPORTS / "tabpfn_feature_ensemble_final_submission/submission_tabpfn_optimized_ensemble_150k.csv",
            REPORTS / "tabpfn_diagscale_mse_final_submission/submission_tabpfn_diagscale_mse_150k.csv"]


def fingerprint():
    paths = sources() + [Path(__file__), ROOT / "analysis/experiments/spatial_temporal_transformer.py",
                         ROOT / "analysis/experiments/gpu_capacity.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


@contextmanager
def run_lock():
    """One worker for this experiment, including CPU preparation and GPU waits."""
    path = OUT / "worker.lock"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Another worker owns {path}; check its PID before removing a stale lock") from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": SCHEMA_VERSION, "hashes": fingerprint(),
        "folds": FOLDS, "variants": {k: v.settings() for k, v in EXPERIMENTS.items()},
        "training_origins": "deterministic label-independent horizons 1..16; initial cold rows retained",
        "rental_cutoff": "row forecast origin, frozen for every query path",
        "macro_cutoff": "one calendar month before rental cutoff; no future realized exogenous path",
        "delay_sensitivity": "all 46 row histories, baseline and sequence shift one month earlier",
        "baseline_policy": "original local forecast, flat beyond horizon 10; real horizon/age retained",
        "adjusted_rent": "2021-only property reference, masked at origins before 2021-12",
        "stock_proxy": "median .SI monthly adjusted-close log return; supplied snapshot, not STI",
        "sequence_channels": SEQUENCE_CHANNELS,
        "property_architecture": {"blocks": 2, "dimension": 128, "heads": 4, "dropout": .1,
                                  "numeric_missingness": "learned vector within each feature token",
                                  "spatial_wavelengths_km": [.25, .5, 1, 2, 4, 8, 16]},
        "temporal_architecture": {"blocks": 1, "dimension": 32, "heads": 2,
                                  "ffn": 64, "dropout": .1, "lstm_hidden": 8,
                                  "linear_control": "jointly trained regularized linear lag head, not closed-form Ridge"},
        "optimization": {"lr": .0003, "weight_decay": .001, "max_epochs": MAX_EPOCHS,
                         "patience": PATIENCE, "batch": 256, "gradient_clip": 1,
                         "loss": "standardized individual residual SGD MSE",
                         "sequence_penalty": .01, "precision": "BF16 where supported; FP32 loss"},
        "selection": "per variant, rounded median best epoch from two earlier disjoint weighted windows",
        "long16_selection": "epoch from early_2022 only, labels ending 2023-06, before long16 origin",
        "seeds": [42, 43, 44], "outer_target_selection": False,
        "versions": {p: version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "joblib")},
    }
    immutable(OUT / "settings.json", config)
    cache_path, manifest_path = CACHE / "prepared.joblib", OUT / "prepared.json"
    if cache_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if sha256(cache_path) != manifest["sha256"]:
            raise ValueError("Prepared feature cache changed")
        print("REUSE CPU preparation", flush=True)
        return joblib.load(cache_path)
    source_manifest = json.loads((REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json").read_text())
    if sha256(BASE) != source_manifest["sha256"]:
        raise ValueError("Source 77-feature artifact disagrees with its manifest")
    base = joblib.load(BASE)
    raw = pd.read_csv(DATA / "train.csv")
    frame = spatial_frame(raw)
    template = pd.concat([base["X_train"], base["X_query"]]).sort_index()
    if not template.index.equals(frame.index):
        raise ValueError("Source feature IDs do not match raw rental rows")
    for name in ("BLOCK_ID", "STREET_ID"):
        frame[name] = template[name].astype(str)
    adjusted, reference = adjusted_rents(frame)
    macro = macro_panel(pd.read_csv(AUX / "sg-coe-prices.csv"), pd.read_csv(AUX / "sg-stock-prices.csv"))
    bank = SequenceBank(frame, adjusted, macro)
    months = pd.PeriodIndex(frame._MONTH, freq="M")
    origins = episode_origins(months, frame.index)
    result = {"folds": {}, "sequence_bank": bank}
    train_variants = {}
    for delay in (0, 1):
        stage_cache = CACHE / f"training_features_d{delay}.joblib"
        stage_manifest = OUT / f"training_features_d{delay}.json"
        if stage_cache.exists() and stage_manifest.exists():
            if sha256(stage_cache) != json.loads(stage_manifest.read_text())["sha256"]:
                raise ValueError("Intermediate training cache changed")
            train_variants[delay] = joblib.load(stage_cache)
            print(f"REUSE training feature preparation delay={delay}", flush=True)
            continue
        delayed = origins - delay
        delayed = pd.PeriodIndex([o if pd.notna(o) and o >= months.min() else pd.NaT for o in delayed], freq="M")
        print(f"CPU rebuild training histories: delay={delay}, origins={len(delayed.dropna().unique())}", flush=True)
        x, baseline = rebuild_histories(template, frame, delayed)
        sequences = bank.sequences(frame, delayed)
        train_variants[delay] = {"x": x, "baseline": baseline, "sequences": sequences,
                                 "origins": delayed, "context": forecast_context(months, delayed)}
        joblib.dump(train_variants[delay], stage_cache, compress=3)
        write_json(stage_manifest, {"sha256": sha256(stage_cache)})
    audit = []
    for name, (start, end, sampled_cache) in FOLDS.items():
        start, end = pd.Period(start, "M"), pd.Period(end, "M")
        fit_ids = frame.index[months < start]
        if sampled_cache:
            old = joblib.load(ROOT / ".cache/tabpfn_v3_features" / sampled_cache)
            query_ids = old["X_query"].index
            weights = old["query_weights"]
        else:
            query_ids = frame.index[(months >= start) & (months <= end)]
            weights = np.ones(len(query_ids))
        for delay in (0, 1):
            key = f"{name}_d{delay}"
            origin = start - 1 - delay
            q_origins = pd.PeriodIndex([origin]*len(query_ids), freq="M")
            print(f"CPU query features {key}: fit={len(fit_ids)} query={len(query_ids)} cutoff={origin}", flush=True)
            qx, qb = rebuild_histories(template.loc[query_ids], frame, q_origins)
            t = train_variants[delay]
            result["folds"][key] = {
                "X_train": t["x"].loc[fit_ids], "X_query": qx,
                "y_train": frame.loc[fit_ids, "MONTHLY_RENT"].to_numpy(float),
                "y_query": frame.loc[query_ids, "MONTHLY_RENT"].to_numpy(float),
                "train_baseline": t["baseline"][fit_ids], "query_baseline": qb,
                "train_sequence": t["sequences"][fit_ids],
                "query_sequence": bank.sequences(frame.loc[query_ids], q_origins),
                "train_context": t["context"][fit_ids],
                "query_context": forecast_context(months[query_ids], q_origins),
                "train_month": months[fit_ids].astype(str).to_numpy(),
                "query_month": months[query_ids].astype(str).to_numpy(),
                "query_weights": np.asarray(weights, float),
                "train_origins": t["origins"][fit_ids], "query_origins": q_origins,
            }
            b = result["folds"][key]
            assert set(fit_ids).isdisjoint(query_ids)
            assert months[fit_ids].max() < months[query_ids].min()
            eligible = ~b["train_origins"].isna()
            assert (b["train_origins"][eligible] < months[fit_ids][eligible]).all()
            audit.append({"fold": key, "fitting_rows": len(fit_ids), "query_rows": len(query_ids),
                          "query_label_min": str(months[query_ids].min()),
                          "query_label_max": str(months[query_ids].max()),
                          "rental_cutoff": str(origin), "macro_cutoff": str(origin-1),
                          "max_query_horizon": int((months[query_ids].asi8-origin.ordinal).max()),
                          "train_cold_rows": int((~eligible).sum())})
    # Large source tables are deliberately not stored in the model bundle.
    result.pop("sequence_bank")
    joblib.dump(reference, CACHE / "property_reference.joblib", compress=3)
    print("CPU saving audited feature bundle", flush=True)
    joblib.dump(result, cache_path, compress=3)
    pd.DataFrame(audit).to_csv(OUT / "availability.csv", index=False)
    histogram = pd.Series(months.asi8[~origins.isna()] - origins.asi8[~origins.isna()]).value_counts().sort_index()
    histogram.rename_axis("horizon").rename("rows").to_csv(OUT / "training_horizons.csv")
    write_json(manifest_path, {"sha256": sha256(cache_path), "folds": audit,
                              "features": list(template), "cpu_preparation": True,
                              "no_submission_writes": True})
    return result


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def arrays(bundle, pre, seq_pre, spec, side):
    prop = pre.transform(bundle[f"X_{side}"])
    sequence = seq_pre.transform(bundle[f"{side}_sequence"][:, -spec.window:], spec.information)
    return (*prop, sequence, bundle[f"{side}_context"])


@torch.inference_mode()
def prediction(model, values, mean, sd, device, batch=512):
    model.eval()
    result = []
    for start in range(0, len(values[0]), batch):
        x = [v[start:start+batch].to(device) for v in values]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result.append(model(*x).float().cpu().numpy().astype(float)*sd+mean)
    values = np.concatenate(result)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite model predictions")
    return values


def weighted_rmse(actual, predicted, weights):
    return float(np.sqrt(np.average((np.asarray(actual)-predicted)**2, weights=weights)))


def capacity_gate(args):
    if args.device == "cuda":
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        wait_for_gpu(min_free_gib=args.min_free_gib, max_utilization=args.max_gpu_utilization,
                     stable_samples=2, poll_seconds=args.gpu_poll_seconds,
                     status_path=OUT / "gpu_status.json")


def train(spec, bundle, run_name, seed, args, *, validation, epochs):
    directory, cache = OUT / run_name, CACHE / run_name
    directory.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    settings = {"spec": spec.settings(), "seed": seed, "validation": validation,
                "epochs": epochs, "smoke": args.smoke, "device": args.device,
                "fit_rows": len(bundle["X_train"]), "query_rows": len(bundle["X_query"]),
                "preparation_sha256": sha256(CACHE / "prepared.joblib")}
    immutable(directory / "settings.json", settings)
    score_path, best_path, latest_path = directory / "result.json", cache / "model.pt", cache / "latest.pt"
    if score_path.exists():
        result = json.loads(score_path.read_text())
        if (sha256(best_path) != result["checkpoint_sha256"]
                or sha256(directory / "predictions.csv") != result["predictions_sha256"]
                or sha256(cache / "preprocessors.joblib") != result["preprocessors_sha256"]):
            raise ValueError("Completed run artifact changed")
        print("REUSE " + run_name, flush=True)
        return result
    capacity_gate(args)
    write_json(OUT / "worker_status.json", {"state": "training", "run": run_name,
                                             "pid": os.getpid(), "device": args.device})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable after capacity gate")
    seed_all(seed)
    started = time.perf_counter()
    pre = FeaturePreprocessor().fit(bundle["X_train"])
    seq_pre = SequencePreprocessor().fit(bundle["train_sequence"][:, -spec.window:])
    joblib.dump((pre, seq_pre), cache / "preprocessors.joblib")
    fit = [torch.from_numpy(a) for a in arrays(bundle, pre, seq_pre, spec, "train")]
    query = [torch.from_numpy(a) for a in arrays(bundle, pre, seq_pre, spec, "query")]
    target = bundle["y_train"]-bundle["train_baseline"]
    mean, sd = float(target.mean()), float(target.std())
    if not np.isfinite(target).all() or sd < 1e-6:
        raise ValueError("Invalid residual targets")
    ty = torch.from_numpy(((target-mean)/sd).astype(np.float32))
    model = RentalTransformer(fit[0].shape[1], pre.cardinalities, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    shuffle = torch.Generator().manual_seed(seed)
    start_epoch, best, best_epoch, stale, history, elapsed_before = 1, float("inf"), 0, 0, [], 0.
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_epoch, best, best_epoch = saved["epoch"]+1, saved["best"], saved["best_epoch"]
        stale, history, elapsed_before = saved["stale"], saved["history"], saved["seconds"]
        shuffle.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"])
        print(f"RESUME {run_name} epoch={start_epoch}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"TRAIN {run_name}: rows={len(ty)} parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    for epoch in range(start_epoch, epochs+1):
        if validation and stale >= PATIENCE:
            break
        model.train()
        total, seen = 0., 0
        for step, ids in enumerate(torch.randperm(len(ty), generator=shuffle).split(256)):
            if args.smoke and step == 2:
                break
            optimizer.zero_grad(set_to_none=True)
            x, y = [v[ids].to(device) for v in fit], ty[ids].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(*x)
                loss = (pred.float()-y).square().mean()
                if spec.processor != "none":
                    loss = loss + .01*model.gate.sigmoid().square()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            total += float(loss.detach())*len(ids)
            seen += len(ids)
        record = {"epoch": epoch, "standardized_loss": total/seen}
        if validation:
            pred = prediction(model, query, mean, sd, device)
            score = weighted_rmse(bundle["y_query"], pred+bundle["query_baseline"], bundle["query_weights"])
            record["weighted_rmse_sgd"] = score
            if score < best-1e-4:
                best, best_epoch, stale = score, epoch, 0
                temporary = best_path.with_suffix(".tmp.pt")
                torch.save({"model": model.state_dict(), "mean": mean, "sd": sd}, temporary)
                temporary.replace(best_path)
            else:
                stale += 1
        else:
            best_epoch = epoch
            temporary = best_path.with_suffix(".tmp.pt")
            torch.save({"model": model.state_dict(), "mean": mean, "sd": sd}, temporary)
            temporary.replace(best_path)
        history.append(record)
        write_json(directory / "learning_curve.json", history)
        write_json(OUT / "worker_status.json", {"state": "training", "run": run_name,
                                                 "pid": os.getpid(), "epoch": epoch,
                                                 "best_epoch": best_epoch})
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
                 "seconds": elapsed_before+time.perf_counter()-started,
                 "shuffle_rng": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}
        temporary = latest_path.with_suffix(".tmp.pt")
        torch.save(saved, temporary)
        temporary.replace(latest_path)
        del saved
        print(f"EPOCH {run_name} {epoch}: loss={record['standardized_loss']:.5f} "
              f"validation={record.get('weighted_rmse_sgd')} best_epoch={best_epoch}", flush=True)
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    pred = prediction(model, query, mean, sd, device)
    clone = RentalTransformer(fit[0].shape[1], pre.cardinalities, spec).to(device)
    clone.load_state_dict(payload["model"])
    replay = prediction(clone, query, mean, sd, device)
    np.testing.assert_array_equal(pred, replay)
    frame = pd.DataFrame({"row_id": bundle["X_query"].index, "month": bundle["query_month"],
                          "actual": bundle["y_query"], "predicted": pred+bundle["query_baseline"],
                          "weight": bundle["query_weights"],
                          "horizon": np.rint(bundle["query_context"][:, 0]*16).astype(int)})
    assert frame.row_id.is_unique and np.isfinite(frame.predicted).all()
    frame.to_csv(directory / "predictions.csv", index=False, float_format="%.10f")
    result = {"run": run_name, "variant": spec.name, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": len(history), "weighted_rmse_sgd": weighted_rmse(frame.actual, frame.predicted, frame.weight),
              **metrics(frame.actual, frame.predicted),
              "rows": len(frame), "parameters": sum(p.numel() for p in model.parameters()),
              "sequence_gate": float(model.gate.sigmoid().detach()) if spec.processor != "none" else 0.,
              "seconds": elapsed_before+time.perf_counter()-started,
              "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0.,
              "checkpoint_replay_max_difference": float(np.max(np.abs(pred-replay))),
              "checkpoint_sha256": sha256(best_path),
              "predictions_sha256": sha256(directory / "predictions.csv"),
              "preprocessors_sha256": sha256(cache / "preprocessors.joblib"),
              "outer_labels_used_for_selection": False}
    write_json(score_path, result)
    print("DONE " + json.dumps(result), flush=True)
    del model, optimizer, clone, payload, fit, query
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def slice_smoke(bundle):
    result = {}
    for k, v in bundle.items():
        n = 512 if k.startswith(("train_", "y_train", "X_train")) else 64
        ids = np.linspace(0, len(v)-1, min(n, len(v))).astype(int)
        result[k] = v.iloc[ids] if isinstance(v, pd.DataFrame) else v[ids]
    return result


def select_epochs(first, second):
    return max(1, int(np.floor((first["best_epoch"]+second["best_epoch"])/2+.5)))


def run_variant(spec, prepared, args):
    calibration = []
    for fold in ("early_2022", "early_2023"):
        bundle = prepared["folds"][f"{fold}_d{spec.delay}"]
        calibration.append(train(spec, bundle, f"calibration/{spec.name}/{fold}", 42, args,
                                 validation=True, epochs=MAX_EPOCHS))
    epochs = select_epochs(*calibration)
    immutable(OUT / "selection" / f"{spec.name}.json", {
        "variant": spec.settings(), "refit_epochs": epochs,
        "earlier_best_epochs": [r["best_epoch"] for r in calibration],
        "selection_label_max": "2024-04", "outer_labels_used": False})
    if args.stage == "calibration":
        return
    seeds = (42, 43, 44) if spec in PRIMARY else (42,)
    for seed in seeds:
        train(spec, prepared["folds"][f"outer_d{spec.delay}"], f"outer/{spec.name}/seed{seed}",
              seed, args, validation=False, epochs=epochs)
        summarize()
    if spec.name in ("T0", "T1", "T2"):
        train(spec, prepared["folds"][f"long16_d{spec.delay}"], f"long16/{spec.name}/seed42",
              42, args, validation=False, epochs=calibration[0]["best_epoch"])


def paired_interval(reference, candidate, repeats=2000):
    """Contiguous three-month blocks, retaining each month's rental population."""
    a = reference.set_index("row_id")
    b = candidate.set_index("row_id").reindex(a.index)
    if b.predicted.isna().any() or not np.array_equal(a.month, b.month):
        raise ValueError("Paired prediction alignment failed")
    np.testing.assert_allclose(a.actual, b.actual, atol=1e-8, rtol=0)
    stats = pd.DataFrame({"month": a.month, "a": (a.predicted-a.actual)**2*a.weight,
                          "b": (b.predicted-b.actual)**2*a.weight, "n": a.weight}).groupby("month").sum()
    n = len(stats)
    width = min(3, n)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, n-width+1, size=(repeats, int(np.ceil(n/width))))
    ids = (starts[:, :, None]+np.arange(width)).reshape(repeats, -1)[:, :n]
    values = stats.to_numpy()[ids].sum(1)
    delta = np.sqrt(values[:, 1]/values[:, 2])-np.sqrt(values[:, 0]/values[:, 2])
    return [float(v) for v in np.quantile(delta, [.025, .975])]


def summarize():
    results, monthly, horizons = [], [], []
    for path in sorted((OUT / "outer").glob("*/seed*/result.json")):
        result = json.loads(path.read_text())
        results.append(result)
        frame = pd.read_csv(path.parent / "predictions.csv")
        for month, part in frame.groupby("month"):
            monthly.append({"variant": result["variant"], "seed": result["seed"], "month": month,
                            "rows": len(part), **metrics(part.actual, part.predicted)})
        for lo, hi in ((1, 3), (4, 6), (7, 10), (11, 16)):
            part = frame[frame.horizon.between(lo, hi)]
            if len(part):
                horizons.append({"variant": result["variant"], "seed": result["seed"],
                                 "horizon": f"{lo}-{hi}", "rows": len(part), **metrics(part.actual, part.predicted)})
    if not results:
        return
    scores = pd.DataFrame(results)
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_scores.csv", index=False)
    pd.DataFrame(horizons).to_csv(OUT / "horizon_scores.csv", index=False)
    comparisons = []
    for ref, candidate in (("T0", "T1"), ("T1", "T2"), ("T2", "T3"),
                           ("T1", "C1"), ("T2", "C2"), ("T1", "C3"), ("T2", "C4"),
                           ("T1", "S1"), ("T2", "S2"), ("S3", "S4")):
        for seed in (42, 43, 44):
            pa, pb = [OUT / "outer" / v / f"seed{seed}" / "predictions.csv" for v in (ref, candidate)]
            if not pa.exists() or not pb.exists():
                continue
            a, b = pd.read_csv(pa), pd.read_csv(pb)
            delta = metrics(b.actual, b.predicted)["rmse_sgd"]-metrics(a.actual, a.predicted)["rmse_sgd"]
            low, high = paired_interval(a, b)
            comparisons.append({"reference": ref, "candidate": candidate, "seed": seed,
                                "delta_rmse_sgd": delta, "conditional_month_block_ci_low": low,
                                "conditional_month_block_ci_high": high})
    pd.DataFrame(comparisons).to_csv(OUT / "comparisons.csv", index=False)
    long_rows = []
    for path in sorted((OUT / "long16").glob("*/seed*/predictions.csv")):
        frame = pd.read_csv(path)
        for lo, hi in ((1, 3), (4, 6), (7, 10), (11, 16)):
            part = frame[frame.horizon.between(lo, hi)]
            if len(part):
                long_rows.append({"variant": path.parent.parent.name, "horizon": f"{lo}-{hi}",
                                  "rows": len(part), **metrics(part.actual, part.predicted)})
    pd.DataFrame(long_rows).to_csv(OUT / "long16_scores.csv", index=False)
    summary = scores[["variant", "seed", "rmse_sgd", "mae_sgd", "bias_sgd", "best_epoch", "peak_allocated_vram_gib"]]
    header = "| " + " | ".join(summary.columns) + " |"
    table_rows = [header, "| " + " | ".join(["---"]*len(summary.columns)) + " |"]
    table_rows.extend("| " + " | ".join(str(v) if not isinstance(v, float) else f"{v:.4f}" for v in row) + " |"
                      for row in summary.itertuples(index=False, name=None))
    rows = ["# Spatial and temporal transformers from scratch", "",
            "Runs below are completed; additional queued variants may still be pending.", "",
            "All models start from random weights. Original labels and source/submission files are preserved.",
            "Training episodes cover horizons 1–16. All rent-derived inputs share the row cutoff; macro inputs have an extra month buffer.",
            "The inherited baseline plateaus beyond month ten; the neural horizon input does not.", "",
            "## Individual-rental development results", "", "\n".join(table_rows), "",
            "Primary auxiliary comparison: T2 (rent + COE) versus T1 (rent only), matched by seed.",
            "T3 adds the supplied Singapore median stock-return proxy. Company/worksite interactions are excluded.", "",
            "## Limits", "",
            "June 2024–March 2025 and the overlapping 16-month stress window are repeatedly examined development periods, not untouched tests.",
            "There are only 51 distinct labeled months, with serial dependence. Row count does not multiply independent macro observations.",
            "Intervals are descriptive, conditional on fitted models, and omit selection/fitting uncertainty.",
            "All runs are fixed-origin. Monthly-updated forecasts are a different information setting and are not silently mixed here.",
            "Linear controls use a jointly optimized linear lag head with weight decay; they are not closed-form Ridge fits.", "",
            "See settings.json, availability.csv, training_horizons.csv, per-run checkpoints/audits, comparisons.csv and long16_scores.csv."]
    (OUT / "report.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


def main():
    global OUT, CACHE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "calibration", "primary", "controls", "sensitivity", "all", "summarize"), default="all")
    parser.add_argument("--variant", choices=EXPERIMENTS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=8.)
    parser.add_argument("--max-gpu-utilization", type=float, default=35.)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    OUT, CACHE = args.output.resolve(), args.cache.resolve()
    args.smoke = args.stage == "smoke"
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    with run_lock(), threadpool_limits(limits=4):
        if args.stage == "summarize":
            summarize()
            return
        prepared = prepare()
        if args.stage == "prepare":
            print("CPU PREPARATION COMPLETE; no GPU allocation requested", flush=True)
            return
        if args.smoke:
            for name in ([args.variant] if args.variant else ("T0", "T1", "T2", "T3", "C1", "C3")):
                spec = EXPERIMENTS[name]
                bundle = slice_smoke(prepared["folds"][f"early_2022_d{spec.delay}"])
                train(spec, bundle, f"smoke/{args.device}/{name}", 42, args, validation=True, epochs=1)
            return
        specs = ([EXPERIMENTS[args.variant]] if args.variant else
                 PRIMARY if args.stage in ("primary", "calibration") else
                 CONTROLS if args.stage == "controls" else
                 SENSITIVITIES if args.stage == "sensitivity" else PRIMARY+CONTROLS+SENSITIVITIES)
        for spec in specs:
            run_variant(spec, prepared, args)
        summarize()
        expected = json.loads((OUT / "settings.json").read_text())["hashes"]
        if fingerprint() != expected:
            raise ValueError("Source, submission or code changed during the experiment")
        write_json(OUT / "last_stage.json", {"stage": args.stage, "completed_variants": [s.name for s in specs],
                                            "submissions_changed": False})
        write_json(OUT / "worker_status.json", {"state": "completed", "stage": args.stage,
                                                 "pid": os.getpid(), "variants": [s.name for s in specs]})


if __name__ == "__main__":
    main()
