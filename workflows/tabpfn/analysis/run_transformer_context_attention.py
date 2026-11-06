"""Rank 4: learned attention over causally available historical rental tokens."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.config import DATA, ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.rental_context_attention import ComparableAttention, retrieve


SOURCE = REPORTS / "spatial_temporal_transformer"
SOURCE_CACHE = ROOT / ".cache/spatial_temporal_transformer"
BASE_CACHE = ROOT / ".cache/transformer_ranked_experiments/01_conditioned_correction"
BASE_REPORT = REPORTS / "transformer_ranked_experiments/01_conditioned_correction"
OUT = REPORTS / "transformer_ranked_experiments/04_context_attention"
CACHE = ROOT / ".cache/transformer_ranked_experiments/04_context_attention"
FOLDS = ("early_2022", "early_2023", "outer")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def gate(args):
    if args.device == "cuda":
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        wait_for_gpu(min_free_gib=args.min_free_gib, max_utilization=args.max_gpu_utilization,
                     stable_samples=2, poll_seconds=args.gpu_poll_seconds,
                     status_path=OUT / "gpu_status.json")


def verified_base(fold, seed):
    tag = f"{fold}_seed{seed}"
    path, manifest = BASE_CACHE / "features" / f"{tag}.joblib", BASE_REPORT / "features" / f"{tag}.json"
    if sha256(path) != json.loads(manifest.read_text())["sha256"]:
        raise ValueError(f"Frozen T0 extracted representations changed: {tag}")
    return joblib.load(path)


def candidates(prepared, raw, fold, args):
    path, manifest = CACHE / "candidates" / f"{fold}.joblib", OUT / "candidates" / f"{fold}.json"
    if path.exists() and manifest.exists():
        if sha256(path) != json.loads(manifest.read_text())["sha256"]:
            raise ValueError("Candidate tokens changed")
        print(f"REUSE CANDIDATES {fold}", flush=True)
        return joblib.load(path)
    bundle = prepared["folds"][f"{fold}_d0"]
    if args.smoke:
        from analysis.run_spatial_temporal_transformer import slice_smoke
        bundle = slice_smoke(bundle)
    fitting = raw.loc[bundle["X_train"].index]
    bank = fitting
    result = {}
    for side in ("train", "query"):
        query = raw.loc[bundle[f"X_{side}"].index]
        origins = bundle[f"{side}_origins"]
        values, present = retrieve(bank, query, origins, bundle[f"{side}_baseline"],
                                   slots=32, search=128)
        result[f"{side}_tokens"], result[f"{side}_present"] = values, present
        print(f"RETRIEVAL {fold} {side}: rows={len(query)} coverage={(present.any(1)).mean():.3f}",
              flush=True)
    if not args.smoke:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, path, compress=3)
        write_json(manifest, {"sha256": sha256(path), "origin_filter": "candidate month <= row origin",
                              "candidate_bank": "fold fitting rows only", "slots": 32, "search": 128})
    return result


def predictions(model, features, candidates_data, side, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features[f"{side}_base"]), 1024):
            end = start + 1024
            x = [torch.as_tensor(v[start:end], device=device) for v in
                 (features[f"{side}_cls"], features[f"{side}_context"],
                  candidates_data[f"{side}_tokens"], candidates_data[f"{side}_present"])]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                value = model(*x)
            result.append(value.float().cpu().numpy())
    return np.concatenate(result)


def fit(prepared, raw, fold, seed, args, epochs=None):
    if fold == "outer" and epochs is None:
        raise ValueError("Earlier-selected epochs required")
    phase = f"smoke_{args.device}" if args.smoke else "calibration" if fold != "outer" else "outer"
    label = f"{phase}/{fold}/seed{seed}"
    report, saved = OUT / label, CACHE / label
    report.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    done, checkpoint, latest = report / "result.json", saved / "head.pt", saved / "latest.pt"
    if done.exists():
        result = json.loads(done.read_text())
        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Saved context head changed")
        print(f"REUSE {label}", flush=True)
        return result
    source = verified_base(fold, seed)
    context = candidates(prepared, raw, fold, args)
    if args.smoke:
        source = {key: (value[:512] if key.startswith("train_") or key == "y_train" else
                        value[:64] if key.startswith("query_") else value)
                  for key, value in source.items()}
    if len(source["y_train"]) != len(context["train_tokens"]):
        raise ValueError("Context and T0 features misaligned")
    gate(args)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = ComparableAttention().to(device)
    initially = predictions(model, source, context, "query", device)
    if np.max(np.abs(initially)) > 1e-8:
        raise AssertionError("Initial context head must reproduce T0 exactly")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    fit_x = [torch.as_tensor(v) for v in (source["train_cls"], source["train_context"],
                                          context["train_tokens"], context["train_present"])]
    y, baseline = torch.as_tensor(source["y_train"]), torch.as_tensor(source["train_base"])
    shuffle = torch.Generator().manual_seed(seed)
    first_epoch, best, best_epoch, stale = 1, float("inf"), 0, 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(device)
        first_epoch = state["epoch"] + 1
        best, best_epoch, stale = state["best"], state["best_epoch"], state["stale"]
        shuffle.set_state(state["shuffle"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    final_epoch = 1 if args.smoke else (20 if fold != "outer" else epochs)
    last_epoch = first_epoch - 1
    for epoch in range(first_epoch, final_epoch+1):
        if fold != "outer" and stale >= 4:
            break
        last_epoch = epoch
        model.train()
        total, count = 0., 0
        for step, ids in enumerate(torch.randperm(len(y), generator=shuffle).split(512)):
            if args.smoke and step == 2:
                break
            x = [v[ids].to(device) for v in fit_x]
            target, base = y[ids].to(device), baseline[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                corr = model(*x).float()
                loss = (base.float()+corr-target).square().mean()+.01*corr.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            total += float(loss.detach())*len(ids)
            count += len(ids)
        score = None
        if fold != "outer":
            corr = predictions(model, source, context, "query", device)
            pred = (source["query_baseline"]+source["target_mean"]+
                    source["target_sd"]*(source["query_base"]+corr))
            score = float(np.sqrt(np.average((pred-source["query_actual"])**2,
                                             weights=source["query_weight"])))
            if score < best-1e-4:
                best, best_epoch, stale = score, epoch, 0
                temporary = checkpoint.with_suffix(".tmp.pt")
                torch.save(model.state_dict(), temporary)
                temporary.replace(checkpoint)
            else:
                stale += 1
        else:
            best_epoch = epoch
            temporary = checkpoint.with_suffix(".tmp.pt")
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale,
                 "shuffle": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}
        temporary = latest.with_suffix(".tmp.pt")
        torch.save(state, temporary)
        temporary.replace(latest)
        print(f"EPOCH {label} {epoch}: loss={total/count:.5f} validation={score}", flush=True)
        write_json(OUT / "worker_status.json", {"state": "training", "run": label, "epoch": epoch})
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    corr = predictions(model, source, context, "query", device)
    base_sgd = (source["query_baseline"]+source["target_mean"]+
                source["target_sd"]*source["query_base"])
    pred = base_sgd + source["target_sd"]*corr
    frame = pd.DataFrame({"row_id": source["query_id"], "month": source["query_month"],
                          "actual": source["query_actual"], "predicted": pred,
                          "base_prediction": base_sgd, "correction_sgd": source["target_sd"]*corr,
                          "weight": source["query_weight"], "horizon": source["query_horizon"],
                          "comparable_count": context["query_present"].sum(1)})
    frame.to_csv(report / "predictions.csv", index=False, float_format="%.10f")
    result = {"fold": fold, "seed": seed, "best_epoch": best_epoch, "epochs_run": last_epoch,
              "rows": len(frame), "parameters": sum(p.numel() for p in model.parameters()),
              "weighted_rmse_sgd": float(np.sqrt(np.average((frame.actual-pred)**2,
                                                             weights=frame.weight))),
              **metrics(frame.actual, frame.predicted),
              "base_rmse_sgd": metrics(frame.actual, frame.base_prediction)["rmse_sgd"],
              "mean_correction_sgd": float(frame.correction_sgd.mean()),
              "checkpoint_sha256": sha256(checkpoint),
              "predictions_sha256": sha256(report / "predictions.csv"),
              "outer_labels_used_for_selection": False}
    write_json(done, result)
    print("DONE "+json.dumps(result), flush=True)
    del source, context, model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
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
    lock = OUT / "worker.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Worker lock exists: {lock}") from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        manifest = json.loads((SOURCE / "prepared.json").read_text())
        if sha256(SOURCE_CACHE / "prepared.joblib") != manifest["sha256"]:
            raise ValueError("Original prepared fold changed")
        original = {str(p): sha256(p) for p in (DATA / "train.csv", SOURCE / "prepared.json",
                   ROOT / "analysis/experiments/rental_context_attention.py", Path(__file__))}
        settings = {"sources_sha256": original, "slots": 32, "nearest_search": 128,
                    "distance_cap_km": 5., "matched_T0": True, "seed": [42,43,44],
                    "selection_folds": list(FOLDS[:2])}
        settings_path = OUT / "settings.json"
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError("Settings changed")
        write_json(settings_path, settings)
        from analysis.experiments.spatiotemporal_hybrid import spatial_frame
        raw = spatial_frame(pd.read_csv(DATA / "train.csv"))
        prepared = joblib.load(SOURCE_CACHE / "prepared.joblib")
        torch.set_num_threads(4)
        if args.smoke:
            fit(prepared, raw, "early_2022", 42, args)
        else:
            earlier = [fit(prepared, raw, fold, 42, args) for fold in FOLDS[:2]]
            epochs = max(1, int(np.floor((earlier[0]["best_epoch"]+earlier[1]["best_epoch"])/2+.5)))
            write_json(OUT / "selection.json", {"earlier_best_epochs": [r["best_epoch"] for r in earlier],
                                                 "outer_epochs": epochs, "selection_label_max": "2024-04",
                                                 "outer_labels_used": False})
            scores = [fit(prepared, raw, "outer", seed, args, epochs=epochs) for seed in (42,43,44)]
            pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
        if {str(p): sha256(Path(p)) for p in original} != original:
            raise ValueError("Original source changed")
        write_json(OUT / "worker_status.json", {"state": "completed", "stage": args.stage})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
