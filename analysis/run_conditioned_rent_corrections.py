"""Sequential, resumable frozen-T0 correction ablation with chronological selection."""
from __future__ import annotations

import argparse
import gc
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
import torch

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.conditioned_rent_correction import RentCorrection
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import (
    Experiment, FeaturePreprocessor, RentalTransformer, SequencePreprocessor,
)


SOURCE = REPORTS / "spatial_temporal_transformer"
SOURCE_CACHE = ROOT / ".cache/spatial_temporal_transformer"
OUT = REPORTS / "transformer_ranked_experiments" / "01_conditioned_correction"
CACHE = ROOT / ".cache/transformer_ranked_experiments/01_conditioned_correction"
DESIGNS = {
    "independent_old": (False, False),
    "conditioned_old": (True, False),
    "independent_stable": (False, True),
    "conditioned_stable": (True, True),
}
FOLDS = ("early_2022", "early_2023", "outer")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def verified_sources():
    base = SOURCE_CACHE / "prepared.joblib"
    manifest = json.loads((SOURCE / "prepared.json").read_text())
    if sha256(base) != manifest["sha256"]:
        raise ValueError("Original prepared fold checksum changed")
    paths = [base, Path(__file__), ROOT / "analysis/experiments/conditioned_rent_correction.py",
             ROOT / "analysis/experiments/spatial_temporal_transformer.py",
             SOURCE / "prepared.json"]
    for fold in ("calibration/T0/early_2022", "calibration/T0/early_2023",
                 "outer/T0/seed42", "outer/T0/seed43", "outer/T0/seed44"):
        report, saved = SOURCE / fold, SOURCE_CACHE / fold
        result = json.loads((report / "result.json").read_text())
        for file, key in (("model.pt", "checkpoint_sha256"),
                          ("preprocessors.joblib", "preprocessors_sha256")):
            if sha256(saved / file) != result[key]:
                raise ValueError(f"Original T0 artifact changed: {saved / file}")
            paths.append(saved / file)
        paths.append(report / "predictions.csv")
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def gate(args):
    if args.device == "cuda":
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        wait_for_gpu(min_free_gib=args.min_free_gib,
                     max_utilization=args.max_gpu_utilization,
                     stable_samples=2, poll_seconds=args.gpu_poll_seconds,
                     status_path=OUT / "gpu_status.json")


@torch.inference_mode()
def extract_side(model, values, device):
    n = len(values[0])
    representations, predictions = [], []
    captured = []
    hook = model.property_encoder.register_forward_hook(
        lambda _module, _input, output: captured.append(output[:, 0].float().cpu().numpy()))
    try:
        for start in range(0, n, 512):
            x = [torch.as_tensor(a[start:start+512], device=device) for a in values]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predicted = model(*x)
            predictions.append(predicted.float().cpu().numpy())
            representations.append(captured.pop())
    finally:
        hook.remove()
    return np.concatenate(representations), np.concatenate(predictions)


def base_data(prepared, fold, seed, args):
    tag = f"{'smoke_' + args.device + '_' if args.smoke else ''}{fold}_seed{seed}"
    path = CACHE / "features" / f"{tag}.joblib"
    meta = OUT / "features" / f"{tag}.json"
    if path.exists() and meta.exists():
        expected = json.loads(meta.read_text())
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Changed extracted features: {path}")
        print(f"REUSE FEATURES {tag}", flush=True)
        return joblib.load(path)
    gate(args)
    bundle = prepared["folds"][f"{fold}_d0"]
    if args.smoke:
        from analysis.run_spatial_temporal_transformer import slice_smoke
        bundle = slice_smoke(bundle)
    base_tag = f"calibration/T0/{fold}" if fold != "outer" else f"outer/T0/seed{seed}"
    source_result = json.loads((SOURCE / base_tag / "result.json").read_text())
    pre, seq_pre = joblib.load(SOURCE_CACHE / base_tag / "preprocessors.joblib")
    payload = torch.load(SOURCE_CACHE / base_tag / "model.pt", map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    model = RentalTransformer(len(pre.numerical), pre.cardinalities,
                              Experiment("T0", "none", "none")).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    values = {}
    for side in ("train", "query"):
        prop = pre.transform(bundle[f"X_{side}"])
        tensors = (*prop, np.zeros((len(prop[0]), 12, 34), np.float32),
                   bundle[f"{side}_context"])
        cls, base = extract_side(model, tensors, device)
        sequence = seq_pre.transform(bundle[f"{side}_sequence"][:, -12:], "rent")
        values[f"{side}_cls"] = cls
        values[f"{side}_base"] = base
        values[f"{side}_sequence"] = sequence
        values[f"{side}_context"] = bundle[f"{side}_context"]
    values["y_train"] = ((bundle["y_train"] - bundle["train_baseline"] -
                          payload["mean"]) / payload["sd"]).astype(np.float32)
    values["query_actual"] = bundle["y_query"]
    values["query_baseline"] = bundle["query_baseline"]
    values["query_weight"] = bundle["query_weights"]
    values["query_month"] = bundle["query_month"]
    values["query_id"] = bundle["X_query"].index.to_numpy()
    values["query_horizon"] = np.rint(bundle["query_context"][:, 0] * 16).astype(int)
    values["target_mean"] = float(payload["mean"])
    values["target_sd"] = float(payload["sd"])
    values["base_source_sha256"] = source_result["checkpoint_sha256"]
    if not args.smoke:
        original = pd.read_csv(SOURCE / base_tag / "predictions.csv")
        if not np.array_equal(values["query_id"], original.row_id.to_numpy()):
            raise ValueError("T0 query IDs changed")
        restored = values["query_baseline"] + payload["mean"] + payload["sd"] * values["query_base"]
        np.testing.assert_allclose(restored, original.predicted, atol=2e-4, rtol=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(values, path, compress=3)
    write_json(meta, {"sha256": sha256(path), "source_checkpoint_sha256": source_result["checkpoint_sha256"],
                      "rows_train": len(values["y_train"]), "rows_query": len(values["query_actual"]),
                      "base_replay": True, "smoke": args.smoke})
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"FEATURES READY {tag}", flush=True)
    return values


def infer(model, data, side, device):
    model.eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(data[f"{side}_base"]), 1024):
            x = [torch.as_tensor(data[f"{side}_{key}"][start:start+1024], device=device)
                 for key in ("cls", "sequence", "context")]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                correction = model(*x)
            pieces.append(correction.float().cpu().numpy())
    return np.concatenate(pieces)


def fit(prepared, fold, seed, design, args, epochs=None):
    if fold == "outer" and epochs is None:
        raise ValueError("Outer epochs must come from earlier folds")
    phase = f"smoke_{args.device}" if args.smoke else "calibration" if fold != "outer" else "outer"
    tag = f"{phase}/{design}/{fold}/seed{seed}"
    report, saved = OUT / tag, CACHE / tag
    report.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    done, best_path, latest_path = report / "result.json", saved / "head.pt", saved / "latest.pt"
    if done.exists():
        result = json.loads(done.read_text())
        if sha256(best_path) != result["checkpoint_sha256"]:
            raise ValueError(f"Changed saved head: {best_path}")
        print(f"REUSE {tag}", flush=True)
        return result
    data = base_data(prepared, fold, seed, args)
    gate(args)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = RentCorrection(conditioned=DESIGNS[design][0], stabilized=DESIGNS[design][1]).to(device)
    if DESIGNS[design][1]:
        first = infer(model, data, "query", device)
        if np.max(np.abs(first)) > 1e-8:
            raise AssertionError("Zero-initialized correction is not an exact T0 fallback")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    train_x = [torch.as_tensor(data[f"train_{key}"]) for key in ("cls", "sequence", "context")]
    train_y = torch.as_tensor(data["y_train"])
    best, best_epoch, stale, first_epoch = float("inf"), 0, 0, 1
    shuffle = torch.Generator().manual_seed(seed)
    if latest_path.exists():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.to(device)
        best, best_epoch, stale, first_epoch = state["best"], state["best_epoch"], state["stale"], state["epoch"]+1
        shuffle.set_state(state["shuffle"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
    validation = fold != "outer"
    final_epoch = 1 if args.smoke else 20 if validation else epochs
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(first_epoch, final_epoch+1):
        if validation and stale >= 4:
            break
        model.train()
        losses = []
        for step, ids in enumerate(torch.randperm(len(train_y), generator=shuffle).split(512)):
            if args.smoke and step == 2:
                break
            x = [a[ids].to(device) for a in train_x]
            y = train_y[ids].to(device)
            base = torch.as_tensor(data["train_base"][ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                correction = model(*x)
                loss = (base.float() + correction.float() - y).square().mean() + model.regularizer(correction.float())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "training_loss": float(np.mean(losses))}
        if validation:
            corr = infer(model, data, "query", device)
            prediction = (data["query_baseline"] + data["target_mean"] +
                          data["target_sd"] * (data["query_base"] + corr))
            score = float(np.sqrt(np.average((prediction-data["query_actual"])**2,
                                             weights=data["query_weight"])))
            row["weighted_rmse_sgd"] = score
            if score < best-1e-4:
                best, best_epoch, stale = score, epoch, 0
                temp = best_path.with_suffix(".tmp.pt")
                torch.save(model.state_dict(), temp)
                temp.replace(best_path)
            else:
                stale += 1
        else:
            best_epoch = epoch
            temp = best_path.with_suffix(".tmp.pt")
            torch.save(model.state_dict(), temp)
            temp.replace(best_path)
        history.append(row)
        write_json(report / "learning_curve.json", history)
        write_json(OUT / "worker_status.json", {"state": "training", "run": tag, "epoch": epoch,
                                                 "pid": os.getpid()})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best": best, "best_epoch": best_epoch, "stale": stale,
                 "shuffle": shuffle.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}
        temp = latest_path.with_suffix(".tmp.pt")
        torch.save(state, temp)
        temp.replace(latest_path)
        print(f"EPOCH {tag} {epoch}: train={row['training_loss']:.5f} validation={row.get('weighted_rmse_sgd')}", flush=True)
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    correction = infer(model, data, "query", device)
    output = (data["query_baseline"] + data["target_mean"] +
              data["target_sd"] * (data["query_base"] + correction))
    frame = pd.DataFrame({"row_id": data["query_id"], "month": data["query_month"],
                          "actual": data["query_actual"], "predicted": output,
                          "base_prediction": data["query_baseline"]+data["target_mean"]+
                                             data["target_sd"]*data["query_base"],
                          "correction_sgd": data["target_sd"]*correction,
                          "weight": data["query_weight"], "horizon": data["query_horizon"]})
    frame.to_csv(report / "predictions.csv", index=False, float_format="%.10f")
    result = {"design": design, "fold": fold, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": epoch, "rows": len(frame), "parameters": sum(p.numel() for p in model.parameters()),
              "weighted_rmse_sgd": float(np.sqrt(np.average((output-frame.actual)**2, weights=frame.weight))),
              **metrics(frame.actual, frame.predicted),
              "mean_correction_sgd": float(frame.correction_sgd.mean()),
              "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0.,
              "seconds": time.perf_counter()-started, "checkpoint_sha256": sha256(best_path),
              "predictions_sha256": sha256(report / "predictions.csv"),
              "base_checkpoint_sha256": data["base_source_sha256"],
              "outer_labels_used_for_selection": False}
    write_json(done, result)
    print("DONE " + json.dumps(result), flush=True)
    del model, optimizer, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize():
    rows = [json.loads(p.read_text()) for p in (OUT / "outer").glob("*/*/seed*/result.json")]
    if not rows:
        return
    pd.DataFrame(rows).sort_values(["design", "seed"]).to_csv(OUT / "scores.csv", index=False)
    comparisons = []
    for row in rows:
        path = OUT / "outer" / row["design"] / "outer" / f"seed{row['seed']}" / "predictions.csv"
        frame = pd.read_csv(path)
        delta = metrics(frame.actual, frame.predicted)["rmse_sgd"] - metrics(frame.actual, frame.base_prediction)["rmse_sgd"]
        comparisons.append({"design": row["design"], "seed": row["seed"],
                            "delta_rmse_vs_frozen_T0_sgd": delta, "mean_correction_sgd": row["mean_correction_sgd"]})
    pd.DataFrame(comparisons).sort_values(["design", "seed"]).to_csv(OUT / "comparisons.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "all", "summarize"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=8.)
    parser.add_argument("--max-gpu-utilization", type=float, default=35.)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.)
    args = parser.parse_args()
    args.smoke = args.stage == "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        summarize()
        return
    lock = OUT / "worker.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Worker lock exists: {lock}") from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        before = verified_sources()
        settings_path = OUT / "settings.json"
        settings = {"source_hashes": before, "designs": {k: list(v) for k, v in DESIGNS.items()}, "selection_folds": list(FOLDS[:2]),
                    "outer_fold": "outer", "seeds": [42,43,44], "max_calibration_epochs": 20,
                    "patience": 4, "learning_rate": .0003, "batch": 512, "correction_penalty": .01,
                    "frozen_backbone": True, "original_submissions_modified": False}
        if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
            raise ValueError("Settings/source changed: use a new output directory")
        write_json(settings_path, settings)
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("high")
        prepared = joblib.load(SOURCE_CACHE / "prepared.joblib")
        if args.smoke:
            for design in DESIGNS:
                fit(prepared, "early_2022", 42, design, args)
        else:
            for design in DESIGNS:
                earlier = [fit(prepared, fold, 42, design, args) for fold in FOLDS[:2]]
                epochs = max(1, int(np.floor((earlier[0]["best_epoch"]+earlier[1]["best_epoch"])/2+.5)))
                write_json(OUT / "selection" / f"{design}.json",
                           {"design": design, "earlier_best_epochs": [r["best_epoch"] for r in earlier],
                            "outer_epochs": epochs, "outer_labels_used": False})
                for seed in (42, 43, 44):
                    fit(prepared, "outer", seed, design, args, epochs=epochs)
                summarize()
        if verified_sources() != before:
            raise ValueError("Original experiment sources changed during this run")
        write_json(OUT / "worker_status.json", {"state": "completed", "stage": args.stage,
                                                 "pid": os.getpid(), "completed_designs": list(DESIGNS)})
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
