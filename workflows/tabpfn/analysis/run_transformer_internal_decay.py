"""Rank 23 follow-up: train T1 with an internal horizon-decaying sequence gate."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import (
    Experiment, FeaturePreprocessor, RentalTransformer, SequencePreprocessor,
)
from analysis.run_spatial_temporal_transformer import arrays, prediction, seed_all, slice_smoke


SOURCE = REPORTS / "spatial_temporal_transformer"
SOURCE_CACHE = ROOT / ".cache/spatial_temporal_transformer"
OUT = REPORTS / "transformer_ranked_experiments/23b_internal_decay"
CACHE = ROOT / ".cache/transformer_ranked_experiments/23b_internal_decay"
DESIGNS = ("decay3",)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


class InternalDecayTransformer(RentalTransformer):
    def forward(self, numeric, missing, categorical, spatial, sequence, context):
        tokens = numeric[:, :, None]*self.numeric_weight + self.numeric_bias + missing[:, :, None]*self.missing_weight
        cats = [module(categorical[:, i])[:, None] for i, module in enumerate(self.categories)]
        tokens = torch.cat([self.cls.expand(len(numeric), -1, -1), tokens,
                            *cats, self.spatial(spatial)[:, None]], dim=1)
        property_result = self.property_head(self.property_encoder(tokens)[:, 0]).squeeze(-1)
        monthly = self.sequence_input(sequence)+self.position
        monthly = torch.cat([monthly, self.context(context)[:, None]], dim=1)
        temporal = self.sequence_head(self.sequence_encoder(monthly)[:, -1]).squeeze(-1)
        horizon = context[:, 0]*16.
        decay = torch.exp(-(horizon-1.).clamp_min(0.)/3.)
        return property_result+self.gate.sigmoid()*decay*temporal


def gate(args):
    if args.device == "cuda":
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        wait_for_gpu(min_free_gib=args.min_free_gib, max_utilization=args.max_gpu_utilization,
                     stable_samples=2, poll_seconds=args.gpu_poll_seconds,
                     status_path=OUT / "gpu_status.json")


def fit(prepared, fold, design, seed, args, fixed_epochs=None):
    if fold in ("outer", "long16") and fixed_epochs is None:
        raise ValueError("Outer epochs must be selected earlier")
    stage = f"smoke_{args.device}" if args.smoke else "calibration" if fold.startswith("early") else fold
    label = f"{stage}/{design}/seed{seed}" if stage != "calibration" else f"{stage}/{design}/{fold}"
    report, saved = OUT / label, CACHE / label
    report.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    done, checkpoint = report / "result.json", saved / "model.pt"
    if done.exists():
        result = json.loads(done.read_text())
        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Model checkpoint changed")
        print(f"REUSE {label}", flush=True)
        return result
    bundle = prepared["folds"][f"{fold}_d0"]
    if args.smoke:
        bundle = slice_smoke(bundle)
    gate(args)
    device = torch.device(args.device)
    seed_all(seed)
    pre = FeaturePreprocessor().fit(bundle["X_train"])
    seq_pre = SequencePreprocessor().fit(bundle["train_sequence"])
    spec = Experiment("T1")
    fit_x = [torch.from_numpy(a) for a in arrays(bundle, pre, seq_pre, spec, "train")]
    query_x = [torch.from_numpy(a) for a in arrays(bundle, pre, seq_pre, spec, "query")]
    residual = bundle["y_train"]-bundle["train_baseline"]
    mean, sd = float(residual.mean()), float(residual.std())
    y = torch.from_numpy(((residual-mean)/sd).astype(np.float32))
    model = InternalDecayTransformer(fit_x[0].shape[1], pre.cardinalities, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    shuffle = torch.Generator().manual_seed(seed)
    best, best_epoch, stale = float("inf"), 0, 0
    total_epochs = 1 if args.smoke else 20 if fixed_epochs is None else fixed_epochs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, total_epochs+1):
        if fixed_epochs is None and stale >= 4:
            break
        model.train()
        loss_total, seen = 0., 0
        for step, ids in enumerate(torch.randperm(len(y), generator=shuffle).split(256)):
            if args.smoke and step == 2:
                break
            inputs = [a[ids].to(device) for a in fit_x]
            target = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(*inputs).float()
                loss = (output-target).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            loss_total += float(loss.detach())*len(ids)
            seen += len(ids)
        validation = None
        if fixed_epochs is None:
            pred = prediction(model, query_x, mean, sd, device)+bundle["query_baseline"]
            validation = float(np.sqrt(np.average((pred-bundle["y_query"])**2,
                                                  weights=bundle["query_weights"])))
            if validation < best-1e-4:
                best, best_epoch, stale = validation, epoch, 0
                temp = checkpoint.with_suffix(".tmp.pt")
                torch.save({"model": model.state_dict(), "mean": mean, "sd": sd}, temp)
                temp.replace(checkpoint)
            else:
                stale += 1
        else:
            best_epoch = epoch
            temp = checkpoint.with_suffix(".tmp.pt")
            torch.save({"model": model.state_dict(), "mean": mean, "sd": sd}, temp)
            temp.replace(checkpoint)
        print(f"EPOCH {label} {epoch}: loss={loss_total/seen:.5f} validation={validation}", flush=True)
        save_json(OUT / "worker_status.json", {"state": "training", "run": label, "epoch": epoch})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    pred = prediction(model, query_x, mean, sd, device)+bundle["query_baseline"]
    frame = pd.DataFrame({"row_id": bundle["X_query"].index, "month": bundle["query_month"],
                          "actual": bundle["y_query"], "predicted": pred,
                          "weight": bundle["query_weights"],
                          "horizon": np.rint(bundle["query_context"][:, 0]*16).astype(int)})
    frame.to_csv(report / "predictions.csv", index=False, float_format="%.10f")
    result = {"fold": fold, "design": design, "seed": seed,
              "best_epoch": best_epoch, "rows": len(frame),
              "weighted_rmse_sgd": float(np.sqrt(np.average((pred-frame.actual)**2,
                                                             weights=frame.weight))),
              **metrics(frame.actual, frame.predicted),
              "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0.,
              "checkpoint_sha256": sha256(checkpoint),
              "predictions_sha256": sha256(report / "predictions.csv"),
              "outer_labels_used_for_selection": False}
    save_json(done, result)
    print("DONE "+json.dumps(result), flush=True)
    del model, optimizer, fit_x, query_x
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
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Another internal-decay worker owns the lock") from None
    try:
        os.write(handle, str(os.getpid()).encode())
        os.close(handle)
        manifest = json.loads((SOURCE / "prepared.json").read_text())
        if sha256(SOURCE_CACHE / "prepared.joblib") != manifest["sha256"]:
            raise ValueError("Original prepared fold changed")
        before = {str(p): sha256(p) for p in (Path(__file__),
                 ROOT / "analysis/experiments/spatial_temporal_transformer.py",
                 SOURCE_CACHE / "prepared.joblib")}
        settings = {"sources_sha256": before, "designs": list(DESIGNS),
                    "internal_sequence_decay_months": 3., "seeds": [42,43,44],
                    "selection_folds": ["early_2022", "early_2023"]}
        path = OUT / "settings.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError("Settings/source changed")
        save_json(path, settings)
        torch.set_num_threads(4)
        prepared = joblib.load(SOURCE_CACHE / "prepared.joblib")
        if args.smoke:
            for design in DESIGNS:
                fit(prepared, "early_2022", design, 42, args)
            return
        rows = []
        for design in DESIGNS:
            first = fit(prepared, "early_2022", design, 42, args)
            second = fit(prepared, "early_2023", design, 42, args)
            for result in (first, second):
                baseline = json.loads((SOURCE / f"calibration/T0/{result['fold']}/result.json").read_text())
                rows.append({"design": design, "fold": result["fold"],
                             "delta_rmse_sgd": result["weighted_rmse_sgd"]-baseline["weighted_rmse_sgd"]})
        comparison = pd.DataFrame(rows)
        comparison.to_csv(OUT / "earlier_comparison.csv", index=False)
        mean = comparison.groupby("design").delta_rmse_sgd.mean()
        chosen = mean.idxmin() if mean.min() < 0 else "none"
        save_json(OUT / "selection.json", {"chosen": chosen, "earlier_mean_delta": mean.to_dict(),
                                            "outer_labels_used": False})
        if chosen != "none":
            first = json.loads((OUT / f"calibration/{chosen}/early_2022/result.json").read_text())
            second = json.loads((OUT / f"calibration/{chosen}/early_2023/result.json").read_text())
            epochs = max(1, int(np.floor((first["best_epoch"]+second["best_epoch"])/2+.5)))
            scores = [fit(prepared, "outer", chosen, seed, args, fixed_epochs=epochs)
                      for seed in (42,43,44)]
            fit(prepared, "long16", chosen, 42, args, fixed_epochs=first["best_epoch"])
            pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
        save_json(OUT / "worker_status.json", {"state": "completed", "chosen": chosen})
        if {str(p): sha256(Path(p)) for p in before} != before:
            raise ValueError("Original source changed")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
