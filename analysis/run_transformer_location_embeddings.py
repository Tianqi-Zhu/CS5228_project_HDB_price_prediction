"""Rank 10: frozen-T0 hierarchical town/area/block embedding correction."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.run_conditioned_rent_corrections import base_data, verified_sources
from analysis.run_spatial_temporal_transformer import slice_smoke

SOURCE = REPORTS / "spatial_temporal_transformer"
OUT = REPORTS / "transformer_ranked_experiments/10_location_embeddings"
CACHE = ROOT / ".cache/transformer_ranked_experiments/10_location_embeddings"
LOCATION = ("TOWN_CLEAN", "PLANNING_AREA", "BLOCK_ID")


class LocationHead(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.location = nn.ModuleList([nn.Embedding(size, 16, padding_idx=0) for size in sizes])
        self.property = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 16), nn.Tanh())
        self.context = nn.Linear(4, 16)
        self.output = nn.Linear(16*4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, cls, context, ids):
        local = [table(ids[:, i]) for i, table in enumerate(self.location)]
        query = self.property(cls)+self.context(context)
        pieces = [query*local[0], query*local[1], query*local[2],
                  local[0]+local[1]+local[2]]
        return self.output(torch.cat(pieces, dim=1)).squeeze(-1)


def encode_locations(train, query):
    matrices = []
    sizes = []
    for column in LOCATION:
        series = train[column].astype("string")
        vocab = {value: index+1 for index, value in enumerate(series.dropna().unique())}
        matrices.append((series.map(vocab).fillna(0).to_numpy(np.int64),
                         query[column].astype("string").map(vocab).fillna(0).to_numpy(np.int64)))
        sizes.append(len(vocab)+1)
    ids_train = np.column_stack([pair[0] for pair in matrices])
    ids_query = np.column_stack([pair[1] for pair in matrices])
    return ids_train, ids_query, sizes


@torch.inference_mode()
def infer(model, data, ids, side, device):
    model.eval()
    pieces = []
    for start in range(0, len(ids), 2048):
        sl = slice(start, start+2048)
        inputs = [torch.as_tensor(data[f"{side}_{key}"][sl], device=device)
                  for key in ("cls", "context")]
        batch_ids = torch.as_tensor(ids[sl], device=device)
        pieces.append(model(*inputs, batch_ids).float().cpu().numpy())
    return np.concatenate(pieces)


def fit(prepared, fold, seed, args, epochs=None):
    if fold == "outer" and epochs is None:
        raise ValueError("Outer epochs must be set from earlier folds")
    tag = f"{'smoke_' + args.device + '_' if args.smoke else ''}{fold}_seed{seed}"
    report, saved = OUT / tag, CACHE / tag
    report.mkdir(parents=True, exist_ok=True)
    saved.mkdir(parents=True, exist_ok=True)
    done = report / "result.json"
    checkpoint = saved / "head.pt"
    if done.exists():
        result = json.loads(done.read_text())
        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("Changed location checkpoint")
        print("REUSE", tag, flush=True)
        return result
    data = base_data(prepared, fold, seed, args)
    bundle = prepared["folds"][f"{fold}_d0"]
    if args.smoke:
        bundle = slice_smoke(bundle)
    if args.device == "cuda":
        gc.collect()
        wait_for_gpu(min_free_gib=args.min_free_gib,
                     max_utilization=args.max_gpu_utilization,
                     stable_samples=2, poll_seconds=args.gpu_poll_seconds,
                     status_path=OUT / "gpu_status.json")
    train_ids, query_ids, sizes = encode_locations(bundle["X_train"], bundle["X_query"])
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = LocationHead(sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    cls = torch.as_tensor(data["train_cls"])
    context = torch.as_tensor(data["train_context"])
    train_ids = torch.as_tensor(train_ids)
    target = torch.as_tensor(data["y_train"])
    base = torch.as_tensor(data["train_base"])
    rng = torch.Generator().manual_seed(seed)
    best, best_epoch, stale = float("inf"), 0, 0
    max_epochs = 1 if args.smoke else 12 if epochs is None else epochs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, max_epochs+1):
        if epochs is None and stale >= 3:
            break
        model.train()
        losses = []
        for step, ix in enumerate(torch.randperm(len(target), generator=rng).split(512)):
            if args.smoke and step >= 2:
                break
            x, t, loc = cls[ix].to(device), context[ix].to(device), train_ids[ix].to(device)
            y, b = target[ix].to(device), base[ix].to(device)
            optimizer.zero_grad(set_to_none=True)
            corr = model(x, t, loc)
            loss = ((b+corr-y)**2).mean() + .01*corr.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite location loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        score = None
        if epochs is None:
            corr = infer(model, data, query_ids, "query", device)
            pred = data["query_baseline"]+data["target_mean"]+data["target_sd"]*(data["query_base"]+corr)
            score = float(np.sqrt(np.average((pred-data["query_actual"])**2,
                                              weights=data["query_weight"])))
            if score < best-1e-4:
                best, best_epoch, stale = score, epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
        else:
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        print(f"EPOCH {tag} {epoch}: train={np.mean(losses):.5f} validation={score}", flush=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    corr = infer(model, data, query_ids, "query", device)
    pred = data["query_baseline"]+data["target_mean"]+data["target_sd"]*(data["query_base"]+corr)
    baseline = data["query_baseline"]+data["target_mean"]+data["target_sd"]*data["query_base"]
    frame = pd.DataFrame({"row_id": data["query_id"], "month": data["query_month"],
        "actual": data["query_actual"], "predicted": pred, "base_prediction": baseline,
        "weight": data["query_weight"], "horizon": data["query_horizon"]})
    frame.to_csv(report / "predictions.csv", index=False, float_format="%.10f")
    result = {"fold": fold, "seed": seed, "best_epoch": best_epoch,
        "rows": len(frame), "location_vocabularies": sizes,
        "weighted_rmse_sgd": float(np.sqrt(np.average((frame.predicted-frame.actual)**2,
                                                       weights=frame.weight))),
        "base_rmse_sgd": float(np.sqrt(np.average((frame.base_prediction-frame.actual)**2,
                                                 weights=frame.weight))),
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0.,
        "checkpoint_sha256": sha256(checkpoint),
        "predictions_sha256": sha256(report / "predictions.csv"),
        "outer_labels_used_for_selection": False}
    done.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("DONE", json.dumps(result), flush=True)
    del data, model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=8.)
    parser.add_argument("--max-gpu-utilization", type=float, default=35.)
    parser.add_argument("--gpu-poll-seconds", type=float, default=15.)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    before = verified_sources()
    prepared = joblib.load(ROOT / ".cache/spatial_temporal_transformer/prepared.joblib")
    if args.smoke:
        fit(prepared, "early_2022", 42, args)
        return
    earlier = [fit(prepared, fold, 42, args) for fold in ("early_2022", "early_2023")]
    chosen_epochs = max(1, int(np.floor(np.mean([r["best_epoch"] for r in earlier])+.5)))
    (OUT / "selection.json").write_text(json.dumps({"outer_epochs": chosen_epochs,
        "earlier_results": earlier, "outer_labels_used": False}, indent=2), encoding="utf-8")
    outer = [fit(prepared, "outer", seed, args, epochs=chosen_epochs) for seed in (42,43,44)]
    pd.DataFrame(outer).to_csv(OUT / "scores.csv", index=False)
    if verified_sources() != before:
        raise ValueError("Original source changed")
    (OUT / "verification.json").write_text(json.dumps({"source_hashes": before,
        "frozen_backbone": True, "outer_labels_used_for_selection": False}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
