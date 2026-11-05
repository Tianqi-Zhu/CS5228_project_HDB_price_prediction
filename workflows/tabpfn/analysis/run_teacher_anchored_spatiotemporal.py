"""Chronological TabPFN-anchored spatial/temporal correction experiment.

Uses existing out-of-origin ABCDE forecasts. No teacher prediction may have the
query's label in its context. Historical origin 1 fits, origin 2 selects, and
the combined historical rows fit a locked model for the retrospective outer run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.gpu_capacity import wait_for_gpu


SOURCE = REPORTS / "tabpfn_abcde_ensemble/predictions_all_origins.csv"
PREPARED = ROOT / ".cache/spatial_temporal_transformer/prepared.joblib"
OUT = REPORTS / "teacher_anchored_spatiotemporal"
ORIGINS = {"early_2022": "2022-09", "early_2023": "2023-07", "outer": "2024-06"}
NUMERIC = [
    "FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "MAX_FLOOR", "LATITUDE", "LONGITUDE",
    "DISTANCE_TO_CBD_KM", "HIST_BLOCK_FLAT_COUNT", "HIST_AREA_FLAT_COUNT",
    "RECENT_BLOCK_FLAT_6M_COUNT", "RECENT_AREA_FLAT_6M_COUNT",
    "AREA_RECENT_MINUS_LONG", "BLOCK_RECENT_MINUS_LONG",
    "BLOCK_MINUS_AREA_RECENT", "TEMP_LOCAL_WEIGHT", "TEMP_FORECAST_RMSE",
    "TEMP_FORECAST_HORIZON",
]
LOG_COUNTS = {"HIST_BLOCK_FLAT_COUNT", "HIST_AREA_FLAT_COUNT",
              "RECENT_BLOCK_FLAT_6M_COUNT", "RECENT_AREA_FLAT_6M_COUNT"}
RENT_CHANNELS = 12
SEEDS = [42, 43, 44]
REGULARIZATIONS = [0.25, 1.0, 4.0]
MAX_EPOCHS = 35


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_frames():
    manifest = json.loads((REPORTS / "spatial_temporal_transformer/prepared.json").read_text())
    if sha256(PREPARED) != manifest["sha256"]:
        raise ValueError("Transformer feature bundle changed")
    teacher = pd.read_csv(SOURCE, float_precision="round_trip")
    prepared = joblib.load(PREPARED)
    result = {}
    for key, origin in ORIGINS.items():
        b = prepared["folds"][f"{key}_d0"]
        t = teacher[teacher.origin.eq(origin)].set_index("row_id").reindex(b["X_query"].index)
        if t.ABCDE_equal.isna().any() or len(t) != len(b["X_query"]):
            raise ValueError(f"Teacher rows missing for {key}")
        np.testing.assert_allclose(t.actual.to_numpy(float), b["y_query"], atol=1e-8)
        np.testing.assert_array_equal(t.month.to_numpy(), b["query_month"])
        assert set(b["X_train"].index).isdisjoint(b["X_query"].index)
        assert pd.PeriodIndex(b["train_month"], freq="M").max() < pd.Period(origin, "M")
        assert (pd.PeriodIndex(b["query_month"], freq="M") >= pd.Period(origin, "M")).all()
        assert (pd.PeriodIndex(b["query_origins"], freq="M") == pd.Period(origin, "M") - 1).all()
        result[key] = (b, t)
    return result


def raw_inputs(bundle, teacher):
    x = bundle["X_query"]
    n = x[NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    for j, name in enumerate(NUMERIC):
        if name in LOG_COUNTS:
            n[:, j] = np.log1p(np.maximum(n[:, j], 0))
    # Teacher deviation from the same baseline is available without labels.
    n = np.column_stack([n, (teacher.ABCDE_equal.to_numpy(float) -
                            bundle["query_baseline"]) / 1000.])
    lat = pd.to_numeric(x.LATITUDE, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(x.LONGITUDE, errors="coerce").to_numpy(float)
    east = (lon - 103.85) * 111.2 * math.cos(math.radians(1.35))
    north = (lat - 1.35) * 111.2
    coords = np.stack([east, north], axis=1)
    angles = 2 * np.pi * coords[:, :, None] / np.array([0.5, 2., 8.])
    spatial = np.column_stack([coords / 20., np.sin(angles).reshape(len(x), -1),
                               np.cos(angles).reshape(len(x), -1)])
    types = pd.get_dummies(x.FLAT_TYPE_CLEAN.astype(str)).reindex(
        columns=["1-room", "2-room", "3-room", "4-room", "5-room", "executive"],
        fill_value=0).to_numpy(float)
    fixed = np.concatenate([spatial, types, bundle["query_context"]], axis=1)
    sequence = bundle["query_sequence"][:, :, :RENT_CHANNELS].astype(float)
    return n, fixed, sequence


class Preprocess:
    def fit(self, frames):
        numeric, _, sequence = frames
        self.nmedian = np.nanmedian(numeric, axis=0)
        self.nmedian = np.nan_to_num(self.nmedian)
        clean = np.where(np.isfinite(numeric), numeric, self.nmedian)
        self.nmean = clean.mean(0)
        self.nsd = np.maximum(clean.std(0), 1e-5)
        self.smean = np.nan_to_num(np.nanmean(sequence, axis=(0, 1)))
        self.ssd = np.nan_to_num(np.nanstd(sequence, axis=(0, 1)), nan=1.)
        self.ssd = np.maximum(self.ssd, 1e-5)
        return self

    def transform(self, frames):
        numeric, fixed, sequence = frames
        nmissing = (~np.isfinite(numeric)).astype(float)
        n = np.where(np.isfinite(numeric), numeric, self.nmedian)
        n = np.clip((n - self.nmean) / self.nsd, -8, 8)
        seq_missing = np.isfinite(sequence).astype(float)
        seq = np.where(np.isfinite(sequence), sequence, self.smean)
        seq = np.clip((seq - self.smean) / self.ssd, -8, 8)
        return (np.concatenate([n, nmissing, fixed], axis=1).astype(np.float32),
                np.concatenate([seq, seq_missing], axis=2).astype(np.float32))


class Correction(nn.Module):
    def __init__(self, property_dim):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(property_dim, 32), nn.GELU(), nn.LayerNorm(32))
        self.lags = nn.Linear(2 * RENT_CHANNELS, 32)
        self.position = nn.Parameter(torch.randn(1, 12, 32) * .01)
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.gate = nn.Linear(property_dim, 1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.2)

    def forward(self, prop, seq):
        q = self.query(prop).unsqueeze(1)
        lag_tokens = self.lags(seq) + self.position
        context, _ = self.attention(q, lag_tokens, lag_tokens, need_weights=False)
        representation = torch.cat([q.squeeze(1), context.squeeze(1)], dim=1)
        return 150. * torch.tanh(self.head(representation).squeeze(1)) * \
            torch.sigmoid(self.gate(prop).squeeze(1))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(actual, predicted, weights):
    return float(np.sqrt(np.average((actual - predicted) ** 2, weights=weights)))


def predict(model, data, device):
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(data[0]), 1024):
            x = torch.as_tensor(data[0][start:start + 1024], device=device)
            s = torch.as_tensor(data[1][start:start + 1024], device=device)
            out.append(model(x, s).cpu().numpy())
    return np.concatenate(out).astype(float)


def train(train_frames, validation_frames, regularization, seed, epochs, device):
    seed_all(seed)
    arrays = [raw_inputs(*item) for item in train_frames]
    train_raw = tuple(np.concatenate([a[j] for a in arrays], axis=0) for j in range(3))
    pre = Preprocess().fit(train_raw)
    train_data = pre.transform(train_raw)
    tables = [item[1] for item in train_frames]
    y = np.concatenate([t.actual.to_numpy(float) - t.ABCDE_equal.to_numpy(float) for t in tables])
    weights = np.concatenate([t.weight.to_numpy(float) for t in tables])
    weights = weights / weights.mean()
    x = [torch.as_tensor(v, device=device) for v in train_data]
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    wt = torch.as_tensor(weights, dtype=torch.float32, device=device)
    model = Correction(x[0].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    rng = np.random.default_rng(seed)
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        indices = rng.permutation(len(y))
        for start in range(0, len(indices), 256):
            ii = torch.as_tensor(indices[start:start + 256], device=device)
            delta = model(x[0][ii], x[1][ii])
            loss = torch.sum(wt[ii] * ((yt[ii] - delta).square() +
                                       regularization * delta.square())) / wt[ii].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        if validation_frames and (epoch in (1, 2, 3, 5, 8, 12, 18, 25, 35)):
            vb, vt = validation_frames[0]
            vd = pre.transform(raw_inputs(vb, vt))
            correction = predict(model, vd, device)
            base = vt.ABCDE_equal.to_numpy(float)
            records.append({"epoch": epoch,
                            "validation_rmse": rmse(vt.actual.to_numpy(float), base + correction,
                                                     vt.weight.to_numpy(float)),
                            "validation_delta_mean_abs": float(np.mean(np.abs(correction)))})
    return model, pre, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=8.)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [SOURCE, PREPARED,
        REPORTS / "tabpfn_abcde_ensemble/oof_provenance_2022-09.json",
        REPORTS / "tabpfn_abcde_ensemble/oof_provenance_2023-07.json"]}
    frames = load_frames()
    for name in ("2022-09", "2023-07"):
        proof = json.loads((REPORTS / f"tabpfn_abcde_ensemble/oof_provenance_{name}.json").read_text())
        if proof["context_max_month"] >= name or proof["validation_targets_used"] or proof["fold_target_labels_in_context"]:
            raise ValueError("Historical teacher provenance is noncausal")
    settings = {"sources": source_hashes, "selection":
        "2022-09 fit, 2023-07 validation; locked regularization and epoch then refit on both",
        "teacher": "ABCDE_equal, out-of-origin at all three origins", "seeds": SEEDS,
        "regularizations": REGULARIZATIONS, "max_epochs": MAX_EPOCHS,
        "correction": "zero-initialized 12-lag rent cross-attention, capped by tanh and sigmoid gate",
        "outer_selection": False}
    settings_path = OUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing experiment settings differ")
    write_json(settings_path, settings)
    if args.device == "cuda":
        wait_for_gpu(min_free_gib=args.min_free_gib, max_utilization=35.,
                     stable_samples=2, poll_seconds=15., status_path=OUT / "gpu_status.json")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable after capacity gate")
    device = torch.device(args.device)
    first, second = frames["early_2022"], frames["early_2023"]
    base = second[1].ABCDE_equal.to_numpy(float)
    base_rmse = rmse(second[1].actual.to_numpy(float), base, second[1].weight.to_numpy(float))
    candidates = [{"regularization": None, "epoch": 0,
                   "validation_rmse": base_rmse, "delta_rmse": 0.}]
    for reg in REGULARIZATIONS:
        _, _, records = train([first], [second], reg, 42, MAX_EPOCHS, device)
        for row in records:
            candidates.append({"regularization": reg, "epoch": row["epoch"],
                               "validation_rmse": row["validation_rmse"],
                               "delta_rmse": row["validation_rmse"] - base_rmse,
                               "mean_abs_correction": row["validation_delta_mean_abs"]})
        print("SELECT", reg, min(r["validation_rmse"] for r in records), flush=True)
    candidates = pd.DataFrame(candidates).sort_values(["validation_rmse", "epoch"])
    candidates.to_csv(OUT / "selection.csv", index=False)
    selected = candidates.iloc[0]
    write_json(OUT / "selection.json", {"regularization": None if pd.isna(selected.regularization)
               else float(selected.regularization), "epochs": int(selected.epoch),
               "validation_rmse": float(selected.validation_rmse), "baseline_rmse": base_rmse,
               "chosen_no_change": bool(pd.isna(selected.regularization))})
    print("LOCKED", selected.to_dict(), flush=True)
    outer = frames["outer"]
    scores = []
    monthly = []
    if pd.isna(selected.regularization):
        print("Historical validation chose no correction; outer diagnostic skipped", flush=True)
    else:
        for seed in SEEDS:
            model, pre, _ = train([first, second], [], float(selected.regularization), seed,
                                  int(selected.epoch), device)
            correction = predict(model, pre.transform(raw_inputs(*outer)), device)
            pred = outer[1].ABCDE_equal.to_numpy(float) + correction
            actual = outer[1].actual.to_numpy(float)
            weight = outer[1].weight.to_numpy(float)
            output = outer[1][["origin", "month", "actual", "weight", "ABCDE_equal"]].copy()
            output.insert(0, "row_id", outer[1].index.to_numpy())
            output["correction"] = correction
            output["hybrid"] = pred
            output.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
            scores.append({"seed": seed, "rows": len(output), "teacher_rmse": rmse(actual,
                outer[1].ABCDE_equal.to_numpy(float), weight), "hybrid_rmse": rmse(actual, pred, weight),
                "delta_rmse": rmse(actual, pred, weight) - rmse(actual,
                    outer[1].ABCDE_equal.to_numpy(float), weight),
                "mean_correction": float(np.mean(correction)),
                "mean_abs_correction": float(np.mean(np.abs(correction)))})
            for month, group in output.groupby("month", sort=True):
                w = group.weight.to_numpy(float)
                y = group.actual.to_numpy(float)
                monthly.append({"seed": seed, "month": month, "rows": len(group),
                    "teacher_rmse": rmse(y, group.ABCDE_equal.to_numpy(float), w),
                    "hybrid_rmse": rmse(y, group.hybrid.to_numpy(float), w),
                    "mean_correction": float(np.mean(group.correction)),
                    "teacher_bias": float(np.average(group.ABCDE_equal - group.actual, weights=w))})
            print("OUTER", scores[-1], flush=True)
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_scores.csv", index=False)
    if {str(p.relative_to(ROOT)): sha256(p) for p in [SOURCE, PREPARED,
        REPORTS / "tabpfn_abcde_ensemble/oof_provenance_2022-09.json",
        REPORTS / "tabpfn_abcde_ensemble/oof_provenance_2023-07.json"]} != source_hashes:
        raise ValueError("Source artifacts changed during experiment")
    write_json(OUT / "verification.json", {"sources_unchanged": True,
        "historical_teacher_out_of_origin": True, "outer_labels_used_for_selection": False,
        "outer_rows": len(outer[1]), "historical_rows": len(first[1]) + len(second[1])})


if __name__ == "__main__":
    main()
