"""Train official TabM from scratch on the exact TabPFN chronological features.

Earlier full December 2023-May 2024 queries select configuration and epoch.
The selected model is refitted on all 121,005 fitting rows for a fixed epoch
count, then scored on the full June 2024-March 2025 development holdout.
"""

from __future__ import annotations

import argparse
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
import torch

from analysis.config import ROOT, REPORTS, DATA
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import HISTORY_COLUMNS, FEATURES, origin_history, LocalTemporalModel
from analysis.experiments.spatiotemporal_hybrid import spatial_frame, TARGET
from analysis.experiments.tabm_regression import TabMPreprocessor, make_model, member_mse, predict


OUT = REPORTS / "tabm_80_20"
CACHE = ROOT / ".cache/tabm_80_20"
BASE = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
REFERENCE = REPORTS / "tabpfn_v3_5_fast_calendar_80_20/predictions_outer_categorical.csv"
CANDIDATES = {
    "plain_lr002": {"embedding": "none", "lr": .002, "k": 32},
    "ple_lr002": {"embedding": "ple", "lr": .002, "k": 32},
    "ple_lr001": {"embedding": "ple", "lr": .001, "k": 32},
}
SEEDS = (42, 43, 44)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Settings changed; preserve the existing directory: {path}")
    write_json(path, value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def load_base():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json").read_text())
    if sha256(BASE) != manifest["sha256"]:
        raise ValueError("TabPFN source feature artifact changed")
    base = joblib.load(BASE)
    assert base["X_train"].shape == (121005, 77) and base["X_query"].shape == (28995, 77)
    assert list(base["X_train"]) == list(base["X_query"])
    assert pd.PeriodIndex(base["train_month"], freq="M").max() < pd.PeriodIndex(base["query_month"], freq="M").min()
    assert set(base["X_train"].index).isdisjoint(base["X_query"].index)
    settings = {
        "source_artifact_sha256": manifest["sha256"], "source_features": list(base["X_train"]),
        "categorical_features": list(base["X_train"].select_dtypes(include="category")),
        "outer_fit_rows": 121005, "outer_query_rows": 28995,
        "outer_fit_months": ["2021-01", "2024-05"], "outer_query_months": ["2024-06", "2025-03"],
        "inner_origin": "2023-12", "inner_query_months": ["2023-12", "2024-05"],
        "inner_query_sampling": "all rows; histories rebuilt at November 2023 cutoff",
        "target": "rent minus saved temporal baseline; reconstruct by adding baseline once",
        "candidates": CANDIDATES, "max_epochs": 100, "patience": 12, "batch_size": 256,
        "architecture": {"package": "official tabm", "type": "tabm", "n_blocks": 3,
                         "d_block": 256, "dropout": .1, "k": 32, "categoricals": "official one-hot",
                         "ple_bins": 48, "ple_dimension": 16, "ple_version": "B"},
        "optimizer": {"name": "AdamW", "weight_decay": .0003, "gradient_clip": 1.0,
                      "schedule": "constant learning rate"},
        "preprocessing": "fit-only median imputation, missing indicators, standardization, constant removal; unknown/missing category bucket 0",
        "precision": "BF16 autocast, FP32 member MSE and prediction averaging; TF32 high",
        "loss": "mean squared error of each internal ensemble member before averaging",
        "selection": "minimum earlier full-query SGD RMSE; ties use declared candidate order",
        "final_refit": "all 121005 rows, selected epoch count, no outer early stopping",
        "primary_seed": 42, "final_seeds": SEEDS, "outer_labels_used_for_selection": False,
        "versions": {p: version(p) for p in ("tabm", "rtdl_num_embeddings", "torch", "numpy", "pandas", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(), "gpu_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                          (DATA / "train.csv", DATA / "test.csv", REFERENCE)},
        "code_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                        (Path(__file__), ROOT / "analysis/experiments/tabm_regression.py")},
        "library_hashes": {p: sha256(ROOT / ".venv/Lib/site-packages" / p)
                           for p in ("tabm.py", "rtdl_num_embeddings.py")},
    }
    immutable(OUT / "settings.json", settings)
    return base


def prepare_inner(base):
    path, manifest = CACHE / "inner_features.joblib", OUT / "inner_features.json"
    if path.exists() and manifest.exists():
        if sha256(path) != json.loads(manifest.read_text())["sha256"]:
            raise ValueError("Inner feature cache changed")
        return joblib.load(path)
    origin = pd.Period("2023-12", "M")
    months = pd.PeriodIndex(base["train_month"], freq="M")
    fit_pos, query_pos = np.flatnonzero(months < origin), np.flatnonzero(months >= origin)
    x, q = base["X_train"].iloc[fit_pos].copy(), base["X_train"].iloc[query_pos].copy()
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    fitting = frame.loc[x.index]
    blind = frame.loc[q.index].drop(columns=TARGET)
    assert fitting["_MONTH"].max() < blind["_MONTH"].min()
    q.loc[:, HISTORY_COLUMNS] = origin_history(fitting, blind).reindex(q.index).to_numpy()
    q.loc[:, FEATURES] = LocalTemporalModel(fitting).features(blind, origin-1).reindex(q.index).to_numpy()
    baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    if not np.isfinite(baseline).all():
        raise ValueError("Inner query baseline is missing")
    result = {"X_train": x, "X_query": q,
              "y_train": np.asarray(base["y_train"], float)[fit_pos],
              "y_query": np.asarray(base["y_train"], float)[query_pos],
              "train_baseline": np.asarray(base["train_baseline"], float)[fit_pos], "query_baseline": baseline,
              "train_month": np.asarray(base["train_month"])[fit_pos],
              "query_month": np.asarray(base["train_month"])[query_pos]}
    assert set(x.index).isdisjoint(q.index) and len(x) == 102829 and len(q) == 18176
    joblib.dump(result, path, compress=3)
    write_json(manifest, {"artifact": str(path.relative_to(ROOT)), "sha256": sha256(path),
                         "fitting_rows": len(x), "calibration_rows": len(q), "feature_count": len(x.columns),
                         "query_history_cutoff": "2023-11", "query_targets_used_for_features": False})
    print(f"PREPARED inner: {len(x)} fit, {len(q)} complete earlier queries", flush=True)
    return result


def fit_model(name, config, seed, bundle, epochs, *, validation, smoke=False):
    report_dir, model_dir = OUT / name, CACHE / name
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path, latest_path = model_dir / "model.pt", model_dir / "latest.pt"
    result_path = report_dir / "training.json"
    immutable(report_dir / "settings.json", {"config": config, "seed": seed, "epochs": epochs,
              "validation": validation, "smoke": smoke, "fit_rows": len(bundle["X_train"])})
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if sha256(final_path) != result["model_sha256"]:
            raise ValueError("Saved TabM checkpoint changed")
        print(f"REUSE {name}", flush=True)
        return result
    seed_all(seed)
    clear()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pre_path = model_dir / "preprocessor.joblib"
    if pre_path.exists():
        pre = joblib.load(pre_path)
    else:
        pre = TabMPreprocessor().fit(bundle["X_train"])
        joblib.dump(pre, pre_path)
    numeric, categorical = pre.transform(bundle["X_train"])
    target = np.asarray(bundle["y_train"], float)-bundle["train_baseline"]
    mean, std = float(target.mean()), float(target.std())
    assert std > 0 and np.isfinite(target).all()
    x_num, x_cat = torch.as_tensor(numeric, device="cuda"), torch.as_tensor(categorical, device="cuda")
    y = torch.as_tensor(((target-mean)/std).astype(np.float32), device="cuda")
    model, bins = make_model(config, numeric, pre.cardinalities)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.0003)
    shuffle = torch.Generator(device="cuda").manual_seed(seed)
    if validation:
        val_num, val_cat = pre.transform(bundle["X_query"])
        if smoke:
            val_num, val_cat = val_num[:256], val_cat[:256]
        v_num, v_cat = torch.as_tensor(val_num,device="cuda"),torch.as_tensor(val_cat,device="cuda")
    start_epoch, best, best_epoch, stale, history, previous_seconds = 1, float("inf"), 0, 0, [], 0.
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best, best_epoch, stale = saved["epoch"]+1, saved["best"], saved["best_epoch"], saved["stale"]
        history, previous_seconds = saved["history"], saved["seconds"]
        shuffle.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    print(f"TRAIN {name}: rows={len(numeric)} numeric={numeric.shape[1]} onehot={sum(pre.cardinalities)} "
          f"parameters={sum(p.numel() for p in model.parameters())} epochs={epochs}", flush=True)
    for epoch in range(start_epoch, epochs+1):
        model.train()
        loss_sum, seen = 0., 0
        for step, ids in enumerate(torch.randperm(len(y),generator=shuffle,device="cuda").split(256)):
            if smoke and step == 2:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(x_num[ids], x_cat[ids])
                loss = member_mse(prediction, y[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(grad):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            loss_sum += float(loss.detach())*len(ids)
            seen += len(ids)
        record = {"epoch": epoch, "train_member_mse_standardized": loss_sum/seen}
        if validation:
            prediction = predict(model,v_num,v_cat,mean,std)
            baseline = np.asarray(bundle["query_baseline"][:len(prediction)],float)
            actual = np.asarray(bundle["y_query"][:len(prediction)],float)
            score = metrics(actual,prediction+baseline)["rmse_sgd"]
            record["calibration_rmse_sgd"] = score
            if score < best-1e-4:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"model": model.state_dict(), "bins": bins, "config": config,
                            "target_mean": mean,"target_std": std},final_path)
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(record)
        write_json(report_dir / "learning_curve.json",history)
        elapsed = previous_seconds+time.perf_counter()-started
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best": best,"best_epoch": best_epoch,"stale": stale,"history": history,
                    "seconds": elapsed,"shuffle_rng": shuffle.get_state(),
                    "torch_rng": torch.get_rng_state(),"cuda_rng": torch.cuda.get_rng_state()},latest_path)
        print(f"EPOCH {name} {epoch}: loss={record['train_member_mse_standardized']:.5f} "
              f"cal_RMSE={record.get('calibration_rmse_sgd')} best_epoch={best_epoch} seconds={elapsed:.1f}",flush=True)
        if validation and stale >= 12:
            break
    if not validation:
        torch.save({"model": model.state_dict(), "bins": bins, "config": config,
                    "target_mean": mean,"target_std": std},final_path)
    result = {"name": name,"config": config,"seed": seed,"best_epoch": best_epoch,
              "epochs_run": len(history),"calibration_rmse_sgd": best if validation else None,
              "seconds": previous_seconds+time.perf_counter()-started,
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3,
              "fit_rows": len(numeric),"processed_numeric_features": numeric.shape[1],
              "category_cardinalities": pre.cardinalities,"model_sha256": sha256(final_path),
              "preprocessor_sha256": sha256(pre_path),"outer_labels_used_for_selection": False}
    if validation:
        payload = torch.load(final_path,map_location="cpu",weights_only=True)
        model.load_state_dict(payload["model"])
        expected = predict(model,v_num,v_cat,mean,std)
        clone,_ = make_model(config,numeric,pre.cardinalities,bins=payload["bins"])
        clone.load_state_dict(payload["model"])
        clone = clone.cuda()
        replay = predict(clone,v_num,v_cat,mean,std)
        np.testing.assert_array_equal(expected,replay)
        result["checkpoint_replay_max_difference"] = float(np.max(np.abs(expected-replay)))
        if not smoke:
            frame = pd.DataFrame({"row_id": bundle["X_query"].index,"month": bundle["query_month"],
                                  "actual": bundle["y_query"],"predicted": replay+bundle["query_baseline"]})
            frame.to_csv(report_dir / "predictions.csv",index=False,float_format="%.10f")
        del clone,payload
    write_json(result_path,result)
    del model, optimizer, x_num, x_cat, y
    if validation:
        del v_num,v_cat
    clear()
    print("DONE "+json.dumps(result),flush=True)
    return result


def score_final(name, base):
    directory, cache = OUT / name,CACHE / name
    if (directory / "predictions.csv").exists():
        print(f"REUSE scored {name}",flush=True)
        return
    clear()
    started = time.perf_counter()
    pre = joblib.load(cache / "preprocessor.joblib")
    payload = torch.load(cache / "model.pt",map_location="cpu",weights_only=True)
    numeric, categorical = pre.transform(base["X_query"])
    model,_ = make_model(payload["config"],numeric,pre.cardinalities,bins=payload["bins"])
    model.load_state_dict(payload["model"])
    model = model.cuda()
    residual = predict(model,torch.as_tensor(numeric,device="cuda"),torch.as_tensor(categorical,device="cuda"),
                       payload["target_mean"],payload["target_std"])
    frame = pd.DataFrame({"row_id":base["X_query"].index,"month":base["query_month"],
                          "actual":base["y_query"],"predicted":residual+base["query_baseline"]})
    frame.to_csv(directory / "predictions.csv",index=False,float_format="%.10f")
    score = {"name": name,"rows":len(frame),**metrics(frame.actual,frame.predicted),
             "predict_seconds":time.perf_counter()-started,"predictions_sha256":sha256(directory / "predictions.csv")}
    write_json(directory / "score.json",score)
    print("OUTER "+json.dumps(score),flush=True)
    del model,payload
    clear()


def summarize():
    reference = pd.read_csv(REFERENCE)
    rows,month_rows,group_rows = [],[],[]
    comparisons = [("tabpfn_frozen",REFERENCE,"predicted"),
                   ("tabpfn_original_diagscale",REPORTS / "tabpfn_diagscale_categorical_alignment_80_20/predictions.csv","diagscale_aligned"),
                   ("tabpfn_mse_only_development",REPORTS / "tabpfn_components_80_20/outer/diag_mse_only/predictions.csv","predicted")]
    comparisons += [(f"tabm_seed{seed}",OUT / f"final_seed{seed}/predictions.csv","predicted") for seed in SEEDS]
    raw = pd.read_csv(DATA / "train.csv").loc[reference.row_id]
    for label,path,column in comparisons:
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(frame.row_id,reference.row_id)
        np.testing.assert_array_equal(frame.month,reference.month)
        np.testing.assert_allclose(frame.actual,reference.actual,atol=1e-8,rtol=0)
        assert len(frame)==28995 and frame.row_id.is_unique and np.isfinite(frame[column]).all()
        rows.append({"model":label,"n":len(frame),**metrics(frame.actual,frame[column])})
        for month, part in frame.groupby("month",sort=True):
            month_rows.append({"model":label,"month":month,"n":len(part),**metrics(part.actual,part[column])})
        for kind,mask in [("floor_area_above_130sqm",raw.FLOOR_AREA_SQM.to_numpy()>130),
                          ("executive",raw.FLAT_TYPE.str.casefold().to_numpy()=="executive")]:
            group_rows.append({"model":label,"group":kind,"n":int(mask.sum()),
                               **metrics(frame.actual.to_numpy()[mask],frame[column].to_numpy()[mask])})
    scores = pd.DataFrame(rows)
    baseline = float(scores.set_index("model").loc["tabpfn_frozen","rmse_sgd"])
    scores["delta_vs_frozen_sgd"] = scores.rmse_sgd-baseline
    scores.to_csv(OUT / "scores.csv",index=False,float_format="%.8f")
    pd.DataFrame(month_rows).to_csv(OUT / "monthly_scores.csv",index=False,float_format="%.8f")
    pd.DataFrame(group_rows).to_csv(OUT / "size_group_scores.csv",index=False,float_format="%.8f")
    print(scores.to_string(index=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("all","prepare","calibration","final","summarize"),default="all")
    parser.add_argument("--candidate",choices=list(CANDIDATES))
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    base = load_base()
    inner = prepare_inner(base)
    if args.smoke:
        for name in ("plain_lr002","ple_lr002"):
            fit_model(f"smoke/{name}",CANDIDATES[name],42,inner,1,validation=True,smoke=True)
        return
    if args.stage in ("all","calibration"):
        for name in ([args.candidate] if args.candidate else CANDIDATES):
            fit_model(f"calibration/{name}",CANDIDATES[name],42,inner,100,validation=True)
    if args.stage in ("all","final"):
        candidates = [json.loads((OUT / "calibration" / n / "training.json").read_text()) for n in CANDIDATES]
        selected = min(candidates,key=lambda r:r["calibration_rmse_sgd"])
        selection = {"candidate":selected["name"].split("/")[-1],"config":selected["config"],
                     "fixed_refit_epochs":selected["best_epoch"],"earlier_rmse_sgd":selected["calibration_rmse_sgd"],
                     "outer_labels_used_for_selection":False}
        immutable(OUT / "selection.json",selection)
        for seed in SEEDS:
            name = f"final_seed{seed}"
            fit_model(name,selection["config"],seed,base,selection["fixed_refit_epochs"],validation=False)
            score_final(name,base)
    if args.stage in ("all","summarize"):
        summarize()


if __name__ == "__main__":
    main()
