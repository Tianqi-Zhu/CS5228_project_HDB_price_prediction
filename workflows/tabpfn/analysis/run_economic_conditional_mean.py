"""Run a matched, joint temporal-economic conditional-mean experiment."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.compare_local_economy_landscape import immutable, write_json
from analysis.experiments.economic_conditional_mean import EconomicConditionalMean
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatial_temporal_transformer import Experiment, FeaturePreprocessor, SequencePreprocessor
from analysis.run_spatial_temporal_transformer import arrays, prediction, seed_all, slice_smoke, weighted_rmse
from analysis.experiments.stock_index_regions import block_positions

SOURCE = REPORTS / "spatial_temporal_transformer"
PREPARED = ROOT / ".cache/spatial_temporal_transformer/prepared.joblib"
OUT = REPORTS / "economic_conditional_mean"
CACHE = ROOT / ".cache/economic_conditional_mean"
DESIGNS = ("rent", "rent_coe")
FOLDS = ("early_2022", "early_2023")


def gpu_gate(args):
    if args.device == "cuda":
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        wait_for_gpu(min_free_gib=8., max_utilization=35., stable_samples=2,
                     poll_seconds=15., status_path=OUT / "gpu_status.json")


def fit(prepared, fold, design, seed, args, epochs=None):
    if fold == "outer" and epochs is None:
        raise ValueError("Later fit requires earlier-locked epochs")
    phase = f"smoke_{args.device}" if args.stage == "smoke" else "calibration" if fold in FOLDS else fold
    tag = f"{phase}/{design}/{fold}/seed{seed}"
    report, cache = OUT / tag, CACHE / tag
    report.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    immutable(report / "settings.json", dict(fold=fold, design=design, seed=seed, epochs=epochs,
                                             phase=phase, device=args.device))
    result_path, best_path, latest_path = report / "result.json", cache / "model.pt", cache / "latest.pt"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert sha256(best_path) == result["checkpoint_sha256"]
        assert sha256(report / "predictions.csv") == result["predictions_sha256"]
        assert sha256(cache / "preprocessors.joblib") == result["preprocessors_sha256"]
        print("REUSE", tag, flush=True)
        return result
    bundle = prepared["folds"][fold + "_d0"]
    if args.stage == "smoke":
        bundle = slice_smoke(bundle)
    assert set(bundle["X_train"].index).isdisjoint(bundle["X_query"].index)
    assert max(bundle["train_month"]) < min(bundle["query_month"])
    gpu_gate(args)
    seed_all(seed)
    device = torch.device(args.device)
    pre = FeaturePreprocessor().fit(bundle["X_train"])
    seq_pre = SequencePreprocessor().fit(bundle["train_sequence"])
    joblib.dump((pre, seq_pre), cache / "preprocessors.joblib")
    spec = Experiment("conditional_" + design, design, "lstm")
    fitting = [torch.from_numpy(a) for a in arrays(bundle, pre, seq_pre, spec, "train")]
    query = [torch.from_numpy(a) for a in arrays(bundle, pre, seq_pre, spec, "query")]
    residual = bundle["y_train"] - bundle["train_baseline"]
    mean, sd = float(residual.mean()), float(residual.std())
    assert np.isfinite(residual).all() and sd > 1e-6
    y = torch.from_numpy(((residual - mean) / sd).astype(np.float32))
    model = EconomicConditionalMean(fitting[0].shape[1], pre.cardinalities, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    shuffle = torch.Generator().manual_seed(seed)
    first_epoch, best, best_epoch, stale, history = 1, float("inf"), 0, 0, []
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        first_epoch, best, best_epoch, stale, history = saved["epoch"] + 1, saved["best"], saved["best_epoch"], saved["stale"], saved["history"]
        shuffle.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"])
    total_epochs = 1 if args.stage == "smoke" else 30 if epochs is None else epochs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    print("TRAIN", tag, len(y), "rows", flush=True)
    for epoch in range(first_epoch, total_epochs + 1):
        if epochs is None and stale >= 5:
            break
        model.train()
        total, seen = 0., 0
        for step, ids in enumerate(torch.randperm(len(y), generator=shuffle).split(256)):
            if args.stage == "smoke" and step == 2:
                break
            inputs = [a[ids].to(device) for a in fitting]
            target = y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                base, shift, interaction = model.components(*inputs)
                correction = shift.float() + interaction.float()
                error = base.float() + correction - target
                loss = error.square().mean() + .01 * correction.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            seen += len(ids)
        validation = None
        if epochs is None:
            pred = prediction(model, query, mean, sd, device) + bundle["query_baseline"]
            validation = weighted_rmse(bundle["y_query"], pred, bundle["query_weights"])
            improved = validation < best - 1e-4
            best, best_epoch, stale = (validation, epoch, 0) if improved else (best, best_epoch, stale + 1)
        else:
            improved, best_epoch = True, epoch
        if improved:
            temp = best_path.with_suffix(".tmp.pt")
            torch.save(dict(model=model.state_dict(), mean=mean, sd=sd), temp)
            temp.replace(best_path)
        history.append(dict(epoch=epoch, loss=total / seen, validation_rmse_sgd=validation))
        write_json(report / "learning_curve.json", history)
        write_json(OUT / "worker_status.json", dict(state="training", run=tag, epoch=epoch,
                                                   best_epoch=best_epoch, pid=os.getpid()))
        temp = latest_path.with_suffix(".tmp.pt")
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        best=best, best_epoch=best_epoch, stale=stale, history=history,
                        shuffle_rng=shuffle.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None), temp)
        temp.replace(latest_path)
        print(f"EPOCH {tag} {epoch}: loss={total / seen:.5f} validation={validation}", flush=True)
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    pred = prediction(model, query, mean, sd, device) + bundle["query_baseline"]
    clone = EconomicConditionalMean(fitting[0].shape[1], pre.cardinalities, spec).to(device)
    clone.load_state_dict(payload["model"])
    replay = prediction(clone, query, mean, sd, device) + bundle["query_baseline"]
    np.testing.assert_array_equal(pred, replay)
    shifts, interactions = [], []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(pred), 512):
            values = [a[offset:offset + 512].to(device) for a in query]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, shift, interaction = model.components(*values)
            shifts.extend(shift.float().cpu().numpy().astype(float) * sd)
            interactions.extend(interaction.float().cpu().numpy().astype(float) * sd)
    frame = pd.DataFrame(dict(row_id=bundle["X_query"].index, month=bundle["query_month"],
                              actual=bundle["y_query"], predicted=pred, weight=bundle["query_weights"],
                              horizon=np.rint(bundle["query_context"][:, 0] * 16).astype(int),
                              conditional_shift_sgd=shifts, interaction_sgd=interactions))
    assert frame.row_id.is_unique and np.isfinite(frame.predicted).all()
    frame.to_csv(report / "predictions.csv", index=False, float_format="%.17g")
    result = dict(fold=fold, design=design, seed=seed, best_epoch=best_epoch, epochs_run=len(history),
                  weighted_rmse_sgd=weighted_rmse(frame.actual, frame.predicted, frame.weight),
                  bias_sgd=float(np.average(frame.predicted - frame.actual, weights=frame.weight)),
                  mae_sgd=float(np.average(abs(frame.predicted - frame.actual), weights=frame.weight)),
                  rows=len(frame), parameters=sum(p.numel() for p in model.parameters()),
                  seconds=time.perf_counter() - start,
                  peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.,
                  checkpoint_sha256=sha256(best_path), predictions_sha256=sha256(report / "predictions.csv"),
                  preprocessors_sha256=sha256(cache / "preprocessors.joblib"),
                  checkpoint_replay_max_difference=float(abs(pred - replay).max()),
                  outer_labels_used_for_selection=False, all_parameters_jointly_trainable=all(p.requires_grad for p in model.parameters()))
    write_json(result_path, result)
    print("DONE " + json.dumps(result), flush=True)
    del model, clone, optimizer, fitting, query, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def earlier_comparison(results):
    rows = []
    for fold in FOLDS:
        control = json.loads((SOURCE / f"calibration/T0/{fold}/result.json").read_text())
        matched = {r["design"]: r for r in results if r["fold"] == fold}
        for design, result in matched.items():
            rows.append(dict(fold=fold, design=design, rmse_sgd=result["weighted_rmse_sgd"],
                             delta_vs_T0=result["weighted_rmse_sgd"] - control["weighted_rmse_sgd"],
                             delta_vs_rent=result["weighted_rmse_sgd"] - matched["rent"]["weighted_rmse_sgd"],
                             best_epoch=result["best_epoch"]))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "earlier_comparison.csv", index=False)
    coe = comparison.loc[comparison.design.eq("rent_coe")]
    passed = bool((coe.delta_vs_T0 < 0).all() and (coe.delta_vs_rent < 0).all())
    epochs = {design: max(1, int(np.floor(np.mean([r["best_epoch"] for r in results if r["design"] == design]) + .5))) for design in DESIGNS}
    value = dict(advanced=passed, refit_epochs=epochs, criterion="COE improves matched rent-only and T0 in each earlier window", outer_labels_used=False)
    immutable(OUT / "selection.json", value)
    return comparison, value


def summarize(comparison, selection, outer):
    lines = ["# Temporal economic conditioning of expected rent", "",
             "Jointly trained property FT-Transformer plus an eight-unit LSTM over twelve historical months. A mean head predicts a rent-level shift and eight property-dependent temporal loadings. Prediction is the original causal local baseline plus the property residual and temporal conditional correction; no future observed mean is supplied.", "",
             "Matched rent-only and rent+COE designs differ only in the available COE channels. Stocks are masked. Both start with identical property weights and a zero temporal correction. All parameters train with standardized individual-rental MSE plus0.01 times squared correction, using the original chronological folds and extra month of macro lag.", "",
             "## Earlier chronological comparisons", "", "| Window | Design | RMSE (SGD) | Change vs T0 | Change vs matched rent-only | Best epoch |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        lines.append(f"| {row.fold} | {row.design} | {row.rmse_sgd:.3f} | {row.delta_vs_T0:+.3f} | {row.delta_vs_rent:+.3f} | {row.best_epoch} |")
    lines += ["", f"Earlier gate passed: **{selection['advanced']}**. It requires the COE model to beat matched rent-only and original T0 in both earlier windows. Refit epochs selected earlier: " + json.dumps(selection["refit_epochs"]) + ".", ""]
    if outer:
        lines += ["## Reused later development comparison", "", "| Design | Seed | RMSE (SGD) |", "| --- | ---: | ---: |"]
        for row in outer:
            lines.append(f"| {row['design']} | {row['seed']} | {row['weighted_rmse_sgd']:.3f} |")
    else:
        lines += ["The declared earlier gate stopped this candidate. No later model fits, later scoring, or additional architecture search was performed.", ""]
    lines += ["## Limits and reproduction", "",
              "Only51 labeled months exist. Earlier windows each contain2000 sampled queries with population weights, not2000 independent economic observations. Earlier validation selects checkpoints and the gate, so these are selection results rather than independent tests. Later periods have already been repeatedly examined in the project. Existing baseline histories, code, checkpoints and submission CSVs remain unchanged.", "",
              "Run `.venv/Scripts/python.exe analysis/run_economic_conditional_mean.py --stage all --device cuda`. CPU and CUDA smoke use `--stage smoke --device cpu` and `--stage smoke --device cuda`. Settings record sources/hashes; per-run artifacts contain learning curves, conditional-shift predictions, exact checkpoint replay, timing and peak VRAM. Worker resumes completed runs and per-epoch optimizer/RNG state.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = OUT / "worker.lock"
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    try:
        manifest = json.loads((SOURCE / "prepared.json").read_text())
        assert sha256(PREPARED) == manifest["sha256"]
        paths = [Path(__file__), ROOT / "analysis/experiments/economic_conditional_mean.py",
                 ROOT / "analysis/experiments/spatial_temporal_transformer.py", PREPARED,
                 ROOT / "EXPERIMENT_PLAN_ECONOMIC_CONDITIONAL_MEAN.md", SOURCE / "prepared.json"]
        paths += [SOURCE / f"calibration/T0/{fold}/result.json" for fold in FOLDS]
        paths += [REPORTS / "tabpfn_top2_final_submission" / name for name in ("submission_tabpfn_ABCDE_equal_150k.csv", "submission_tabpfn_ABDE_weighted_150k.csv")]
        paths += [REPORTS / "tabpfn_api_77/submission_v3.5_plus_77_150k/submission_tabpfn_api.csv"]
        hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        immutable(OUT / "settings.json", dict(source_sha256=hashes, designs=DESIGNS,
            architecture="FT128x2 + LSTM8 + conditional mean shift and eight property loadings",
            protocol="EXPERIMENT_PLAN_ECONOMIC_CONDITIONAL_MEAN.md", seeds=[42, 43, 44],
            max_epochs=30, patience=5, correction_penalty=.01, gpu_free_gib=8, gpu_utilization_max=35))
        torch.set_num_threads(4)
        prepared = joblib.load(PREPARED)
        if args.stage == "smoke":
            for design in DESIGNS:
                fit(prepared, FOLDS[0], design, 42, args)
            return
        for device in ("cpu", "cuda") if args.device == "cuda" else ("cpu",):
            for design in DESIGNS:
                smoke = OUT / f"smoke_{device}/{design}/early_2022/seed42/result.json"
                assert smoke.exists(), f"Run {device} smoke first"
        results = [fit(prepared, fold, design, 42, args) for fold in FOLDS for design in DESIGNS]
        comparison, selection = earlier_comparison(results)
        outer = []
        if selection["advanced"]:
            outer = [fit(prepared, "outer", design, seed, args, epochs=selection["refit_epochs"][design])
                     for seed in (42, 43, 44) for design in DESIGNS]
            pd.DataFrame(outer).to_csv(OUT / "outer_scores.csv", index=False)
            outer_comparisons()
        summarize(comparison, selection, outer)
        for path, digest in hashes.items():
            assert sha256(ROOT / path) == digest, path
        write_json(OUT / "verification.json", dict(verified=True, source_hashes_unchanged=True,
            submissions_unchanged=True, all_checkpoint_predictions_exactly_replayed=True,
            outer_labels_used_for_selection=False, earlier_gate_passed=selection["advanced"],
            completed_calibration_fits=len(results), completed_outer_fits=len(outer)))
        write_json(OUT / "worker_status.json", dict(state="completed", advanced=selection["advanced"], pid=os.getpid()))
        print(comparison.to_string(index=False), flush=True)
        print("COMPLETE advanced=", selection["advanced"], flush=True)
    finally:
        lock.unlink(missing_ok=True)


def outer_comparisons():
    monthly, comparisons = [], []
    for seed in (42, 43, 44):
        coe = pd.read_csv(OUT / f"outer/rent_coe/outer/seed{seed}/predictions.csv", float_precision="round_trip")
        refs = {"rent": OUT / f"outer/rent/outer/seed{seed}/predictions.csv",
                "T0": SOURCE / f"outer/T0/seed{seed}/predictions.csv",
                "rank23": REPORTS / f"transformer_ranked_experiments/23_forced_decay/outer_seed{seed}.csv"}
        for name, path in refs.items():
            other = pd.read_csv(path, float_precision="round_trip")
            for column in ("row_id", "month", "actual"):
                np.testing.assert_array_equal(coe[column], other[column])
            data = coe.copy()
            data["reference"] = other["decayed_prediction" if name == "rank23" else "predicted"].to_numpy()
            stats = []
            for month, group in data.groupby("month", sort=True):
                delta = weighted_rmse(group.actual, group.predicted, group.weight) - weighted_rmse(group.actual, group.reference, group.weight)
                monthly.append(dict(seed=seed, reference=name, month=month, delta_rmse=delta,
                                    mean_shift_sgd=float(group.conditional_shift_sgd.mean()),
                                    bias_sgd=float((group.predicted - group.actual).mean())))
                stats.append([len(group), np.sum((group.predicted - group.actual)**2), np.sum((group.reference - group.actual)**2)])
            stats = np.asarray(stats)
            summed = stats[block_positions(len(stats), repeats=5000, block=3)].sum(axis=1)
            differences = np.sqrt(summed[:, 1] / summed[:, 0]) - np.sqrt(summed[:, 2] / summed[:, 0])
            low, high = np.quantile(differences, [.025, .975])
            comparisons.append(dict(seed=seed, reference=name,
                delta_rmse=weighted_rmse(data.actual, data.predicted, data.weight) - weighted_rmse(data.actual, data.reference, data.weight),
                ci_low=low, ci_high=high))
    pd.DataFrame(monthly).to_csv(OUT / "monthly_comparisons.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)


if __name__ == "__main__":
    main()
