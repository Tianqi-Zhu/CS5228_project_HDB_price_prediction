"""CLI entry point for the neural fusion ablation.

Every invocation gets its own ``run_id`` directory, so smoke, probe and full runs
never overwrite each other.  Nothing outside ``reports/neural_fusion_v1`` is touched.

Examples
--------
python main.py --split orig --models lstm_transformer --seed 42 --mode full
python main.py --split all --models all --seeds 42,77,2026 --mode full --run-id final-3seed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    CONTEXT_DIM,
    EXPERIMENT,
    HISTORY_LEN,
    CAT_FEATURES,
    MAX_HORIZON,
    PANEL_CHANNELS,
    REPORT,
    ROOT,
    SPLITS,
    TARGET,
    TARGET_SCALE,
    TRAIN_SPATIAL,
    month_index,
    month_label,
)
from data import assemble, history_stats, load_frame, split_bounds, split_frames
from models import build_model, group_index_map
from panel import Panel, save_panel_state, training_samples, validation_samples
from preprocessing import StaticPreprocessor
from training import (
    TrainingConfig,
    band_report,
    build_and_count,
    final_inner_rmse,
    fit_early_stopping,
    fit_epochs,
    metrics,
    monthly_report,
    predict,
    predictions_frame,
    save_json,
    set_seed,
)

MODEL_NAMES = ["mlp", "static_transformer", "lstm_concat", "lstm_transformer"]
MODES = {"smoke": dict(epochs=3, patience=2, batch_size=256, row_cap=20000),
         "probe": dict(epochs=25, patience=6, batch_size=512, row_cap=None),
         "full": dict(epochs=60, patience=8, batch_size=512, row_cap=None)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "platform": platform.platform(), "host": socket.gethostname(),
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "mps_available": bool(torch.backends.mps.is_available()), "device": "cpu",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def code_fingerprint() -> dict:
    """Hash of every source file in this experiment, so a run names the code that ran."""
    files = sorted(EXPERIMENT.glob("*.py"))
    return {path.name: sha256_file(path) for path in files}


def default_run_id(mode: str) -> str:
    return f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"


class SplitState:
    """Everything one split needs; stage A and stage B statistics are kept apart."""

    def __init__(self, frame: pd.DataFrame, split_id: str, seed: int, mode: str, inner_months: int):
        spec = SPLITS[split_id]
        self.split_id = split_id
        self.cutoff, self.val_end = split_bounds(split_id)
        self.cutoff_label = spec["cutoff"]
        self.val_end_label = spec["val_end"]
        self.fit, self.outer = split_frames(frame, self.cutoff, self.val_end)
        self.inner_cutoff = self.cutoff - inner_months
        if self.inner_cutoff <= 0:
            raise ValueError("Inner cutoff would precede the first month")

        # --- stage A: fitted on inner fitting months only -------------------
        inner_fit_rows = frame.loc[frame["MONTH_INDEX"] <= self.inner_cutoff]
        self.inner_panel = Panel.build(inner_fit_rows, self.inner_cutoff)
        self.inner_prep = StaticPreprocessor.fit(inner_fit_rows)
        rng = np.random.default_rng(seed)
        self.inner_train = training_samples(inner_fit_rows, self.inner_cutoff, rng, MAX_HORIZON)
        self.inner_val = validation_samples(frame, self.inner_cutoff, self.cutoff)

        # --- stage B: fitted on the whole fitting period --------------------
        self.panel = Panel.build(self.fit, self.cutoff)
        self.prep = StaticPreprocessor.fit(self.fit)
        self.train_all = training_samples(self.fit, self.cutoff, np.random.default_rng(seed), MAX_HORIZON)
        self.outer_val = validation_samples(frame, self.cutoff, self.val_end)
        self.group_indices, self.group_names = group_index_map(self.prep.numeric_columns)
        inner_indices, _ = group_index_map(self.inner_prep.numeric_columns)
        if inner_indices != self.group_indices:
            raise AssertionError("Stage A and stage B numeric group indices disagree")
        self.group_summary = {g: len(v) for g, v in self.group_names.items()}

        # --- diagnostics / boundary assertions -----------------------------
        expected = SPLITS[split_id]["val_rows"]
        if expected is not None:
            if len(self.outer_val) != expected or self.outer_val["MONTH_INDEX"].nunique() != SPLITS[split_id]["val_months"]:
                raise AssertionError(
                    f"{split_id}: outer validation is {len(self.outer_val)} rows / "
                    f"{self.outer_val['MONTH_INDEX'].nunique()} months, expected {expected} / "
                    f"{SPLITS[split_id]['val_months']}")
        assert self.outer["MONTH_INDEX"].max() <= self.val_end
        assert self.fit["MONTH_INDEX"].max() == self.cutoff
        assert self.inner_fit_last_month() == self.inner_cutoff

    def inner_fit_last_month(self) -> int:
        return int(self.inner_train["MONTH_INDEX"].max())


def prepare_mode(frame: pd.DataFrame, state: SplitState, mode: str, seed: int) -> None:
    """Uniform random row caps for smoke runs; the cap is recorded in metadata."""
    cap = MODES[mode]["row_cap"]
    if cap is None:
        return
    for name in ("inner_train", "train_all", "outer_val", "inner_val"):
        rows = getattr(state, name)
        if len(rows) > cap:
            setattr(state, name, rows.sample(n=cap, random_state=seed).sort_index())


def run_one(state: SplitState, model_name: str, seed: int, mode: str, run_dir: Path, command: str) -> dict:
    spec = MODES[mode]
    device = torch.device("cpu")
    set_seed(seed)
    uses_history = model_name in ("lstm_concat", "lstm_transformer")
    started = time.perf_counter()

    # stage A tensors: inner-fitting rows + inner holdout, both from inner statistics
    a_train = assemble(state.inner_train, state.inner_prep, state.inner_panel, uses_history)
    a_inner = assemble(state.inner_val, state.inner_prep, state.inner_panel, uses_history)
    config = TrainingConfig(model=model_name, seed=seed, epochs=spec["epochs"],
                            patience=spec["patience"], batch_size=spec["batch_size"])

    model_a, n_params = build_and_count(model_name, state.inner_prep.cardinalities(),
                                        state.group_indices, CONTEXT_DIM, config, state.group_names)
    stage_a = fit_early_stopping(
        model_a, a_train, a_inner, config, device,
        curve_path=run_dir / "curves" / f"{state.split_id}_{model_name}_seed{seed}.svg",
        title=f"{state.split_id} {model_name} seed{seed} stage A inner RMSE",
    )
    selected_epochs = max(stage_a.best_epoch, 1)

    # stage B tensors: full fitting period statistics; the epoch count is FIXED
    b_fit = assemble(state.train_all, state.prep, state.panel, uses_history)
    b_outer = assemble(state.outer_val, state.prep, state.panel, uses_history)
    config_b = TrainingConfig(**{**config.__dict__, "epochs": selected_epochs, "patience": 0})
    set_seed(seed)              # seed BEFORE construction so weights are reproducible
    model_b, _ = build_and_count(model_name, state.prep.cardinalities(),
                                 state.group_indices, CONTEXT_DIM, config_b, state.group_names)
    refit_started = time.perf_counter()
    final_train_loss = fit_epochs(model_b, b_fit, config_b, device, selected_epochs)
    refit_seconds = time.perf_counter() - refit_started

    predicted = b_outer.residual_to_sgd(predict(model_b, b_outer, device))
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"{model_name}: non-finite predictions")

    frame = predictions_frame(state.outer_val, predicted, split_id=state.split_id,
                              cutoff_month=state.cutoff_label, model_name=model_name, seed=seed)
    overall = metrics(frame["actual_rent_sgd"].to_numpy(), frame["predicted_rent_sgd"].to_numpy())

    # --- persist artefacts, all inside this run's directory --------------
    artefacts = {"predictions": run_dir / "predictions" / state.split_id,
                 "checkpoints": run_dir / "checkpoints" / state.split_id,
                 "metrics": run_dir / "metrics" / state.split_id,
                 "preprocessing": run_dir / "preprocessing" / state.split_id}
    for path in artefacts.values():
        path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artefacts["predictions"] / f"{model_name}_seed{seed}.csv", index=False)

    checkpoint = artefacts["checkpoints"] / f"{model_name}_seed{seed}.pt"
    torch.save({
        "model": model_name, "state_dict": model_b.state_dict(),
        "model_kwargs": config_b.model_kwargs(), "cardinalities": state.prep.cardinalities(),
        "group_indices": state.group_indices, "group_names": state.group_names,
        "numeric_columns": state.prep.numeric_columns,
        "model_kwargs_actual": config_b.model_kwargs(),
        "context_dim": CONTEXT_DIM,
        "split_id": state.split_id, "cutoff_month": state.cutoff_label,
        "val_end_month": state.val_end_label, "seed": seed, "target_scale": TARGET_SCALE,
        "selected_epochs": selected_epochs, "uses_history": uses_history,
        "preprocessing": state.prep.to_dict(),
        "panel_files": [f"preprocessing/{state.split_id}/panel.npz",
                        f"preprocessing/{state.split_id}/panel.json"],
        "trained_rows": len(b_fit),
    }, checkpoint)

    result = {
        "run_id": run_dir.name, "mode": mode, "split_id": state.split_id, "model": model_name,
        "seed": seed, "uses_history": uses_history, "n_parameters": n_params,
        "n_fit_rows_stage_a": len(a_train), "n_inner_val_rows": len(a_inner),
        "n_fit_rows_stage_b": len(b_fit), "n_validation_rows": len(b_outer),
        "stage_a_selected_epochs": selected_epochs,
        "stage_a_best_inner_rmse_sgd": stage_a.best_inner_rmse,
        "stage_a_epochs_run": stage_a.epochs_run,
        "stage_b_seconds": round(refit_seconds, 2),
        "stage_b_final_train_mse_ksgd2": final_train_loss,
        "stage_a_seconds": round(stage_a.seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "overall": overall, "by_horizon_band": band_report(frame),
        "by_month": monthly_report(frame).to_dict(orient="records"),
        "history_coverage": {name: history_stats(ds) for name, ds in
                             (("stage_a", a_train), ("stage_b", b_fit), ("outer", b_outer))},
        "unseen_outer": state.panel.unseen_report(state.outer_val),
        "fitting_months": [state.fit["RENT_APPROVAL_DATE"].min(), state.fit["RENT_APPROVAL_DATE"].max()],
        "inner_fitting_months": [frame_month for frame_month in
                                 (state.inner_train["RENT_APPROVAL_DATE"].min(),
                                  state.inner_train["RENT_APPROVAL_DATE"].max())],
        "validation_months": [frame["RENT_APPROVAL_DATE"].min(), frame["RENT_APPROVAL_DATE"].max()],
        "stage_a_curve": stage_a.history,
        "command": command,
    }
    save_json(artefacts["metrics"] / f"{model_name}_seed{seed}.json", result)
    return {k: v for k, v in result.items() if k not in ("by_month", "stage_a_curve")}


def verify_checkpoint(state: SplitState, model_name: str, seed: int, run_dir: Path) -> dict:
    """Reload the saved checkpoint and reproduce the exported predictions exactly."""
    checkpoint = torch.load(run_dir / "checkpoints" / state.split_id / f"{model_name}_seed{seed}.pt",
                            map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model"], checkpoint["cardinalities"], checkpoint["group_indices"],
                        checkpoint["context_dim"], group_names=checkpoint.get("group_names"),
                        **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    prep = StaticPreprocessor.from_dict(checkpoint["preprocessing"])
    from panel import load_panel_state

    panel = load_panel_state(run_dir / "preprocessing" / state.split_id, "panel")
    dataset = assemble(state.outer_val, prep, panel, checkpoint["uses_history"])
    reloaded = dataset.residual_to_sgd(predict(model, dataset, torch.device("cpu")))

    exported = pd.read_csv(run_dir / "predictions" / state.split_id / f"{model_name}_seed{seed}.csv")
    exported = exported.sort_values("SOURCE_ROW_ID", kind="stable").reset_index(drop=True)
    if len(exported) != len(reloaded):
        return {"ok": False, "reason": f"row count {len(exported)} != {len(reloaded)}"}
    delta = np.abs(exported["predicted_rent_sgd"].to_numpy() - reloaded)
    return {"ok": bool(delta.max() < 1e-4), "max_abs_delta_sgd": float(delta.max()),
            "n": int(len(reloaded)),
            "preprocessing_matches": prep.to_dict()["median"] == state.prep.to_dict()["median"],
            "panel_cutoff": panel.cutoff_month, "panel_matches": int(panel.n_months == state.panel.n_months)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neural fusion ablation for HDB rent prediction")
    parser.add_argument("--split", default="orig", choices=list(SPLITS) + ["all"])
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--mode", default="full", choices=list(MODES))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--inner-months", type=int, default=12)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    models = MODEL_NAMES if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    splits = list(SPLITS) if args.split == "all" else [args.split]
    if args.epochs is not None:
        MODES[args.mode]["epochs"] = args.epochs

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    torch.set_num_threads(2)

    run_id = args.run_id or default_run_id(args.mode)
    run_dir = REPORT / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join([".venv/bin/python", "main.py"] + (argv if argv is not None else sys.argv[1:]))
    framing = {
        "run_id": run_id, "mode": args.mode, "source": str(TRAIN_SPATIAL.relative_to(ROOT)),
        "source_sha256": sha256_file(TRAIN_SPATIAL), "environment": environment(),
        "code_fingerprint": code_fingerprint(),
        "split_metadata": {},
        "args": vars(args), "models": models, "seeds": seeds, "splits": splits,
        "command": command, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "history_len": HISTORY_LEN, "max_horizon": MAX_HORIZON, "panel_channels": PANEL_CHANNELS,
        "context_dim": CONTEXT_DIM, "target": TARGET, "target_scale": TARGET_SCALE,
        "split_specs": {sid: SPLITS[sid] for sid in splits},
    }
    save_json(run_dir / "run_metadata.json", framing)

    frame = load_frame(TRAIN_SPATIAL)
    results, checks = [], []
    for split_id in splits:
        state = SplitState(frame, split_id, seeds[0], args.mode, args.inner_months)
        prepare_mode(frame, state, args.mode, seeds[0])
        (run_dir / "preprocessing" / split_id).mkdir(parents=True, exist_ok=True)
        state.prep.save(run_dir / "preprocessing" / split_id / "stage_b_preprocessing.json")
        state.inner_prep.save(run_dir / "preprocessing" / split_id / "stage_a_preprocessing.json")
        save_panel_state(state.panel, run_dir / "preprocessing" / split_id, "panel")
        save_panel_state(state.inner_panel, run_dir / "preprocessing" / split_id, "panel_stage_a")
        framing["split_metadata"][split_id] = {
            "cutoff_month": state.cutoff_label, "val_end_month": state.val_end_label,
            "fit_months": [state.fit["RENT_APPROVAL_DATE"].min(), state.fit["RENT_APPROVAL_DATE"].max()],
            "inner_cutoff_month": month_label(state.inner_cutoff),
            "n_fit_rows": int(len(state.fit)), "n_outer_rows": int(len(state.outer_val)),
            "numeric_column_order": state.prep.numeric_columns,
            "group_indices": state.group_indices, "group_names": state.group_names,
            "panel_cutoff_month": int(state.panel.cutoff_month),
            "stage_a_panel_cutoff_month": int(state.inner_panel.cutoff_month),
            "categorical_feature_order": list(CAT_FEATURES),
            "context_dim": CONTEXT_DIM,
            "panel_channels": list(PANEL_CHANNELS),
        }
        save_json(run_dir / "run_metadata.json", framing)

        for model_name in models:
            for seed in seeds:
                print(f"[{split_id}] {model_name} seed={seed} mode={args.mode} ...", flush=True)
                result = run_one(state, model_name, seed, args.mode, run_dir, command)
                results.append(result)
                print(f"    outer RMSE {result['overall']['rmse_sgd']:.2f} SGD | "
                      f"stage-A inner {result['stage_a_best_inner_rmse_sgd']:.2f} | "
                      f"epochs {result['stage_a_selected_epochs']} | {result['n_parameters']} params | "
                      f"{result['total_seconds']:.1f}s", flush=True)
                if args.verify_checkpoints:
                    check = verify_checkpoint(state, model_name, seed, run_dir)
                    checks.append({"split_id": split_id, "model": model_name, "seed": seed, **check})
                    print(f"    checkpoint reload: {check}", flush=True)

    summary = {**framing, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "results": results, "checkpoint_verification": checks}
    save_json(run_dir / "run_summary.json", summary)
    with open(REPORT / "commands.log", "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{run_id}] {command} (exit 0)\n")
    print(f"saved {run_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
