"""Combine completed TabPFN context experiments without rerunning the model.

Run after Phase B checkpoints have been scored on the outer validation period.
Missing experiments are omitted, so an interrupted run cannot acquire a score.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import DATA, REPORTS


ROOT = REPORTS / "tabpfn_context_sampling"
PHASE_A = ROOT / "phase_a_outer_budget10000_queriesfull_seed42"
FEATURE_ID = "calendar_80_20_categorical_77_residual"


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame.predicted.to_numpy(float) - frame.actual.to_numpy(float)
    return {"rmse_sgd": float(np.sqrt(np.mean(error**2))),
            "mae_sgd": float(np.mean(np.abs(error))),
            "bias_sgd": float(np.mean(error))}


def _check_predictions(frame: pd.DataFrame, *, expected_rows: int = 28_995) -> None:
    if len(frame) != expected_rows or not frame.row_id.is_unique:
        raise AssertionError("Validation prediction IDs are incomplete or repeated")
    if not np.isfinite(frame[["actual", "predicted"]].to_numpy(float)).all():
        raise AssertionError("Validation predictions contain nonfinite values")


def main() -> None:
    outer = pd.read_csv(PHASE_A / "scores.csv")
    original_training = json.loads((REPORTS / "tabpfn_diagscale_categorical_alignment_80_20"
                                    / "calibration.json").read_text())
    original_timing = json.loads((REPORTS / "tabpfn_diagscale_categorical_alignment_80_20"
                                  / "timing.json").read_text())
    original_ids = pd.read_csv(PHASE_A / "predictions_S0.csv").row_id.to_numpy()
    raw = pd.read_csv(DATA / "train.csv", usecols=["TOWN"])
    records: list[dict] = []
    group_records: list[dict] = []
    phase_b_s4_predictions: dict[str, pd.DataFrame] = {}
    phase_b_s4_matched8: dict[str, pd.DataFrame] = {}
    phase_b_s0_predictions: dict[str, pd.DataFrame] = {}
    phase_b_s0_matched8: dict[str, pd.DataFrame] = {}
    frozen_table = pd.read_csv(REPORTS / "tabpfn_diagscale_categorical_alignment_80_20/scores.csv")
    frozen = frozen_table.loc[frozen_table.model == "frozen"].iloc[0]
    frozen_prediction = pd.read_csv(REPORTS / "tabpfn_diagscale_categorical_alignment_80_20"
                                    / "predictions.csv")
    frozen_prediction = frozen_prediction[["row_id", "month", "actual", "frozen"]].rename(
        columns={"frozen": "predicted"})
    _check_predictions(frozen_prediction)
    np.testing.assert_array_equal(frozen_prediction.row_id.to_numpy(), original_ids)
    if abs(_metrics(frozen_prediction)["rmse_sgd"] - frozen.rmse_sgd) > 1e-4:
        raise AssertionError("Frozen baseline score does not match predictions")
    records.append({"run_id": "frozen_full_context", "phase": "reference",
                    "sampler_train": "frozen", "sampler_inference": "S0",
                    "feature_pipeline_id": FEATURE_ID, "train_views": 0,
                    "val_views": 0, "inference_views": 1,
                    "context_size": 121_005, "query_size": len(frozen_prediction),
                    "learning_rate": 0, "seed": 42,
                    "MAE": float(frozen.mae_sgd), "RMSE": float(frozen.rmse_sgd),
                    "bias_sgd": float(frozen.bias_sgd),
                    "train_time": 0, "retrieval_time": 0,
                    "peak_VRAM_GB": np.nan, "inference_seconds": np.nan,
                    "episode_support_rows": 0, "episode_query_rows": 0,
                    "episodes_per_epoch": 0, "actual_training_epochs": 0})
    group_records.extend(_groups(frozen_prediction, raw, "frozen_full_context"))
    for row in outer.to_dict("records"):
        prediction = pd.read_csv(PHASE_A / f"predictions_{row['sampler_inference']}.csv")
        _check_predictions(prediction)
        np.testing.assert_array_equal(prediction.row_id.to_numpy(), original_ids)
        calculated = _metrics(prediction)
        if abs(calculated["rmse_sgd"] - row["rmse_sgd"]) > 1e-4:
            raise AssertionError("Phase A saved score does not match predictions")
        records.append({**row, "phase": "A",
                        "train_time": original_training["fit_seconds"],
                        "retrieval_time": row["retrieval_seconds"],
                        "peak_VRAM_GB": (original_timing["outer_peak_vram_gib"]
                                         if row["sampler_inference"] == "S0"
                                         else row["peak_vram_gib"]),
                        "inference_seconds": (original_timing["outer_fit_predict_seconds"]
                                              if row["sampler_inference"] == "S0"
                                              else row["inference_seconds"]),
                        "episode_support_rows": 820, "episode_query_rows": 204,
                        "episodes_per_epoch": np.nan,
                        "actual_training_epochs": len(original_training["epoch_validation"]) - 1,
                        "adapter_sha256": original_training["adapter_sha256"],
                        "MAE": row["mae_sgd"], "RMSE": row["rmse_sgd"]})
        group_records.extend(_groups(prediction, raw, row["run_id"]))
    for run in sorted(ROOT.glob("phase_b_S*_seed*_episodes*_epochs*")):
        if run.name.endswith("_smoke") or not (run / "training.json").exists():
            continue
        training = json.loads((run / "training.json").read_text())
        settings = json.loads((run / "settings.json").read_text())
        for score_path in sorted(run.glob("outer_score_full_*.json")):
            score = json.loads(score_path.read_text())
            inference = score["sampler_inference"]
            prediction = pd.read_csv(run / f"outer_predictions_full_{inference}.csv")
            _check_predictions(prediction)
            np.testing.assert_array_equal(prediction.row_id.to_numpy(), original_ids)
            calculated = _metrics(prediction)
            if abs(calculated["rmse_sgd"] - score["rmse_sgd"]) > 1e-4:
                raise AssertionError("Phase B saved score does not match predictions")
            run_id = f"{run.name}_infer{inference}"
            records.append({
                "run_id": run_id, "phase": "B",
                "sampler_train": settings["strategy"],
                "sampler_inference": inference,
                "feature_pipeline_id": FEATURE_ID,
                "train_views": settings["train_views"],
                "val_views": settings["checkpoint_views"],
                "inference_views": settings["deployment_views"],
                "context_size": score["context_rows"],
                "query_size": score["query_rows"],
                "learning_rate": settings["learning_rate"],
                "seed": settings["seed"],
                "MAE": score["mae_sgd"], "RMSE": score["rmse_sgd"],
                "bias_sgd": score["bias_sgd"],
                "train_time": training["fit_seconds"],
                "retrieval_time": (training["mean_retrieval_seconds_per_episode"]
                                   * settings["episode_config"]["episodes_per_epoch"]
                                   * training["episode_epochs_built"]),
                "peak_VRAM_GB": max(training["peak_vram_gib"], score["peak_vram_gib"]),
                "inference_seconds": score["fit_predict_seconds"],
                "adapter_sha256": training["adapter_sha256"],
                "episode_support_rows": settings["episode_config"]["support_rows"],
                "episode_query_rows": settings["episode_config"]["query_rows"],
                "episodes_per_epoch": settings["episode_config"]["episodes_per_epoch"],
                "actual_training_epochs": training["episode_epochs_built"],
            })
            group_records.extend(_groups(prediction, raw, run_id))
            if (inference == "S4" and settings["seed"] == 42
                    and training["episode_epochs_built"] == 4):
                phase_b_s4_predictions[settings["strategy"]] = prediction
            if (inference == "S4" and settings["seed"] == 42
                    and training["episode_epochs_built"] == 8):
                phase_b_s4_matched8[settings["strategy"]] = prediction
            if (inference == "S0" and settings["seed"] == 42
                    and training["episode_epochs_built"] == 4):
                phase_b_s0_predictions[settings["strategy"]] = prediction
            if (inference == "S0" and settings["seed"] == 42
                    and training["episode_epochs_built"] == 8):
                phase_b_s0_matched8[settings["strategy"]] = prediction
    columns = [
        "run_id", "phase", "sampler_train", "sampler_inference", "feature_pipeline_id",
        "train_views", "val_views", "inference_views", "context_size", "query_size",
        "learning_rate", "seed", "MAE", "RMSE", "bias_sgd", "train_time",
        "retrieval_time", "peak_VRAM_GB", "inference_seconds", "episode_support_rows",
        "episode_query_rows", "episodes_per_epoch", "actual_training_epochs", "adapter_sha256",
    ]
    results = pd.DataFrame(records).reindex(columns=columns)
    results.to_csv(ROOT / "combined_outer_scores.csv", index=False)
    group_scores = pd.DataFrame(group_records)
    group_scores.to_csv(ROOT / "combined_group_scores.csv", index=False)
    _plot_group_comparisons(group_scores)
    if "S1" in phase_b_s4_predictions:
        _bootstrap_training_effect(phase_b_s4_predictions, inference="S4", epochs=4)
    if "S1" in phase_b_s4_matched8:
        _bootstrap_training_effect(phase_b_s4_matched8, inference="S4", epochs=8)
    if "S1" in phase_b_s0_predictions:
        _bootstrap_training_effect(phase_b_s0_predictions, inference="S0", epochs=4)
    if "S1" in phase_b_s0_matched8:
        _bootstrap_training_effect(phase_b_s0_matched8, inference="S0", epochs=8)
    fig, ax = plt.subplots(figsize=(max(9, len(results) * .62), 5))
    labels = []
    for row in results.itertuples():
        if row.phase == "B":
            prefix = f"{row.sampler_train}({int(row.actual_training_epochs)})"
        else:
            prefix = "Prior" if row.phase == "A" else "Frozen"
        labels.append(f"{prefix}/{row.sampler_inference}")
    colors = ["#59a14f" if phase == "reference" else
              "#4e79a7" if phase == "A" else "#e15759" for phase in results.phase]
    ax.bar(range(len(results)), results.RMSE, color=colors)
    ax.axhline(float(outer.loc[outer.sampler_inference == "S0", "rmse_sgd"].iloc[0]),
               color="black", ls="--", lw=1, label="original full context")
    ax.set_ylim(max(0, results.RMSE.min() - 15), results.RMSE.max() + 5)
    ax.set_ylabel("Outer validation RMSE (S$)")
    ax.set_xticks(range(len(results)), labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "combined_outer_rmse.png", dpi=160)
    plt.close(fig)
    print(results[["run_id", "MAE", "RMSE", "train_time", "peak_VRAM_GB"]].to_string(index=False))


def _groups(prediction: pd.DataFrame, raw: pd.DataFrame, run_id: str) -> list[dict]:
    frame = prediction.copy()
    frame["town"] = raw.TOWN.iloc[frame.row_id.to_numpy(int)].to_numpy()
    records = []
    for group in ("month", "town"):
        for value, subset in frame.groupby(group, observed=True):
            records.append({"run_id": run_id, "group": group, "value": value,
                            "rows": len(subset), **_metrics(subset)})
    return records


def _bootstrap_training_effect(predictions: dict[str, pd.DataFrame], *, inference: str,
                               epochs: int) -> None:
    """Resample validation months, retaining paired prediction rows."""
    reference = predictions["S1"]
    month = reference.month.to_numpy()
    months, inverse = np.unique(month, return_inverse=True)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(months), size=(20_000, len(months)))
    denominator = counts[draws].sum(axis=1)
    rmse_draws = {}
    for strategy, frame in predictions.items():
        np.testing.assert_array_equal(frame.row_id.to_numpy(), reference.row_id.to_numpy())
        np.testing.assert_array_equal(frame.month.to_numpy(), month)
        squared = (frame.predicted.to_numpy(float) - frame.actual.to_numpy(float)) ** 2
        sse = np.bincount(inverse, weights=squared, minlength=len(months))
        rmse_draws[strategy] = np.sqrt(sse[draws].sum(axis=1) / denominator)
    result = {}
    for strategy in sorted(set(predictions) - {"S1"}):
        difference = rmse_draws[strategy] - rmse_draws["S1"]
        result[f"{strategy}_minus_S1"] = {
            "median_sgd": float(np.median(difference)),
            "interval_95_sgd": [float(value) for value in np.quantile(difference, [.025, .975])],
        }
    result["design"] = (f"20,000 paired draws of 10 validation months; inference "
                        f"context {inference}; {epochs} training epochs per compared strategy")
    (ROOT / f"phase_b_month_cluster_bootstrap_infer{inference}_{epochs}epochs.json").write_text(
        json.dumps(result, indent=2) + "\n")


def _plot_group_comparisons(groups: pd.DataFrame) -> None:
    selections = {
        "Original full context": "outer_S0_budget10000_seed42",
        "Original adapter, mixed 10k": "outer_S4_budget10000_seed42",
        "Uniform support, 8 epochs, mixed 10k":
            "phase_b_S1_seed42_episodes100_epochs8_patience8_inferS4",
        "Mixed support, 8 epochs, mixed 10k":
            "phase_b_S4_seed42_episodes100_epochs8_inferS4",
    }
    monthly = groups.loc[groups.group == "month"].pivot(
        index="value", columns="run_id", values="rmse_sgd")
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, key in selections.items():
        if key in monthly:
            ax.plot(monthly.index, monthly[key], marker="o", label=label)
    ax.set_ylabel("Monthly RMSE (S$)")
    ax.set_xlabel("Validation month")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "combined_monthly_rmse.png", dpi=160)
    plt.close(fig)

    towns = groups.loc[groups.group == "town"].pivot(
        index="value", columns="run_id", values="rmse_sgd")
    comparison = selections["Mixed support, 8 epochs, mixed 10k"]
    reference = selections["Original full context"]
    if comparison in towns and reference in towns:
        delta = (towns[comparison] - towns[reference]).sort_values()
        fig, ax = plt.subplots(figsize=(9, 8))
        colors = np.where(delta.to_numpy() < 0, "#59a14f", "#e15759")
        ax.barh(delta.index, delta.to_numpy(), color=colors)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("RMSE difference from original full context (S$)")
        ax.set_title("Mixed support + mixed 10k inference by town")
        fig.tight_layout()
        fig.savefig(ROOT / "combined_town_rmse_difference.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
