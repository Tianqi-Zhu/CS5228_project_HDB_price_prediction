"""Blend the original zch anniversary ensemble with frozen anniversary TabPFN.

The TabPFN blend weight is fitted on October 2023-March 2024 predictions.
June 2024-March 2025 labels are used only after the weight is locked.
Run ``--stage screen`` on a free GPU, then ``--stage blend`` on CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.config import DATA, REPORTS, ROOT
from analysis.data import normalized_address, prepare_main, sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import FEATURES as TEMPORAL, HISTORY_COLUMNS, LocalTemporalModel, origin_history
from analysis.experiments.spatiotemporal_hybrid import TARGET, spatial_frame
from analysis.experiments.tabpfn_anniversary_history import AnniversaryHistory, FEATURES as ANNIVERSARY
from analysis.compare_tabpfn_anniversary_history_80_20 import CHECKPOINT, source_bundle


OUT = REPORTS / "zch_anniversary_tabpfn_blend"
ZCH = ROOT / ".cache/zch-reproduction"
SCREEN = OUT / "predictions_screen_tabpfn87.csv"
JUNE_TABPFN = REPORTS / "tabpfn_anniversary_history_80_20/predictions_anniversary87.csv"
FULL_TABPFN = OUT / "predictions_tabpfn87_full.csv"
FULL_ZCH = OUT / "predictions_zch_full.csv"
SUBMISSION = OUT / "submission_zch_tabpfn87_equal_150k.csv"
ZCH_WEIGHTS = {
    "normalized_market/compact_ratio15_market_last": 0.13094,
    "anniversary/broad_lgb15": 0.09659,
    "anniversary/broad_lgb63": 0.10352,
    "anniversary/exact_lgb15": 0.17311,
    "anniversary/exact_lgb63": 0.00658,
    "anniversary/window_cat6": 0.41380,
    "anniversary/window_lgb63": 0.07546,
}
ORIGIN = pd.Period("2023-10", freq="M")


def zch_path(name: str, stage: str) -> Path:
    group, member = name.split("/")
    return ZCH / "output/models/round_02_anniversary_history" / group / stage / member / "predictions.csv"


def joined(stage: str, tabpfn_path: Path) -> pd.DataFrame:
    tab = pd.read_csv(tabpfn_path).sort_values("row_id").reset_index(drop=True)
    required = {"row_id", "actual", "predicted"}
    if not required.issubset(tab):
        raise ValueError("TabPFN prediction columns are incomplete")
    if tab.row_id.duplicated().any() or not np.isfinite(tab[["actual", "predicted"]]).all().all():
        raise ValueError("Invalid TabPFN predictions")
    zch = np.zeros(len(tab))
    for name, weight in ZCH_WEIGHTS.items():
        part = pd.read_csv(zch_path(name, stage)).sort_values("source_row").reset_index(drop=True)
        np.testing.assert_array_equal(part.source_row, tab.row_id)
        np.testing.assert_array_equal(part.actual, tab.actual)
        zch += weight * part.prediction.to_numpy(float)
    tab["zch_original"] = zch
    tab = tab.rename(columns={"predicted": "tabpfn87"})
    return tab


def screen_features() -> dict:
    base = source_bundle()
    months = pd.PeriodIndex(base["train_month"], freq="M")
    fit = np.flatnonzero(months < ORIGIN)
    query = np.flatnonzero((months >= ORIGIN) & (months < pd.Period("2024-04", "M")))
    x, q = base["X_train"].iloc[fit].copy(), base["X_train"].iloc[query].copy()
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    past, blind = frame.loc[x.index], frame.loc[q.index].drop(columns=TARGET)
    q.loc[:, HISTORY_COLUMNS] = origin_history(past, blind).reindex(q.index).to_numpy()
    q.loc[:, TEMPORAL] = LocalTemporalModel(past).features(blind, ORIGIN - 1).reindex(q.index).to_numpy()
    for column in x.select_dtypes(include="category"):
        known = pd.Index(x[column].dropna().astype(str).unique())
        for matrix in (x, q):
            values = matrix[column].astype("string")
            matrix[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
    raw = normalized_address(prepare_main(pd.read_csv(DATA / "train.csv", dtype={"BLOCK": "string"})))
    np.testing.assert_array_equal(raw.loc[x.index, "MONTHLY_RENT"], np.asarray(base["y_train"])[fit])
    np.testing.assert_array_equal(raw.loc[q.index, "MONTHLY_RENT"], np.asarray(base["y_train"])[query])
    history = AnniversaryHistory().fit(raw.loc[x.index])
    x = pd.concat([x, history.transform(raw.loc[x.index])[ANNIVERSARY[:10]]], axis=1)
    q = pd.concat([q, history.transform(raw.loc[q.index].drop(columns="MONTHLY_RENT"))[ANNIVERSARY[:10]]], axis=1)
    if len(x.columns) != 87 or list(x) != list(q) or not np.isfinite(q.TEMP_LOCAL_FORECAST).all():
        raise AssertionError("Earlier-origin feature schema or baseline changed")
    if pd.PeriodIndex(np.asarray(base["train_month"])[fit], freq="M").max() >= ORIGIN:
        raise AssertionError("Earlier query labels entered the context")
    return {"x": x, "q": q, "y": np.asarray(base["y_train"])[fit],
            "actual": np.asarray(base["y_train"])[query],
            "train_baseline": np.asarray(base["train_baseline"])[fit],
            "query_baseline": q.TEMP_LOCAL_FORECAST.to_numpy(float),
            "month": np.asarray(base["train_month"])[query]}


def run_screen() -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    OUT.mkdir(parents=True, exist_ok=True)
    if SCREEN.exists():
        print(f"Reusing {SCREEN}", flush=True)
        return
    bundle = screen_features()
    x, q = bundle["x"], bundle["q"]
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    categorical = [i for i, c in enumerate(x) if isinstance(x[c].dtype, pd.CategoricalDtype)]
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda",
        n_estimators=2, random_state=42, fit_mode="fit_with_cache",
        keep_cache_on_device=True, categorical_features_indices=categorical,
        inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1,
    )
    partial = ROOT / ".cache/tabpfn_anniversary_history_80_20/screen_partial.npy"
    values = np.load(partial) if partial.exists() else np.full(len(q), np.nan)
    if values.shape != (len(q),):
        raise ValueError("Partial screen prediction shape differs")
    try:
        model.fit(x, bundle["y"] - bundle["train_baseline"])
        if model.n_train_samples_ != len(x) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("TabPFN context was changed")
        for start in range(0, len(q), 200):
            stop = min(start + 200, len(q))
            if np.isfinite(values[start:stop]).all():
                continue
            torch.cuda.empty_cache()
            values[start:stop] = np.asarray(model.predict(q.iloc[start:stop], output_type="mean"), float) + bundle["query_baseline"][start:stop]
            np.save(partial, values)
            if start % 2000 == 0 or stop == len(q):
                print(f"Earlier TabPFN: {stop:,}/{len(q):,}", flush=True)
        if not np.isfinite(values).all():
            raise AssertionError("Nonfinite earlier predictions")
        pd.DataFrame({"row_id": q.index, "month": bundle["month"], "actual": bundle["actual"],
                      "predicted": values}).to_csv(SCREEN, index=False, float_format="%.17g")
        (OUT / "screen_provenance.json").write_text(json.dumps({
            "origin": str(ORIGIN), "context_rows": len(x), "query_rows": len(q),
            "checkpoint_sha256": sha256(CHECKPOINT), "prediction_sha256": sha256(SCREEN),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "settings": "87 features, two frozen Fast views, full context, seed 42, native batching",
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def blend() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not np.isclose(sum(ZCH_WEIGHTS.values()), 1, atol=1e-12):
        raise AssertionError("Original zch weights do not sum to one")
    earlier = joined("screen", SCREEN)
    later = joined("june", JUNE_TABPFN)
    if not (pd.PeriodIndex(earlier.month, freq="M").max() < pd.PeriodIndex(later.month, freq="M").min()):
        raise AssertionError("Calibration and evaluation months overlap")
    delta = earlier.tabpfn87.to_numpy(float) - earlier.zch_original.to_numpy(float)
    error = earlier.zch_original.to_numpy(float) - earlier.actual.to_numpy(float)
    tab_weight = float(np.clip(-np.dot(error, delta) / np.dot(delta, delta), 0, 0.6))
    results = []
    for stage, frame in (("screen", earlier), ("june", later)):
        for name, weight in (("zch_original", 0.0), ("tabpfn87", 1.0), ("equal_blend", 0.5), ("earlier_locked_blend", tab_weight)):
            pred = (1 - weight) * frame.zch_original + weight * frame.tabpfn87
            results.append({"stage": stage, "model": name, "tabpfn_weight": weight,
                            "rows": len(frame), **metrics(frame.actual, pred)})
        frame = frame.copy()
        frame["locked_blend"] = (1 - tab_weight) * frame.zch_original + tab_weight * frame.tabpfn87
        frame.to_csv(OUT / f"predictions_{stage}.csv", index=False, float_format="%.17g")
    pd.DataFrame(results).to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly = []
    for month, part in later.groupby("month", sort=True):
        for name, prediction in (
            ("zch_original", part.zch_original),
            ("tabpfn87", part.tabpfn87),
            ("equal_blend", (part.zch_original + part.tabpfn87) / 2),
            ("earlier_locked_blend", (1 - tab_weight) * part.zch_original + tab_weight * part.tabpfn87),
        ):
            monthly.append({"month": month, "model": name, "rows": len(part),
                            **metrics(part.actual, prediction)})
    monthly_frame = pd.DataFrame(monthly)
    monthly_frame.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    sources = {name: {stage: sha256(zch_path(name, stage)) for stage in ("screen", "june")}
               for name in ZCH_WEIGHTS}
    (OUT / "weights.json").write_text(json.dumps({"original_zch_weights": ZCH_WEIGHTS,
        "tabpfn_weight": tab_weight, "weight_fit_months": "2023-10 through 2024-03",
        "evaluation_months": "2024-06 through 2025-03",
        "source_sha256": sources, "tabpfn_screen_sha256": sha256(SCREEN),
        "tabpfn_june_sha256": sha256(JUNE_TABPFN)}, indent=2) + "\n", encoding="utf-8")
    score = pd.DataFrame(results)
    june = score[score.stage == "june"].set_index("model")
    lines = ["# Original zch anniversary ensemble + anniversary TabPFN", "",
             "Seven zch round-02 tree members were reproduced from the branch source in an isolated worktree.",
             "Their published earlier-selected weights are fixed. A single TabPFN weight was fitted on October 2023-March 2024 labels, then evaluated on June 2024-March 2025.",
             "The original zch final 12-member grouped blend is a different model and is not reconstructed here.", "",
             "| Model | TabPFN weight | June RMSE (SGD) | June MAE (SGD) |",
             "|---|---:|---:|---:|"]
    for name in ("zch_original", "tabpfn87", "equal_blend", "earlier_locked_blend"):
        row = june.loc[name]
        lines.append(f"| {name} | {row.tabpfn_weight:.4f} | {row.rmse_sgd:.4f} | {row.mae_sgd:.4f} |")
    monthly_wide = monthly_frame.pivot(index="month", columns="model", values="rmse_sgd")
    wins_zch = int((monthly_wide.equal_blend < monthly_wide.zch_original).sum())
    wins_tab = int((monthly_wide.equal_blend < monthly_wide.tabpfn87).sum())
    lines += ["", f"The equal blend improves monthly RMSE versus zch in {wins_zch}/10 months and versus TabPFN in {wins_tab}/10 months."]
    lines += ["", "Scores use the same 28,995 validation row IDs and actual rents. This period has been used repeatedly for development; it is not an untouched test.",
              "The earlier-locked weight is the strictly earlier-selected option; the equal blend is a simple prespecified candidate that performed better on this development period.",
              "The full-data candidate is `submission_zch_tabpfn87_equal_150k.csv` when generated. Kaggle test accuracy is unknown.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(score.to_string(index=False), flush=True)


def full_features() -> dict:
    from analysis.train_tabpfn_fast_submission import prepare_features

    base = prepare_features()
    if base["X_train"].shape != (150000, 77) or base["X_query"].shape != (50000, 77):
        raise AssertionError("Full TabPFN base feature shape changed")
    train = normalized_address(prepare_main(pd.read_csv(DATA / "train.csv", dtype={"BLOCK": "string"})))
    test = normalized_address(prepare_main(pd.read_csv(DATA / "test.csv", dtype={"BLOCK": "string"})))
    np.testing.assert_array_equal(train.MONTHLY_RENT, base["y_train"])
    if not base["X_train"].index.equals(train.index) or not base["X_query"].index.equals(test.index):
        raise AssertionError("Full TabPFN row order differs from raw data")
    history = AnniversaryHistory().fit(train)
    added_train = history.transform(train)[ANNIVERSARY[:10]]
    added_test = history.transform(test)[ANNIVERSARY[:10]]
    x = pd.concat([base["X_train"], added_train], axis=1)
    q = pd.concat([base["X_query"], added_test], axis=1)
    if x.shape != (150000, 87) or q.shape != (50000, 87) or list(x) != list(q):
        raise AssertionError("Full anniversary feature schema changed")
    if not x.iloc[:, :77].equals(base["X_train"]) or not q.iloc[:, :77].equals(base["X_query"]):
        raise AssertionError("Original full-data TabPFN features changed")
    return {"x": x, "q": q, "y": np.asarray(base["y_train"], float),
            "train_baseline": np.asarray(base["train_baseline"], float),
            "query_baseline": np.asarray(base["query_baseline"], float)}


def run_full_tabpfn() -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    OUT.mkdir(parents=True, exist_ok=True)
    if FULL_TABPFN.exists():
        print(f"Reusing {FULL_TABPFN}", flush=True)
        return
    bundle = full_features()
    x, q = bundle["x"], bundle["q"]
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    categorical = [i for i, c in enumerate(x) if isinstance(x[c].dtype, pd.CategoricalDtype)]
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda",
        n_estimators=2, random_state=42, fit_mode="fit_with_cache",
        keep_cache_on_device=True, categorical_features_indices=categorical,
        inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1,
    )
    partial = ROOT / ".cache/tabpfn_anniversary_history_80_20/full_partial.npy"
    values = np.load(partial) if partial.exists() else np.full(len(q), np.nan)
    if values.shape != (len(q),):
        raise ValueError("Partial full prediction shape differs")
    try:
        model.fit(x, bundle["y"] - bundle["train_baseline"])
        if model.n_train_samples_ != 150000 or model.n_estimators_ != 2 or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("TabPFN full context or estimator count changed")
        for start in range(0, len(q), 200):
            stop = min(start + 200, len(q))
            if np.isfinite(values[start:stop]).all():
                continue
            torch.cuda.empty_cache()
            values[start:stop] = np.asarray(model.predict(q.iloc[start:stop], output_type="mean"), float) + bundle["query_baseline"][start:stop]
            np.save(partial, values)
            if start % 4000 == 0 or stop == len(q):
                print(f"Full TabPFN: {stop:,}/{len(q):,}", flush=True)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise AssertionError("Invalid full TabPFN predictions")
        pd.DataFrame({"row_id": q.index, "predicted": values}).to_csv(
            FULL_TABPFN, index=False, float_format="%.17g")
        (OUT / "tabpfn_full_provenance.json").write_text(json.dumps({
            "context_rows": len(x), "query_rows": len(q), "features": len(x.columns),
            "checkpoint_sha256": sha256(CHECKPOINT), "prediction_sha256": sha256(FULL_TABPFN),
            "source_train_sha256": sha256(DATA / "train.csv"),
            "source_test_sha256": sha256(DATA / "test.csv"),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "settings": "87 features, two frozen Fast views, full context, seed 42, native batching",
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def make_submission() -> None:
    zch = pd.read_csv(FULL_ZCH)
    tabpfn = pd.read_csv(FULL_TABPFN)
    np.testing.assert_array_equal(zch.row_id, np.arange(50000))
    np.testing.assert_array_equal(tabpfn.row_id, zch.row_id)
    expected_zch = json.loads((OUT / "zch_full_provenance.json").read_text(encoding="utf-8"))
    expected_tab = json.loads((OUT / "tabpfn_full_provenance.json").read_text(encoding="utf-8"))
    if sha256(FULL_ZCH) != expected_zch["prediction_sha256"] or sha256(FULL_TABPFN) != expected_tab["prediction_sha256"]:
        raise ValueError("Full prediction files changed")
    template = pd.read_csv(DATA / "example-submission.csv")
    if len(template) != 50000 or list(template) != ["Id", "Predicted"]:
        raise AssertionError("Unexpected competition submission template")
    predicted = (zch.predicted.to_numpy(float) + tabpfn.predicted.to_numpy(float)) / 2
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise AssertionError("Invalid ensemble rent")
    submission = pd.DataFrame({"Id": template.Id, "Predicted": predicted})
    submission.to_csv(SUBMISSION, index=False, float_format="%.6f")
    check = pd.read_csv(SUBMISSION)
    np.testing.assert_array_equal(check.Id, template.Id)
    np.testing.assert_allclose(check.Predicted, predicted, atol=5.1e-7, rtol=0)
    (OUT / "submission_provenance.json").write_text(json.dumps({
        "submission": SUBMISSION.relative_to(ROOT).as_posix(), "sha256": sha256(SUBMISSION),
        "rows": len(check), "weights": {"zch_original": 0.5, "tabpfn87": 0.5},
        "zch_prediction_sha256": sha256(FULL_ZCH), "tabpfn_prediction_sha256": sha256(FULL_TABPFN),
        "selection": "Prespecified equal blend; June 2024-March 2025 development RMSE 440.951958",
        "test_labels_used": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {SUBMISSION} ({len(check):,} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "blend", "full-tabpfn", "submission"), required=True)
    stage = parser.parse_args().stage
    if stage == "screen":
        run_screen()
    elif stage == "blend":
        blend()
    elif stage == "full-tabpfn":
        run_full_tabpfn()
    else:
        make_submission()


if __name__ == "__main__":
    main()
