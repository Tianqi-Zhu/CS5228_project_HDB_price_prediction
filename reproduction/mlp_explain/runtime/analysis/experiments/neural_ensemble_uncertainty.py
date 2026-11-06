"""Independent MLP/TabM ensembles with a separate chronological calibration set.

Run with ``python -m analysis.experiments.neural_ensemble_uncertainty --output-dir ...``.
Intervals concern individual rents, not confidence intervals for the mean.
Temporal dependence/shift prevents an unconditional exchangeability guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def ensemble_summary(member_predictions):
    """Population disagreement across fitted networks; never divide by sqrt(M)."""
    values = np.asarray(member_predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite rows x at-least-two independent models")
    return values.mean(axis=1), values.std(axis=1, ddof=0)


def conformal_quantile(scores, alpha):
    """Finite-sample rank correction, with infinity if calibration is too small."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Calibration scores must be finite nonnegative values")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else float("inf")


class RentIntervalCalibrator:
    """Absolute residual or fixed-floor disagreement-normalized split calibration."""
    def __init__(self, floor_sgd=100., normalized=True):
        if not np.isfinite(floor_sgd) or floor_sgd <= 0:
            raise ValueError("Positive finite uncertainty floor required")
        self.floor_sgd = float(floor_sgd)
        self.normalized = bool(normalized)
        self.quantiles = {}

    def scale(self, std):
        std = np.asarray(std, dtype=float)
        if not np.isfinite(std).all() or (std < 0).any():
            raise ValueError("Nonnegative finite disagreement required")
        return np.maximum(std, self.floor_sgd) if self.normalized else np.ones_like(std)

    def fit(self, actual, prediction, std, levels=(.90, .95)):
        actual, prediction, std = [np.asarray(x, dtype=float) for x in (actual, prediction, std)]
        if actual.ndim != 1 or actual.shape != prediction.shape or actual.shape != std.shape:
            raise ValueError("Calibration actual, prediction and std must align")
        scores = np.abs(actual - prediction) / self.scale(std)
        self.quantiles = {str(int(round(100 * level))): conformal_quantile(scores, 1 - level) for level in levels}
        self.n_calibration = len(scores)
        return self

    def intervals(self, prediction, std):
        if not self.quantiles:
            raise ValueError("Fit calibration before inference")
        prediction, std = [np.asarray(x, dtype=float) for x in (prediction, std)]
        if prediction.ndim != 1 or prediction.shape != std.shape or not np.isfinite(prediction).all():
            raise ValueError("Prediction and disagreement must align and be finite")
        return {level: (prediction - q * self.scale(std), prediction + q * self.scale(std))
                for level, q in self.quantiles.items()}

    def metadata(self):
        return {"normalized": self.normalized, "floor_sgd": self.floor_sgd,
                "quantiles": self.quantiles, "n_calibration": self.n_calibration}


def partition_masks(months):
    dates = pd.PeriodIndex(months, freq="M")
    return {"selection_fit": dates < "2023-10",
            "selection_tune": (dates >= "2023-10") & (dates < "2024-01"),
            "final_fit": dates < "2024-01",
            "calibration": (dates >= "2024-01") & (dates < "2024-04"),
            "outer": dates >= "2024-04"}


def interval_metrics(actual, prediction, std, calibrator):
    error = np.asarray(prediction) - np.asarray(actual)
    result = {"n": len(error), "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
              "mae_sgd": float(np.mean(np.abs(error))), "bias_sgd": float(error.mean()),
              "mean_ensemble_std_sgd": float(np.mean(std))}
    for level, (lower, upper) in calibrator.intervals(prediction, std).items():
        alpha = 1 - int(level) / 100
        result[f"coverage_{level}"] = float(np.mean((actual >= lower) & (actual <= upper)))
        result[f"mean_width_{level}_sgd"] = float(np.mean(upper - lower))
        score = upper - lower + 2 / alpha * (np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))
        result[f"interval_score_{level}_sgd"] = float(np.mean(score))
    return result


def export_predictions(meta, values, calibrator):
    mean, std = ensemble_summary(values)
    result = meta.reset_index(drop=True).copy()
    result["prediction_sgd"] = mean
    result["ensemble_std_sgd"] = std
    result["interval_scale_sgd"] = calibrator.scale(std)
    for level, (lower, upper) in calibrator.intervals(mean, std).items():
        result[f"pi{level}_lower_sgd"] = lower
        result[f"pi{level}_upper_sgd"] = upper
    for i in range(values.shape[1]):
        result[f"member_{i + 1}_sgd"] = values[:, i]
    return result


def _json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    import joblib
    import torch
    from analysis.pipeline.config import DATA, ROOT
    from analysis.pipeline.data import check_output_path, sha256
    from analysis.pipeline.auxiliary import build_auxiliary_features
    from analysis.experiments.embedding_rent_search import fit_model
    from analysis.experiments.tabm_rent_search import fit as fit_tabm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", choices=["both", "mlp", "tabm"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-selection-epochs", type=int, default=35)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("At least two different seeds are required")
    if args.max_selection_epochs < 1:
        parser.error("Epoch budget must be positive")
    out = check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error("Use a new output directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    specs = {
        "mlp": dict(name="residual_mlp", architecture="mlp", blocks=2, width=128,
                    history=False, adjusted_history=True, residual_reference=True,
                    quantile=True, lr=.0005, aux_weight=.1, max_epochs=args.max_selection_epochs),
        "tabm": dict(name="piecewise_tabm", encoding="quantile", history=True, quantile=True,
                     k=16, width=128, blocks=2, lr=.001),
    }
    if args.families != "both":
        specs = {args.families: specs[args.families]}
    source_paths = sorted((DATA / "auxiliary").glob("*.csv")) + [DATA / "train.csv", DATA / "test.csv"]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    code_paths = [Path(__file__), Path(__file__).resolve().parents[2] / "analysis/experiments/embedding_rent_search.py",
                  Path(__file__).resolve().parents[2] / "analysis/experiments/tabm_rent_search.py", Path(__file__).resolve().parents[2] / "analysis/experiments/deep_learning.py",
                  Path(__file__).resolve().parents[2] / "analysis/experiments/adjusted_rent_history.py", Path(__file__).resolve().parents[2] / "analysis/experiments/spatiotemporal_hybrid.py",
                  Path(__file__).resolve().parents[2] / "analysis/pipeline/auxiliary.py", Path(__file__).resolve().parents[2] / "analysis/pipeline/data.py"]
    settings = {"specs": specs, "seeds": args.seeds, "predict_test": args.predict_test,
                "max_selection_epochs": args.max_selection_epochs, "source_sha256": source_hashes,
                "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
                "versions": {p: __import__('importlib.metadata', fromlist=['version']).version(p)
                             for p in ('torch', 'numpy', 'pandas', 'scikit-learn', 'tabm', 'rtdl-num-embeddings')},
                "selection_fit": "2021-01..2023-09", "selection_tune": "2023-10..2023-12",
                "final_fit": "2021-01..2023-12", "calibration": "2024-01..2024-03",
                "outer": "2024-04..2025-03", "history_origin": "2023-12",
                "aggregation": "equal independent seeds; hybrid is equal family means; no calibration-based model selection",
                "primary_interval": "abs residual / max(independent-network std, 100 SGD); finite-sample rank quantile",
                "reference_interval": "constant-width absolute residual calibration; descriptive control, no method selection",
                "limits": ["Time-series dependence and drift: nominal coverage is not guaranteed.",
                           "No posterior probability or confidence-of-correctness per row.",
                           "Disagreement is not total predictive uncertainty or standard error of the mean.",
                           "Historical architectures and outer period previously inspected: exploratory development.",
                           "Models retain December-2023 fitting origin after calibration; no post-calibration refit.",
                           "Static facility snapshots inherited; not fully historical point-in-time geography.",
                           "Long-horizon test intervals are extrapolations, not validated test accuracy."]}
    if args.resume:
        if not (out / "protocol.json").exists() or json.loads((out / "protocol.json").read_text()) != settings:
            parser.error("Resume source/code/settings differ")
    _json(out / "protocol.json", settings)
    for path in code_paths:
        destination = out / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    frame = build_auxiliary_features(pd.read_csv(DATA / "train.csv"))
    for column in ("TOWN", "STREET", "BLOCK"):
        frame[column] = frame[column].astype(str).str.casefold().str.strip()
    frame["STREET_ID"] = frame.TOWN + "|" + frame.STREET
    frame["BUILDING_ID"] = frame.STREET_ID + "|" + frame.BLOCK
    masks = partition_masks(frame.RENT_APPROVAL_DATE)
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"Empty {name} partition")
    if masks["outer"].sum() != 35158:
        raise ValueError("Expected unchanged 35,158 outer rows")
    _json(out / "partition_counts.json", {name: int(mask.sum()) for name, mask in masks.items()})
    future = frame.loc[masks["calibration"] | masks["outer"]].copy()
    query_meta = pd.DataFrame({"source_row": future.index, "month": future.RENT_APPROVAL_DATE,
                               "partition": np.where(future.RENT_APPROVAL_DATE < "2024-04", "calibration", "outer"),
                               "flat_type": future.FLAT_TYPE_CLEAN})
    if args.predict_test:
        raw_test = pd.read_csv(DATA / "test.csv")
        test = build_auxiliary_features(raw_test)
        for column in ("TOWN", "STREET", "BLOCK"):
            test[column] = test[column].astype(str).str.casefold().str.strip()
        test["STREET_ID"] = test.TOWN + "|" + test.STREET
        test["BUILDING_ID"] = test.STREET_ID + "|" + test.BLOCK
        sample = pd.read_csv(DATA / "example-submission.csv")
        if len(test) != len(sample):
            raise ValueError("Sample submission differs from test length")
        test.index = np.arange(len(frame), len(frame) + len(test))
        test_meta = pd.DataFrame({"source_row": test.index, "month": test.RENT_APPROVAL_DATE,
                                  "partition": "test", "flat_type": test.FLAT_TYPE_CLEAN,
                                  "Id": sample.Id.to_numpy()})
        query_meta = pd.concat([query_meta, test_meta], ignore_index=True)
        future = pd.concat([future, test])
    query_meta = query_meta.reset_index(drop=True)
    query_meta["months_after_fit"] = pd.PeriodIndex(query_meta.month, freq="M").asi8 - pd.Period("2023-12", "M").ordinal
    # Even calibration and outer labels are absent from every fixed-epoch inference call.
    blind = future.drop(columns="MONTHLY_RENT", errors="ignore")
    selection_fit, selection_tune = frame.loc[masks["selection_fit"]], frame.loc[masks["selection_tune"]]
    final_fit = frame.loc[masks["final_fit"]]
    predictions = {}
    for family, spec in specs.items():
        selected_file = out / f"{family}_selected.json"
        if selected_file.exists() and args.resume:
            selected = json.loads(selected_file.read_text())
        else:
            print(f"Selecting {family} epochs using October-December 2023 only", flush=True)
            if family == "mlp":
                _, _, epoch, best, trace, _, _ = fit_model(selection_fit, selection_tune, dict(spec, seed=args.seeds[0]))
            else:
                # Use only this runner's declared selection epoch cap.
                _, _, epoch, best, trace, _ = fit_tabm(selection_fit, selection_tune, dict(spec, seed=args.seeds[0], max_epochs=args.max_selection_epochs))
            selected = {"epochs": epoch, "selection_rmse_sgd": best, "selection_seed": args.seeds[0]}
            _json(selected_file, selected)
            pd.DataFrame(trace).to_csv(out / f"{family}_selection_trace.csv", index=False)
        parts = []
        for seed in args.seeds:
            run = out / f"{family}_seed{seed}"
            run.mkdir(exist_ok=True)
            saved_predictions = run / "predictions.csv"
            completion = run / "complete.json"
            if args.resume and completion.exists():
                complete = json.loads(completion.read_text())
                if sha256(saved_predictions) != complete['predictions_sha256']:
                    raise ValueError("Completed predictions changed")
                saved = pd.read_csv(saved_predictions)
                np.testing.assert_array_equal(saved.source_row, query_meta.source_row)
                part = saved.prediction_sgd.to_numpy()
            else:
                print(f"Fitting {family} seed {seed}, fixed {selected['epochs']} epochs, then blind prediction of {len(blind):,} rows", flush=True)
                config = dict(spec, seed=seed)
                if family == "mlp":
                    model, meta, _, _, _, part, _ = fit_model(final_fit, blind, config, epochs=selected['epochs'])
                else:
                    model, meta, _, _, _, part = fit_tabm(final_fit, blind, config, epochs=selected['epochs'])
                if len(part) != len(query_meta) or not np.isfinite(part).all():
                    raise ValueError("Nonfinite or incomplete model predictions")
                pd.DataFrame({'source_row': query_meta.source_row, 'prediction_sgd': part}).to_csv(saved_predictions, index=False)
                torch.save(model.state_dict(), run / "model.pt")
                joblib.dump(meta, run / "preprocessor.joblib")
                _json(completion, {"predictions_sha256": sha256(saved_predictions), "seed": seed, "epochs": selected['epochs']})
                print(f"Completed {family} seed {seed}", flush=True)
            parts.append(part)
        predictions[family] = np.column_stack(parts)
    if len(predictions) == 2:
        predictions['hybrid'] = np.column_stack([predictions['mlp'], predictions['tabm']])

    cal_mask = query_meta.partition.eq('calibration').to_numpy()
    outer_mask = query_meta.partition.eq('outer').to_numpy()
    # Calibration labels are read only after epoch counts and all seed models are frozen.
    calibration_y = frame.loc[query_meta.loc[cal_mask, 'source_row'], 'MONTHLY_RENT'].to_numpy()
    calibrators, references = {}, {}
    for name, values in predictions.items():
        mean, std = ensemble_summary(values)
        calibrators[name] = RentIntervalCalibrator().fit(calibration_y, mean[cal_mask], std[cal_mask])
        references[name] = RentIntervalCalibrator(normalized=False).fit(calibration_y, mean[cal_mask], std[cal_mask])
    _json(out / "frozen_calibration.json", {name: {'primary': value.metadata(), 'constant_width': references[name].metadata()}
                                             for name, value in calibrators.items()})
    # Do not choose architecture, seeds, weights, interval level or scale using outer labels.
    outer_y = frame.loc[query_meta.loc[outer_mask, 'source_row'], 'MONTHLY_RENT'].to_numpy()
    scores, monthly, groups = [], [], []
    for name, values in predictions.items():
        exported = export_predictions(query_meta, values, calibrators[name])
        exported['actual_sgd'] = np.nan
        exported.loc[cal_mask, 'actual_sgd'] = calibration_y
        exported.loc[outer_mask, 'actual_sgd'] = outer_y
        for partition in query_meta.partition.unique():
            piece = exported.loc[exported.partition.eq(partition)].copy()
            if partition == 'test':
                piece = piece.drop(columns='actual_sgd')
            piece.to_csv(out / f"{name}_{partition}_predictions.csv", index=False)
        mean, std = ensemble_summary(values)
        for method, calibrator in [('normalized', calibrators[name]), ('constant_width', references[name])]:
            scores.append({'model': name, 'interval_method': method, **interval_metrics(outer_y, mean[outer_mask], std[outer_mask], calibrator)})
        outer = exported.loc[outer_mask]
        for column, target in [('month', monthly), ('flat_type', groups)]:
            for group, piece in outer.groupby(column, observed=True):
                target.append({'model': name, column: group, **interval_metrics(piece.actual_sgd.to_numpy(), piece.prediction_sgd.to_numpy(), piece.ensemble_std_sgd.to_numpy(), calibrators[name])})
    pd.DataFrame(scores).to_csv(out / "outer_metrics.csv", index=False)
    pd.DataFrame(monthly).to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(groups).to_csv(out / "flat_type_metrics.csv", index=False)
    for name, digest in source_hashes.items():
        if sha256(ROOT / name) != digest:
            raise AssertionError(f"Source changed: {name}")
    _json(out / 'verification.json', {'outer_rows': len(outer_y), 'calibration_rows': len(calibration_y),
                                    'raw_sources_unchanged': True, 'query_targets_removed_before_inference': True,
                                    'outer_ids_match_source': True, 'test_labels_available': False})
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
