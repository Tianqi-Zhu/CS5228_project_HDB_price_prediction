"""Compare searched architectures on the same 27 inputs and training rows.

Use existing inner trials restricted to the compact profile. Reuse an outer
fit only when its selected compact configuration exactly matches. Never
choose a configuration using its outer error.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analysis.experiments.feature_evidence import monthly_squared_errors, paired_month_interval
from analysis.experiments.feature_selection import SOURCES
from analysis.pipeline.config import DATA, DATE, EXPERIMENT_REPORTS, TARGET
from analysis.pipeline.data import check_output_path, load_main, protect_sources
from analysis.pipeline.evaluation import metrics
from analysis.pipeline.features import FEATURE_CANDIDATES, build_feature_frame, model_inputs


OUT = EXPERIMENT_REPORTS / "comparison_same_27_features"
SEARCH = EXPERIMENT_REPORTS / "comparison_model_search" / "2024-04"
ARCHIVED_SELECTIONS = Path(__file__).resolve().parents[3] / "experiment/appendix/X06/common_27_selection_records"
FAMILIES = ("lightgbm", "catboost", "xgboost", "hist_gradient_boosting", "extra_trees",
            "embedding_mlp", "embedding_resnet", "ft_transformer", "ft_transformer_default", "tabnet")


def choose_compact(trials):
    eligible = [t for t in trials if t["profile"] == "compact"]
    if not eligible:
        raise ValueError("No completed compact-profile inner trials")
    return min(eligible, key=lambda t: t["inner_scores"]["rmse_sgd"])


def selection_source(family):
    """Use local trials when present, otherwise the shipped frozen selections."""
    for root in (SEARCH, ARCHIVED_SELECTIONS):
        directory=root/family
        if (directory/'results.json').is_file() and any(directory.glob('trial_*/results.json')):
            return directory
    raise FileNotFoundError(f'Missing earlier-only selection for {family}; restore selection_records or run the original inner search.')


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def source_digests(hashes):
    return {str(path).replace("\\", "/").rsplit("/", 1)[-1]: digest for path, digest in hashes.items()}


def run(output_dir=OUT, device="cuda"):
    from analysis.experiments.model_search import NEURAL, fit_tree
    from analysis.pipeline.neural_models import fit_neural, predict_neural
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    spec = FEATURE_CANDIDATES["compact_history_27"]
    with protect_sources(SOURCES) as hashes:
        raw = load_main(DATA / "train.csv", require_target=True)
        frame = build_feature_frame(raw)
        fit = frame.loc[frame[DATE] < "2024-04"]
        val = frame.loc[(frame[DATE] >= "2024-04") & (frame[DATE] < "2025-04")]
        x_fit, x_val = model_inputs(fit, val, spec)
        actual = val[TARGET].to_numpy(float)
        stage = {"x_fit": x_fit, "x_val": x_val, "y_fit": fit[TARGET].to_numpy(float), "y_val": actual}
        rows, predictions = [], {}
        for family in FAMILIES:
            selected_source=selection_source(family)
            trials = [json.loads(p.read_text(encoding="utf-8"))
                      for p in selected_source.glob("trial_*/results.json")]
            selected = choose_compact(trials)
            old = json.loads((selected_source / "results.json").read_text(encoding="utf-8"))
            if source_digests(old["source_sha256"]) != source_digests(hashes):
                raise ValueError("Original search source hashes differ; rerun search first")
            directory = out / family
            directory.mkdir(parents=True, exist_ok=True)
            began = time.perf_counter()
            if (directory / "results.json").exists() and (directory / "predictions.csv").exists():
                record = json.loads((directory / "results.json").read_text(encoding="utf-8"))
                if record["selection"] != selected or source_digests(record["source_sha256"]) != source_digests(hashes):
                    raise ValueError("Completed common-feature run has a different configuration")
                saved = pd.read_csv(directory / "predictions.csv")
                if not np.array_equal(saved.source_row, val.index) or not np.allclose(saved.actual, actual):
                    raise ValueError("Cached prediction rows do not align")
                prediction = saved.prediction.to_numpy(float)
            elif (old["selection"] == selected and old["feature_spec"] == json.loads(json.dumps(asdict(spec)))
                  and (SEARCH / family / "predictions.csv").exists()):
                saved = pd.read_csv(SEARCH / family / "predictions.csv")
                if not np.array_equal(saved.source_row, val.index) or not np.allclose(saved.actual, actual):
                    raise ValueError("Reused prediction rows do not align")
                prediction = saved.prediction.to_numpy(float)
                record = {"family": family, "selection": selected,
                          "origin": "reused exact matching compact outer fit", "original_result": str((SEARCH / family / "results.json").relative_to(EXPERIMENT_REPORTS.parent.parent))}
            else:
                print(f"Common 27 inputs: refitting {family}, {selected['selected_iterations']} iterations/epochs", flush=True)
                if family in NEURAL:
                    bundle = fit_neural(family, x_fit, stage["y_fit"], spec.categorical,
                        selected["settings"], fixed_epochs=selected["selected_iterations"], device_name=device)
                    prediction = predict_neural(bundle, x_val)
                    del bundle
                    import torch
                    torch.cuda.empty_cache()
                else:
                    model, prediction, _ = fit_tree(family, stage, spec.categorical,
                        selected["settings"], fixed_iterations=selected["selected_iterations"])
                    del model
                record = {"family": family, "selection": selected,
                          "origin": "new outer refit using compact-only inner selection"}
            record.update({"scores": metrics(actual, prediction), "source_sha256": hashes,
                "n_features": len(spec.all_columns), "feature_spec": asdict(spec),
                "fitting_rows": len(fit), "validation_rows": len(val),
                "validation_start": "2024-04", "validation_end_exclusive": "2025-04",
                "processing_seconds": time.perf_counter() - began})
            pd.DataFrame({"source_row": val.index, "approval_month": val[DATE].to_numpy(),
                "actual": actual, "prediction": prediction}).to_csv(directory / "predictions.csv", index=False)
            write_json(directory / "results.json", record)
            rows.append({"family": family, "n_features": len(spec.all_columns), "n_fit": len(fit),
                "n_validation": len(val), "iterations": selected["selected_iterations"],
                "inner_rmse_sgd": selected["inner_scores"]["rmse_sgd"],
                "origin": record["origin"], **record["scores"]})
            predictions[family] = prediction
            pd.DataFrame(rows).to_csv(out / "model_metrics.csv", index=False)
            print(f"Common {family}: RMSE {record['scores']['rmse_sgd']:.2f}", flush=True)
            gc.collect()
        scores = pd.DataFrame(rows)
        sensitivity = []
        for family, prediction in predictions.items():
            monthly = monthly_squared_errors(val[DATE], actual, predictions["lightgbm"], prediction)
            sensitivity.append({"family": family,
                "delta_rmse_vs_lightgbm": float(scores.loc[scores.family.eq(family), "rmse_sgd"].iloc[0]
                    - scores.loc[scores.family.eq("lightgbm"), "rmse_sgd"].iloc[0]),
                **paired_month_interval(monthly)})
        scores = scores.merge(pd.DataFrame(sensitivity), on="family", validate="one_to_one").sort_values("rmse_sgd")
        scores.to_csv(out / "model_metrics.csv", index=False)
        write_json(out / "results.json", {"source_sha256": hashes, "feature_spec": asdict(spec),
            "fitting_rows": len(fit), "validation_rows": len(val), "validation_months": ["2024-04", "2025-03"],
            "selection": "Choose only compact inner trials (2023-10..2024-03) by their inner RMSE, then fix iterations for pre-2024-04 outer refit. No residual offsets or warmup row removal.",
            "metrics": scores.to_dict("records"), "limitations": [
                "Same inputs and rows, but model-specific preprocessing and unequal finite hyperparameter budgets remain.",
                "One seed per architecture and one outer window; not a universal architecture ranking.",
                "Previously examined labels; no new independent holdout.",
                "Paired 3-month block intervals reflect only fixed-prediction month sensitivity, without multiple-comparison correction."]})
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    return run(args.output_dir, args.device)


if __name__ == "__main__":
    main()
