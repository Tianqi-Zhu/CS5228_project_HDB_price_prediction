"""Run the best observed standalone TabPFN configuration in an isolated workspace.

Examples: python run_tabpfn_best.py preflight
          python run_tabpfn_best.py validate
          python run_tabpfn_best.py predict
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(".cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors")
CHECKPOINT_SHA = "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550"
BASE = Path(".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib")
ANN = Path(".cache/tabpfn_anniversary_history_80_20/anniversary87.joblib")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def packages():
    values = {}
    for name in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn", "joblib"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


def copy_verified(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f"Unexpected asset SHA256: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Existing asset differs; choose a new workspace: {target}")
    else:
        shutil.copyfile(source, target)


def rebuild_validation():
    """Build the historical 77 -> 87 recipe from raw CSVs, without old predictions."""
    import joblib
    import pandas as pd
    from analysis.config import DATA
    from analysis.prepare_tabpfn_v3_features import make_bundle
    from analysis.experiments.spatiotemporal_hybrid import MONTH, spatial_frame
    from analysis import compare_tabpfn_aligned_temporal_80_20 as aligned
    from analysis.compare_tabpfn_calendar_80_20 import variant_matrices
    from analysis import compare_tabpfn_anniversary_history_80_20 as anniversary

    if (ROOT / BASE).exists():
        bundle = anniversary.source_bundle()  # Verify the existing manifest first.
    else:
        print("Rebuilding validation features from raw CSVs", flush=True)
        frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
        fitting = frame.loc[frame[MONTH] < pd.Period("2024-06", "M")]
        query = frame.loc[frame[MONTH] >= pd.Period("2024-06", "M")]
        bundle, _ = make_bundle(fitting, query, "2024-06", has_query_labels=True)
        path = ROOT / ".cache/tabpfn_v3_features/best_raw_holdout.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path, compress=3)
        aligned.OUT.mkdir(parents=True, exist_ok=True)
        bundle, _ = aligned.prepare({"artifact": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
        x, q, audit = variant_matrices(bundle, "categorical", "categorical")
        bundle = {**bundle, "X_train": x, "X_query": q}
        joblib.dump(bundle, ROOT / BASE, compress=3)
        write_json(anniversary.BASE_MANIFEST, {**audit, "artifact": BASE.as_posix(),
                   "sha256": digest(ROOT / BASE), "rebuilt_from_raw": True,
                   "residual_target": "y_train - train_baseline",
                   "reconstruction": "predicted_residual + query_baseline"})
        bundle = anniversary.source_bundle()
    if not (ROOT / ANN).exists():
        anniversary.prepare()
    x, q = anniversary.matrices("anniversary87", bundle)
    if x.shape != (121005, 87) or q.shape != (28995, 87):
        raise ValueError("Unexpected best-model validation matrix shapes")
    return bundle


def worker(stage):
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.pop("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", None)
    if digest(ROOT / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Checkpoint does not match the historical Fast model")
    import torch
    if stage != "prepare" and not torch.cuda.is_available():
        raise RuntimeError("This full-context recipe requires a CUDA GPU")
    from analysis.config import DATA
    metadata = {"packages": packages(), "python": sys.version,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "raw_csv_sha256": {p.relative_to(DATA).as_posix(): digest(p)
                                   for p in sorted(DATA.rglob("*.csv"))
                                   if "processed" not in p.relative_to(DATA).parts},
                "helper_sha256": digest(Path(__file__)),
                "settings": {"features": 87, "estimators": 2, "seed": 42,
                             "batch_rows": 200, "subsampling": False,
                             "target": "rent minus temporal baseline"}}
    record = ROOT / "run-provenance" / f"{stage}.json"
    if record.exists() and json.loads(record.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Run environment changed; use a fresh --workspace")
    write_json(record, metadata)
    if stage in ("prepare", "validate"):
        base = rebuild_validation()
        if stage == "validate":
            from analysis.compare_tabpfn_anniversary_history_80_20 import infer, summarize, OUT
            infer("anniversary87", base)
            summarize()
            print((OUT / "scores.csv").read_text(encoding="utf-8"), flush=True)
    else:
        import numpy as np
        import pandas as pd
        from analysis.compare_zch_anniversary_tabpfn_ensemble import run_full_tabpfn, FULL_TABPFN
        (ROOT / ANN).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / "reports/tabpfn_fast_final_submission").mkdir(parents=True, exist_ok=True)
        completed = ROOT / "run-provenance/generated-full-prediction.json"
        if FULL_TABPFN.exists():
            if completed.exists():
                if digest(FULL_TABPFN) != json.loads(completed.read_text(encoding="utf-8"))["sha256"]:
                    raise ValueError("Generated full prediction changed")
            else:
                # prepare() includes archived reports. Their predictions are references,
                # not evidence that this workspace has performed inference.
                reference = ROOT / "archived-reference" / FULL_TABPFN.name
                reference.parent.mkdir(parents=True, exist_ok=True)
                if reference.exists():
                    raise ValueError("Reference already exists; choose a fresh workspace")
                FULL_TABPFN.rename(reference)
                print(f"Archived historical reference; fitting and inference will run afresh", flush=True)
        run_full_tabpfn()
        write_json(completed, {"sha256": digest(FULL_TABPFN), "generated_by_this_runner": True})
        prediction = pd.read_csv(FULL_TABPFN)
        sample = pd.read_csv(DATA / "example-submission.csv")
        if len(prediction) != 50000 or list(sample) != ["Id", "Predicted"]:
            raise ValueError("Unexpected prediction/submission schema")
        np.testing.assert_array_equal(prediction.row_id, sample.Id)
        values = prediction.predicted.to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Invalid submission predictions")
        sample["Predicted"] = values
        output = ROOT / "submission_tabpfn87_150k.csv"
        sample.to_csv(output, index=False, float_format="%.17g")
        write_json(ROOT / "run-provenance/submission.json", {
            "rows": len(sample), "features": 87, "context_rows": 150000,
            "prediction_sha256": digest(FULL_TABPFN), "submission_sha256": digest(output),
            "model": "standalone TabPFN-3.5-Fast; no tree ensemble"})
        print(f"Submission: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "prepare", "validate", "predict"))
    parser.add_argument("--workspace", type=Path, default=ROOT / "output/tabpfn-best")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT)
    parser.add_argument("--assets-from", type=Path,
                        help="Optional trusted local project: reuse only hash-verified feature caches, never predictions")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.stage)
        return 0
    import review_tabpfn
    bundle = review_tabpfn.load()
    verification = review_tabpfn.verify(bundle)
    if verification["failures"]:
        raise ValueError(verification)
    checkpoint = args.checkpoint.resolve()
    if args.stage == "preflight":
        try:
            import torch
        except ImportError:
            torch = None
        cuda = torch is not None and torch.cuda.is_available()
        print(json.dumps({"workflow_files_verified": verification["files_checked"],
            "packages": packages(), "cuda_available": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256_matches": checkpoint.is_file() and digest(checkpoint) == CHECKPOINT_SHA,
            "workspace": str(review_tabpfn.workspace_path(args.workspace)),
            "feature_policy": "Rebuild from raw CSVs unless --assets-from is explicitly supplied",
            "training_performed": False}, indent=2))
        return 0
    if not checkpoint.is_file():
        parser.error("Checkpoint missing. Supply --checkpoint PATH; see docs/guides/tabpfn_best.md")
    target = review_tabpfn.get_workspace(bundle, args.workspace)
    copy_verified(checkpoint, target / CHECKPOINT, CHECKPOINT_SHA)
    if args.assets_from:
        for relative, manifest in (
            (BASE, "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json"),
            (ANN, "tabpfn_anniversary_history_80_20/features_anniversary87.json"),
            (Path(".cache/tabpfn_fast_final_submission/features.joblib"), "tabpfn_fast_final_submission/feature_manifest.json"),
        ):
            full_asset = relative.name == "features.joblib"
            if full_asset != (args.stage == "predict"):
                continue
            source = args.assets_from.resolve() / relative
            if source.is_file():
                manifest_path = target / "reports" / manifest
                if not manifest_path.exists():
                    print(f"No archived hash for {relative}; rebuilding from raw data instead", flush=True)
                    continue
                expected = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
                destination = target / relative
                if destination.exists():
                    if digest(destination) != expected:
                        raise ValueError(f"Existing feature cache changed: {destination}")
                else:
                    copy_verified(source, destination, expected)
                print(f"Verified feature cache: {relative}", flush=True)
    helper = target / Path(__file__).name
    copy_verified(Path(__file__), helper, digest(Path(__file__)))
    return subprocess.call([sys.executable, "-u", str(helper), args.stage, "--worker"], cwd=target)


if __name__ == "__main__":
    raise SystemExit(main())
