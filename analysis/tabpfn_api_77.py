"""Fit and predict through Prior Labs using the saved original 77-feature inputs.

No feature rebuilding, local weight training, or DiagScale adaptation occurs.
Use --stage inspect offline, quote for a dimensions-only estimate, then run.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.evaluation import metrics

BUNDLES = {
    "validation": ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib",
    "submission": ROOT / ".cache/tabpfn_fast_final_submission/features.joblib",
}
TIME_COLUMN = "APPROVAL_MONTH_INDEX"
EXPORTS = ROOT / ".cache/tabpfn_api_77_inputs"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore_categories(frame, schema):
    for column, spec in schema.items():
        categories = np.asarray(spec["values"], dtype=spec["dtype"])
        frame[column] = pd.Categorical(np.asarray(frame[column].astype(object)),
                                       categories=categories, ordered=spec["ordered"])
    return frame


def load_inputs(path, split, train_rows=None, query_rows=None):
    """Keep typed columns, source row order, residual targets, and additive offsets."""
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["split"] != split:
            raise ValueError("Portable input split differs from requested split")
        source = Path(manifest["source"])
        if source.exists() and file_hash(source) != manifest["source_sha256"]:
            raise ValueError("Original feature bundle changed; export the inputs again")
        for name, digest in manifest["files"].items():
            if file_hash(path / name) != digest:
                raise ValueError(f"Portable input file changed: {name}")
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            b = dict(arrays)
        b.update(X_train=restore_categories(pd.read_parquet(path / "train.parquet"), manifest["categories"]),
                 X_query=restore_categories(pd.read_parquet(path / "query.parquet"), manifest["categories"]))
    else:
        b = joblib.load(path)
    x, q = b["X_train"], b["X_query"]
    if not isinstance(x, pd.DataFrame) or not isinstance(q, pd.DataFrame):
        raise ValueError("Expected typed pandas feature matrices")
    if len(x.columns) != 77 or list(x.columns) != list(q.columns) or not x.columns.is_unique:
        raise ValueError("Expected the same ordered 77 feature columns in train and query")
    if "row_id" in x.columns or not x.index.is_unique or not q.index.is_unique:
        raise ValueError("Source row IDs must be unique metadata, never a model input")
    if split == "validation" and len(x.index.intersection(q.index)):
        raise ValueError("Validation and training source IDs overlap")
    train_month, query_month = np.asarray(b["train_month"]), np.asarray(b["query_month"])
    y = np.asarray(b["y_train"], dtype=float)
    train_baseline = np.asarray(b["train_baseline"], dtype=float)
    baseline = np.asarray(b["query_baseline"], dtype=float)
    if any(a.shape != (len(x),) for a in [y, train_baseline, train_month]):
        raise ValueError("Training arrays are not aligned")
    if any(a.shape != (len(q),) for a in [baseline, query_month]):
        raise ValueError("Query arrays are not aligned")
    if not all(np.isfinite(a).all() for a in [y, train_baseline, baseline]):
        raise ValueError("Labels or additive baselines are nonfinite")
    if split == "validation" and max(train_month) >= min(query_month):
        raise ValueError("Validation must follow the training period")
    for c in x.select_dtypes(include="category"):
        if x[c].dtype != q[c].dtype:
            raise ValueError(f"Train/query categorical schema differs: {c}")
    for matrix in (x, q):
        if np.isinf(matrix.select_dtypes(include="number").to_numpy(float)).any():
            raise ValueError("Infinite features are invalid; intentional NaNs are allowed")
    n, m = min(train_rows or len(x), len(x)), min(query_rows or len(q), len(q))
    # Match the original ID order when selecting a smaller, explicitly requested pilot.
    actual = None
    if split == "validation":
        actual = np.asarray(b["y_query"], dtype=float)
        if actual.shape != (len(q),) or not np.isfinite(actual).all():
            raise ValueError("Validation labels are not aligned and finite")
        actual = actual[:m]
    return dict(x=x.iloc[:n].copy(), q=q.iloc[:m].copy(),
                target=(y - train_baseline)[:n], baseline=baseline[:m],
                actual=actual, month=query_month[:m])


def export_inputs(source, split):
    """Bridge pandas 3 pickles to the API client's pandas 2 using checked Parquet."""
    data = load_inputs(source, split)
    b = joblib.load(source)
    out = EXPORTS / split
    out.mkdir(parents=True, exist_ok=True)
    files = []
    categories = {c: dict(values=data["x"][c].cat.categories.tolist(),
                          dtype="int64" if data["x"][c].cat.categories.dtype.kind in "iu" else "object",
                          ordered=data["x"][c].cat.ordered)
                  for c in data["x"].select_dtypes(include="category")}
    for key, name in [("x", "train.parquet"), ("q", "query.parquet")]:
        frame = data[key].copy()
        # pandas 2 cannot deserialize pandas 3's new StringDtype in categories.
        # Keep categorical values, ordering and unused categories with object labels.
        frame = restore_categories(frame, categories)
        frame.to_parquet(out / name, engine="pyarrow", index=True)
        restored = restore_categories(pd.read_parquet(out / name), categories)
        pd.testing.assert_frame_equal(restored, frame, check_column_type=False, check_exact=True)
        files.append(name)
    arrays = {key: np.asarray(b[key]) for key in
              ["y_train", "train_baseline", "query_baseline"]}
    arrays.update({key: np.asarray(b[key], dtype=str) for key in ["train_month", "query_month"]})
    if split == "validation":
        arrays["y_query"] = np.asarray(b["y_query"])
    np.savez_compressed(out / "arrays.npz", **arrays)
    files.append("arrays.npz")
    write_json(out / "manifest.json", dict(split=split, source=str(source),
               source_sha256=file_hash(source), files={name: file_hash(out / name) for name in files},
               categories=categories, pandas_version=version("pandas"),
               context_rows=len(data["x"]), query_rows=len(data["q"])))
    print(f"Verified portable 77-feature inputs: {out}", flush=True)


def estimator_options(args):
    options = dict(model_path=args.model, n_estimators=args.n_estimators,
                   random_state=args.seed, fit_mode=args.fit_mode,
                   inference_config={"SUBSAMPLE_SAMPLES": None})
    if args.thinking_effort:
        if args.fit_mode == "fit_with_cache":
            raise ValueError("Thinking mode does not support fit_with_cache")
        options.update(thinking_effort=args.thinking_effort,
                       thinking_metric="mean_squared_error", time_col=TIME_COLUMN,
                       thinking_timeout_s=args.thinking_timeout)
    return options


def api_client(args):
    try:
        import tabpfn_client as client
    except ImportError as exc:
        raise RuntimeError("Install requirements-tabpfn-api.txt into your Python environment") from exc
    if args.token_file:
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("The supplied token file is empty")
        client.set_access_token(token)
    elif not os.environ.get("TABPFN_TOKEN"):
        # This project's existing helper stores its key here for weight downloads.
        cached = Path.home() / ".cache/tabpfn/auth_token"
        if cached.exists():
            client.set_access_token(cached.read_text(encoding="utf-8").strip())
    # Force a settings request even when set_access_token bypasses init checks.
    return client


def quote_run(client, data, args):
    """Quote each distinct batch size; never upload feature values here."""
    model_version = None if args.model in ("auto", "default") else args.model.removesuffix("_default")
    common = dict(model_version=model_version, n_estimators=args.n_estimators)
    result = {"quotes": [], "currency": "API quota units; see pricing_version, not SGD/USD"}
    if args.thinking_effort:
        fit = client.estimate_cost(data["x"], operation="thinking_fit",
                                   thinking_effort=args.thinking_effort, **common)
        result["thinking_fit"] = fit.model_dump(mode="json")
        result["thinking_prediction_note"] = (
            "Pre-fit prediction estimate only. The tuned model's actual ensemble count "
            "can differ from the requested count; re-quote using its resolved configuration.")
    operation = ("thinking_predict" if args.thinking_effort else
                 "cache_predict" if args.fit_mode == "fit_with_cache" else "predict")
    sizes = {}
    for start in range(0, len(data["q"]), args.batch_size):
        size = min(args.batch_size, len(data["q"]) - start)
        sizes[size] = sizes.get(size, 0) + 1
    for size, count in sizes.items():
        quote = client.estimate_cost(data["x"], data["q"].iloc[:size],
                                     operation=operation, **common)
        result["quotes"].append(dict(repeated_batches=count, **quote.model_dump(mode="json")))
    versions = {q["pricing_version"] for q in result["quotes"]}
    if len(versions) == 1:
        result["estimated_prediction_total"] = sum(q["estimated_cost"] * q["repeated_batches"]
                                                   for q in result["quotes"])
    return result


def fit_model(client, data, options, cache):
    path = cache / "model.json"
    if path.exists():
        model = client.TabPFNRegressor.load_model(path)
        print("Reusing saved API fit (including any pending job)", flush=True)
    else:
        model = client.TabPFNRegressor(**options)
        model.submit_fit(data["x"], data["target"])
        model.save_model(path)
        print("API fit submitted and saved", flush=True)
    try:
        model.wait_for_fit()
    finally:
        model.save_model(path)
    print("API fit ready", flush=True)
    return model


def predict_batches(client, model, data, cache, batch_size):
    """Persist each accepted prediction job before polling, and each finished batch."""
    progress_path, pending_path = cache / "progress.json", cache / "pending_prediction.json"
    values_path = cache / "predictions.npy"
    total = len(data["q"])
    if progress_path.exists():
        start = json.loads(progress_path.read_text(encoding="utf-8"))["next_row"]
        values = np.load(values_path, allow_pickle=False)
        if values.shape != (total,) or not 0 <= start <= total or not np.isfinite(values[:start]).all():
            raise ValueError("Saved prediction progress is corrupt")
    else:
        values, start = np.full(total, np.nan), 0
    for offset in range(start, total, batch_size):
        size = min(batch_size, total - offset)
        pending = json.loads(pending_path.read_text(encoding="utf-8")) if pending_path.exists() else None
        if pending and pending["offset"] < offset:
            pending_path.unlink()  # Batch was already committed before an interruption.
            pending = None
        if pending:
            if (pending["offset"], pending["size"]) != (offset, size):
                raise ValueError("Pending prediction belongs to a different batch")
            model = client.TabPFNRegressor.load_model(pending["model"])
        else:
            model.submit_predict(data["q"].iloc[offset:offset + size], output_type="mean")
            write_json(pending_path, dict(offset=offset, size=size, model=model.save_model()))
        residual = np.asarray(model.wait_for_predict(), dtype=float)
        if residual.shape != (size,) or not np.isfinite(residual).all():
            raise ValueError(f"Invalid API predictions at offset {offset}")
        values[offset:offset + size] = residual + data["baseline"][offset:offset + size]
        temporary = cache / "predictions.tmp.npy"
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(values_path)
        write_json(progress_path, {"next_row": offset + size})
        pending_path.unlink()
        write_json(cache / "last_timings.json", model.get_timings())
        print(f"Predicted {offset + size:,}/{total:,} rows", flush=True)
    return values


def save_results(data, values, args, out):
    if values.shape != (len(data["q"]),) or not np.isfinite(values).all():
        raise ValueError("Predictions are incomplete or nonfinite")
    frame = pd.DataFrame(dict(row_id=data["q"].index, month=data["month"], predicted=values))
    if data["actual"] is not None:
        frame["actual"] = data["actual"]
        score = metrics(data["actual"], values)
        write_json(out / "metrics.json", score)
        monthly = [dict(month=str(month), n=len(part), **metrics(part.actual, part.predicted))
                   for month, part in frame.groupby("month", sort=True)]
        pd.DataFrame(monthly).to_csv(out / "monthly_metrics.csv", index=False)
        print(json.dumps(score), flush=True)
    elif args.query_rows is None:
        template = pd.read_csv(ROOT / "data/example-submission.csv")
        if list(template) != ["Id", "Predicted"] or len(template) != len(values):
            raise ValueError("Submission template differs from full query data")
        if not np.array_equal(data["q"].index.to_numpy(), np.arange(len(template))):
            raise ValueError("Submission query order differs from source CSV row order")
        template["Predicted"] = values
        template.to_csv(out / "submission_tabpfn_api.csv", index=False, float_format="%.17g")
    frame.to_csv(out / "predictions.csv", index=False, float_format="%.17g")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["export", "inspect", "models", "quote", "fit", "predict", "run"], default="inspect")
    p.add_argument("--split", choices=list(BUNDLES), default="validation")
    p.add_argument("--bundle", type=Path, help="Override saved typed input bundle")
    p.add_argument("--model", default="v3.5_default", help="API model_path, or auto for latest server default")
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fit-mode", choices=["fit_preprocessors", "fit_with_cache"], default="fit_preprocessors")
    p.add_argument("--thinking-effort", choices=["medium", "high"])
    p.add_argument("--thinking-timeout", type=float, default=600)
    p.add_argument("--train-rows", type=int, help="Explicit pilot: use first N context rows in saved order")
    p.add_argument("--query-rows", type=int, help="Explicit pilot: use first N query rows in saved order")
    p.add_argument("--run-name", help="Separate immutable experiment directory")
    p.add_argument("--token-file", type=Path, help="Read existing API key privately; never copied to outputs")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.n_estimators <= 8 or args.batch_size <= 0:
        raise ValueError("Use 1-8 estimators and a positive batch size")
    if any(n is not None and n <= 0 for n in [args.train_rows, args.query_rows]):
        raise ValueError("Pilot row limits must be positive")
    if not 0 < args.thinking_timeout <= 2400:
        raise ValueError("Thinking timeout must be between 0 and 2400 seconds")
    if args.stage == "models":
        client = api_client(args)
        print(json.dumps(client.TabPFNRegressor.list_available_models(), indent=2))
        print("These are client-supported names; account access is checked by the server.")
        return
    options = estimator_options(args)
    if args.stage == "export":
        export_inputs((args.bundle or BUNDLES[args.split]).resolve(), args.split)
        return
    portable = EXPORTS / args.split
    bundle_path = (args.bundle or (portable if portable.exists() else BUNDLES[args.split])).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Prepare the original 77-feature bundle first: {bundle_path}")
    data = load_inputs(bundle_path, args.split, args.train_rows, args.query_rows)
    options["categorical_features_indices"] = [int(data["x"].columns.get_loc(c))
                                               for c in data["x"].select_dtypes(include="category")]
    digest = file_hash(bundle_path / "manifest.json" if bundle_path.is_dir() else bundle_path)
    summary = dict(split=args.split, bundle=str(bundle_path), bundle_sha256=digest,
                   context_rows=len(data["x"]), query_rows=len(data["q"]),
                   features=list(data["x"]), categorical_columns=list(data["x"].select_dtypes(include="category")),
                   target="rent minus train_baseline; add query_baseline once after prediction",
                   estimator=options, batch_size=args.batch_size,
                   train_rows_limit=args.train_rows, query_rows_limit=args.query_rows)
    if args.stage == "inspect":
        print(json.dumps(summary, indent=2))
        return
    name = args.run_name or f"{args.split}_{args.model}_{args.thinking_effort or 'standard'}_e{args.n_estimators}"
    if name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name):
        raise ValueError("Run name must be a simple directory name")
    out, cache = ROOT / "reports/tabpfn_api_77" / name, ROOT / ".cache/tabpfn_api_77" / name
    client = api_client(args)
    if args.stage == "quote":
        quote = quote_run(client, data, args)
        write_json(out / "quote.json", dict(inputs=summary, **quote))
        print(json.dumps(quote, indent=2))
        return
    summary["client_version"] = version("tabpfn-client")
    summary["runner_sha256"] = file_hash(Path(__file__))
    settings_path = cache / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != summary:
        raise ValueError("Inputs/settings changed; use a new --run-name")
    write_json(settings_path, summary)
    write_json(out / "settings.json", summary)
    if args.stage == "predict":
        model = client.TabPFNRegressor.load_model(cache / "model.json")
        model.wait_for_fit()
    else:
        model = fit_model(client, data, options, cache)
    if args.stage != "fit":
        values = predict_batches(client, model, data, cache, args.batch_size)
        save_results(data, values, args, out)
    print(f"Outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
