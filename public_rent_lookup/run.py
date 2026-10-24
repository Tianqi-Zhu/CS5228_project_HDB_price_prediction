"""Reconstruct rents from public HDB records in an isolated experiment.

No model is trained. Known-label errors measure source reconstruction, not
past-only forecasting. This script writes local files and makes no network calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
KEYS = ["month", "town", "block", "street", "flat_type"]
TARGET = "MONTHLY_RENT"
SOURCE_PAGE = "https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view"
STREET_WORDS = {
    "AVENUE": "AVE", "ROAD": "RD", "STREET": "ST", "DRIVE": "DR",
    "CRESCENT": "CRES", "CLOSE": "CL", "PLACE": "PL", "BOULEVARD": "BLVD",
    "CENTRAL": "CTRL", "JALAN": "JLN", "BUKIT": "BT", "UPPER": "UPP",
    "TANJONG": "TG", "KAMPONG": "KG", "LORONG": "LOR", "SAINT": "ST",
    "TERRACE": "TER", "MOUNT": "MT", "NORTH": "NTH", "SOUTH": "STH",
    "GARDENS": "GDNS", "HEIGHTS": "HTS", "PARK": "PK",
    "COMMONWEALTH": "CWEALTH", "MARKET": "MKT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Keep published summaries independent of the author's machine path."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def normalize_text(values: pd.Series) -> pd.Series:
    return (values.astype("string").fillna("").str.upper().str.strip()
            .str.replace(r"\s+", " ", regex=True))


def canonical_keys(frame: pd.DataFrame, *, public: bool = False) -> pd.DataFrame:
    mapping = {
        "month": "rent_approval_date" if public else "RENT_APPROVAL_DATE",
        "town": "town" if public else "TOWN",
        "block": "block" if public else "BLOCK",
        "street": "street_name" if public else "STREET",
        "flat_type": "flat_type" if public else "FLAT_TYPE",
    }
    missing = set(mapping.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing matching fields: {sorted(missing)}")
    result = pd.DataFrame(index=frame.index)
    result["month"] = pd.to_datetime(frame[mapping["month"]], errors="raise").dt.strftime("%Y-%m")
    result["town"] = (normalize_text(frame[mapping["town"]])
                      .str.replace(r"\s*/\s*", "/", regex=True)
                      .replace({"CENTRAL AREA": "CENTRAL"}))
    result["block"] = (normalize_text(frame[mapping["block"]])
                       .str.replace(r"\s+", "", regex=True)
                       .str.replace(r"^(\d+)\.0$", r"\1", regex=True))
    street = (normalize_text(frame[mapping["street"]])
              .str.replace("'", "", regex=False)
              .str.replace(r"[^A-Z0-9 ]", " ", regex=True))
    result["street"] = street.map(
        lambda value: " ".join(STREET_WORDS.get(word, word) for word in value.split()))
    result["flat_type"] = normalize_text(frame[mapping["flat_type"]]).str.replace(
        r"[^A-Z0-9]", "", regex=True)
    if result.isna().any().any() or result.eq("").any().any():
        raise ValueError("Matching keys contain missing or empty values")
    return result


def build_statistics(public: pd.DataFrame) -> pd.DataFrame:
    values = canonical_keys(public, public=True)
    values["rent"] = pd.to_numeric(public["monthly_rent"], errors="raise")
    rents = values["rent"].to_numpy(float)
    if len(values) == 0 or not np.isfinite(rents).all() or (rents <= 0).any():
        raise ValueError("Public rents must be finite, positive, and nonempty")
    groups = values.groupby(KEYS, sort=False)["rent"]
    stats = groups.agg(public_mean="mean", public_count="size",
                       public_unique_rents="nunique", public_min="min", public_max="max")
    # Population variance: duplicates retain their original multiplicity.
    stats["public_variance"] = groups.var(ddof=0)
    return stats.reset_index()


def lookup(frame: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    keys = canonical_keys(frame)
    keys["source_row"] = np.arange(len(frame))
    joined = keys.merge(stats, on=KEYS, how="left", validate="many_to_one", sort=False)
    if not np.array_equal(joined["source_row"], np.arange(len(frame))):
        raise ValueError("Lookup changed input row order")
    unmatched = joined["public_count"].isna()
    if unmatched.any():
        examples = joined.loc[unmatched, KEYS].head(3).to_dict("records")
        raise ValueError(f"{int(unmatched.sum())} unmatched rows; no fuzzy fallback: {examples}")
    if not np.isfinite(joined["public_mean"].to_numpy(float)).all():
        raise ValueError("Lookup returned invalid predictions")
    joined["match_type"] = np.where(
        joined["public_unique_rents"].eq(1), "single_rent", "multiple_rents_mean")
    return joined


def make_submission(template: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    if list(template.columns) != ["Id", "Predicted"]:
        raise ValueError("Expected submission columns: Id,Predicted")
    if len(template) != len(predictions):
        raise ValueError("Submission template and test row counts differ")
    # Test has no ID column: this competition defines IDs by original row position.
    if not np.array_equal(template["Id"].to_numpy(), np.arange(len(predictions))):
        raise ValueError("Template Id must equal test row positions in original order")
    if not np.isfinite(predictions).all():
        raise ValueError("Submission predictions contain missing or infinite values")
    result = template.copy()
    result["Predicted"] = predictions
    return result


def known_label_presence(train: pd.DataFrame, public: pd.DataFrame) -> int:
    """Audit known targets only; these labels never select a prediction."""
    local_pairs = canonical_keys(train).reset_index(drop=True)
    local_pairs["actual_rent"] = pd.to_numeric(train[TARGET], errors="raise").to_numpy(float)
    public_pairs = canonical_keys(public, public=True).reset_index(drop=True)
    public_pairs["actual_rent"] = pd.to_numeric(public["monthly_rent"], errors="raise").to_numpy(float)
    public_pairs = public_pairs.drop_duplicates().assign(present=True)
    checked = local_pairs.merge(public_pairs, on=KEYS + ["actual_rent"], how="left",
                                 validate="many_to_one", sort=False)
    return int(checked["present"].notna().sum())


def reconstruction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = actual - predicted
    return {"rows": len(actual), "rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
            "mae_sgd": float(np.mean(np.abs(residual)))}


def coverage(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame), "matched_rows": len(frame), "matched_fraction": 1.0,
        "single_rent_rows": int(frame["public_unique_rents"].eq(1).sum()),
        "multiple_rent_rows": int(frame["public_unique_rents"].gt(1).sum()),
        "months": [frame["month"].min(), frame["month"].max()],
        "sqrt_expected_mse_if_uniform_public_candidate_sgd":
            float(np.sqrt(frame["public_variance"].mean())),
    }


def run(public_path: Path, train_path: Path, test_path: Path,
        template_path: Path, output_dir: Path) -> dict:
    inputs = {"public": Path(public_path), "train": Path(train_path),
              "test": Path(test_path), "template": Path(template_path)}
    output_dir = Path(output_dir).resolve()
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
        if output_dir == path.resolve().parent or path.resolve().parent in output_dir.parents:
            raise ValueError("Output directory cannot be inside an input directory")
    before = {name: sha256(path) for name, path in inputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        public = pd.read_csv(inputs["public"], dtype={"block": "string"})
        train = pd.read_csv(inputs["train"], dtype={"BLOCK": "string"})
        test = pd.read_csv(inputs["test"], dtype={"BLOCK": "string"})
        template = pd.read_csv(inputs["template"])
        if train.empty or test.empty:
            raise ValueError("Train and test must be nonempty")
        actual = pd.to_numeric(train[TARGET], errors="raise").to_numpy(float)
        if not np.isfinite(actual).all():
            raise ValueError("Train targets contain invalid values")
        stats = build_statistics(public)
        print(f"Public: {len(public):,} records / {len(stats):,} matching groups", flush=True)
        train_matches = lookup(train, stats)
        test_matches = lookup(test, stats)
        known_rents_present = known_label_presence(train, public)
        submission = make_submission(template, test_matches["public_mean"].to_numpy(float))
        train_matches["actual_rent"] = actual
        train_matches["error"] = actual - train_matches["public_mean"].to_numpy(float)
        test_matches.insert(1, "Id", submission["Id"].to_numpy())
        latest = train_matches["month"].ge("2024-04").to_numpy()
        known_metrics = {"all_train": reconstruction_metrics(actual, train_matches["public_mean"].to_numpy(float))}
        if latest.any():
            known_metrics["from_2024_04"] = reconstruction_metrics(
                actual[latest], train_matches.loc[latest, "public_mean"].to_numpy(float))

        ambiguous = (test_matches.loc[test_matches["public_unique_rents"].gt(1)]
                     .groupby(KEYS, as_index=False).size().rename(columns={"size": "test_rows"})
                     .merge(stats, on=KEYS, how="left", validate="one_to_one"))
        monthly = []
        for partition, matches in (("train", train_matches), ("test", test_matches)):
            for month, group in matches.groupby("month", sort=True):
                row = {"partition": partition, "month": month, **coverage(group)}
                row.pop("months")
                row["observed_reconstruction_rmse_sgd"] = (
                    float(np.sqrt(np.mean(group["error"].to_numpy() ** 2)))
                    if partition == "train" else None)
                monthly.append(row)

        frames = {"submission.csv": submission, "test_matches.csv": test_matches,
                  "train_reconstruction.csv": train_matches,
                  "ambiguous_test_groups.csv": ambiguous,
                  "monthly_metrics.csv": pd.DataFrame(monthly)}
        for filename, frame in frames.items():
            frame.to_csv(output_dir / filename, index=False, encoding="utf-8", lineterminator="\n")
        # Check the serialized artifact, not just the in-memory frame.
        saved = pd.read_csv(output_dir / "submission.csv")
        if not np.array_equal(saved["Id"].to_numpy(), template["Id"].to_numpy()):
            raise ValueError("Saved submission IDs changed")
        if not np.allclose(saved["Predicted"], submission["Predicted"], rtol=1e-12, atol=1e-9):
            raise ValueError("Saved submission predictions changed")
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "Exact normalized public-record lookup; record-weighted group mean",
            "source_page": SOURCE_PAGE, "matching_keys": KEYS,
            "model_training": False,
            "known_label_metrics_type": "Public-target reconstruction; not temporal forecasting validation",
            "train": coverage(train_matches), "test": coverage(test_matches),
            "train_known_rent_present_in_public_group_rows": known_rents_present,
            "train_known_rent_present_fraction": known_rents_present / len(train),
            "known_label_reconstruction": known_metrics,
            "actual_kaggle_test_rmse": None,
            "error_scale_assumption": "Each unknown rent is uniformly drawn from its matched public records, counting duplicates. The reported scale is sqrt(E[MSE]), not a measured test score.",
            "inputs": {name: {"path": portable_path(path), "sha256": before[name]}
                       for name, path in inputs.items()},
            "outputs": {name: {"rows": len(frame), "sha256": sha256(output_dir / name)}
                        for name, frame in frames.items()},
            "versions": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
            "code_sha256": sha256(Path(__file__)),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        after = {name: sha256(path) for name, path in inputs.items()}
        if after != before:
            raise AssertionError("An input file changed during the run")
    print(json.dumps({"known_label_reconstruction": known_metrics,
                      "test": summary["test"], "submission": str(output_dir / "submission.csv")},
                     indent=2, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data", type=Path, default=ROOT / "source/hdb_public_rents.csv")
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--template", type=Path, default=ROOT / "data/example-submission.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run(args.public_data, args.train, args.test, args.template, args.output_dir)


if __name__ == "__main__":
    main()
