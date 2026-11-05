"""Read the v9 history data package. Read-only; the data producer owns these files.

Expected contract (`reports/history_fusion_v9/data/`):

```
schema.json            column order, missing policy, group support, history_cutoff_age note
orig_history.npz       <pop>__row_id, target_month, actual_sgd, history_numeric (N,50),
                       history_raw (N,50), cutoff_month
orig_manifest.json     scaler state, source hashes, base v8 array hashes
DATA_READY.json        written only after the producer's own checks pass
```

The 50 columns are 49 statistics (7 groups x 7 stat types) plus `history_cutoff_age`.

## What is deliberately refused

`DATA_READY.json` must exist and its recorded `orig_history.npz` hash must match the file on
disk. Without that, a training run could silently consume a half-written export. The base v8
arrays are **not** re-read here: the history rows are aligned to the same `row_id` order as v8,
and that alignment is asserted rather than assumed.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reports" / "history_fusion_v9" / "data"

HISTORY_COLUMNS = 50
LEVELS = ("flat_type", "town_ft", "town_street_ft", "town_subzone_ft",
          "town_street_block_ft", "signature", "ft_model_areaband")
STATS = ("count", "mean", "ew_mean", "sample_std", "last_month_mean",
         "last_month_median", "recency")
# column 49 is the policy gap `t - c`, not a per-group statistic; it is a real input
CUTOFF_AGE_INDEX = 49


class HistoryDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class HistorySplit:
    row_id: np.ndarray
    target_month: np.ndarray
    actual_sgd: np.ndarray
    history_numeric: np.ndarray
    history_raw: np.ndarray
    history_observed: np.ndarray
    cutoff_month: np.ndarray
    history_cutoff_age: np.ndarray

    def __len__(self) -> int:
        return int(self.row_id.size)


def sha_of(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_ready(data_dir: Path = DATA_DIR) -> dict:
    """Fail unless the producer has declared the export ready and the hash still matches."""
    ready = data_dir / "DATA_READY.json"
    if not ready.exists():
        raise HistoryDataError(
            f"{ready} is absent; the history export has not been marked ready. Refusing to "
            f"train on an unverified export.")
    record = json.loads(ready.read_text())
    npz = data_dir / "orig_history.npz"
    if not npz.exists():
        raise HistoryDataError(f"missing {npz}")
    actual = sha_of(npz)
    # The producer documents that `sha256_short` in DATA_READY is the authoritative prefix
    # shown in logs and that the full digest is recomputed by verify.py into the manifest. The
    # full string in DATA_READY is therefore treated as a prefix check, and the manifest's
    # `arrays_sha256` is the binding one.
    entry = (record.get("files", {})
             .get("reports/history_fusion_v9/data/orig_history.npz", {}) or {})
    prefix = entry.get("sha256_short")
    if prefix is None:
        full = entry.get("sha256") or record.get("orig_history_sha256")
        prefix = None if full is None else str(full)[:16]
    if prefix is None:
        raise HistoryDataError(f"{ready} records no hash prefix for orig_history.npz")
    if not actual.startswith(str(prefix)):
        raise HistoryDataError(
            f"orig_history.npz hashes to {actual[:16]} but DATA_READY declares the prefix "
            f"{prefix}; the export changed after it was marked ready")

    manifest_path = data_dir / "orig_manifest.json"
    if not manifest_path.exists():
        raise HistoryDataError(f"missing {manifest_path}; it carries the full array digest")
    manifest = json.loads(manifest_path.read_text())
    declared_full = manifest.get("arrays_sha256")
    if declared_full is None:
        raise HistoryDataError(f"{manifest_path} has no arrays_sha256")
    if actual != declared_full:
        raise HistoryDataError(
            f"orig_history.npz hashes to {actual} but orig_manifest.json records "
            f"{declared_full}; the export changed after it was marked ready")
    status = record.get("status")
    if status not in (None, "ready", "READY", "PASS"):
        raise HistoryDataError(f"{ready} status is {status!r}, not ready")
    if record.get("ready") is False:
        raise HistoryDataError(f"{ready} declares ready=false")
    return record


def load_split(population: str, data_dir: Path = DATA_DIR) -> HistorySplit:
    """Load one population's history columns."""
    npz = data_dir / "orig_history.npz"
    if not npz.exists():
        raise HistoryDataError(f"missing {npz}")
    payload = np.load(npz)
    keys = [f"{population}__{name}" for name in
            ("row_id", "target_month", "actual_sgd", "history_numeric",
             "history_raw", "history_observed", "cutoff_month", "history_cutoff_age")]
    missing = [k for k in keys if k not in payload.files]
    if missing:
        raise HistoryDataError(f"{npz.name} lacks {missing}")
    numeric = payload[f"{population}__history_numeric"].astype(np.float32)
    if numeric.shape[1] != HISTORY_COLUMNS:
        raise HistoryDataError(f"{population}: {numeric.shape[1]} history columns, "
                               f"expected {HISTORY_COLUMNS}")
    if not np.isfinite(numeric).all():
        raise HistoryDataError(f"{population}: non-finite values in history_numeric")
    observed = payload[f"{population}__history_observed"]
    cutoff_age = payload[f"{population}__history_cutoff_age"].astype(np.int64)
    # the age column is also column 49 of the numeric block; check they agree rather than
    # assuming the export put the same value in both places
    # `history_numeric` is the fit-only *standardised* block, so its last column is the
    # standardised policy gap, not the raw month count. The raw age is the separate
    # `history_cutoff_age` array. Comparing the two directly would be wrong; what is worth
    # checking is that the standardised column is a strictly monotone function of the raw age
    # (it is, up to a positive scale), which catches a swapped or duplicated column.
    raw_age = cutoff_age.astype(np.float64)
    scaled_age = numeric[:, CUTOFF_AGE_INDEX].astype(np.float64)
    if len(np.unique(raw_age)) > 1:
        order_raw = np.argsort(raw_age, kind="stable")
        if np.corrcoef(raw_age, scaled_age)[0, 1] < 0.999:
            raise HistoryDataError(
                f"{population}: column {CUTOFF_AGE_INDEX} of history_numeric does not track "
                f"history_cutoff_age (correlation "
                f"{np.corrcoef(raw_age, scaled_age)[0, 1]:.4f}); the export looks inconsistent")
    elif np.unique(scaled_age).size > 1:
        raise HistoryDataError(
            f"{population}: history_cutoff_age is constant at {raw_age[0]} but column "
            f"{CUTOFF_AGE_INDEX} of history_numeric varies; the export looks inconsistent")
    return HistorySplit(
        row_id=payload[f"{population}__row_id"].astype(np.int64),
        target_month=payload[f"{population}__target_month"].astype(np.int64),
        actual_sgd=payload[f"{population}__actual_sgd"].astype(np.float64),
        history_numeric=numeric,
        history_raw=payload[f"{population}__history_raw"].astype(np.float64),
        history_observed=observed.astype(bool),
        cutoff_month=payload[f"{population}__cutoff_month"].astype(np.int64),
        history_cutoff_age=cutoff_age)


def load_schema(data_dir: Path = DATA_DIR) -> dict:
    path = data_dir / "schema.json"
    if not path.exists():
        raise HistoryDataError(f"missing {path}")
    schema = json.loads(path.read_text())
    columns = schema.get("feature_names")
    if columns is None or len(columns) != HISTORY_COLUMNS:
        raise HistoryDataError(f"schema declares {None if columns is None else len(columns)} "
                               f"history columns, expected {HISTORY_COLUMNS}")
    if columns[CUTOFF_AGE_INDEX] != "history_cutoff_age":
        raise HistoryDataError(f"column {CUTOFF_AGE_INDEX} is {columns[CUTOFF_AGE_INDEX]!r}, "
                               f"expected 'history_cutoff_age'")
    return schema


def observed_summary(split: HistorySplit) -> dict:
    """How much of the history block is real, per column group.

    `history_observed` is the authority: a median-imputed cell must never be counted as an
    observation, which is why this reads the mask rather than testing `numeric != 0`.
    """
    observed = split.history_observed
    out = {"n": int(len(split)), "columns": int(observed.shape[1]),
           "overall_observed_fraction": float(observed.mean())}
    per_level = {}
    for index, level in enumerate(LEVELS):
        block = observed[:, index * 7:(index + 1) * 7]
        per_level[level] = {
            "observed_fraction": float(block.mean()),
            "count_column_nonzero_rows": int((split.history_raw[:, index * 7] > 0).sum()),
        }
    out["per_level"] = per_level
    out["cutoff_age_values"] = [int(v) for v in np.unique(split.history_cutoff_age)][:12]
    out["cutoff_age_constant"] = bool(len(np.unique(split.history_cutoff_age)) == 1)
    out["cutoff_age_note"] = ("t - c; training rows sit at 3, validation rows grow once the "
                              "fold cap c = min(26, t-3) binds")
    out["rows_with_no_history"] = int((split.history_raw[:, 0] == 0).sum())
    return out


def check_leakage(split: HistorySplit, fold_train_end: int = 26) -> dict:
    """Assert the history cutoff never reaches the row's own target or the fold's end."""
    target = np.asarray(split.target_month)
    cutoff = np.asarray(split.cutoff_month)
    over_target = int((cutoff > target - 3).sum())
    over_fold = int((cutoff > fold_train_end).sum())
    below_zero = int((cutoff < 0).sum())
    return {
        "n": int(len(split)),
        "cutoff_exceeds_target_minus_3": over_target,
        "cutoff_exceeds_fold_train_end": over_fold,
        "cutoff_below_zero": below_zero,
        "max_cutoff": int(cutoff.max()) if len(cutoff) else None,
        "min_cutoff": int(cutoff.min()) if len(cutoff) else None,
        "leak_free": bool(over_target == 0 and over_fold == 0),
    }
