"""Loading the real v3 macro arrays and reshaping them for recurrence.

What is preserved, and why it matters
-------------------------------------
* **Real dates and units travel with the tensors.** ``dbs_dates`` (YYYYMMDD),
  ``dbs_asof_date``, ``coe_months`` (YYYYMM) and ``window_cutoff_month`` are kept so
  a consumer can audit a token against the calendar it came from, instead of
  trusting a positional index.
* **The policy is carried, not assumed.** ``origin_asof`` and
  ``target_previous_month`` have different information budgets; the loader records
  which one produced the arrays and a consumer that wants a single-policy model can
  assert on it.
* **Padding is data, not zero.** The masks are loaded alongside the values. A window
  with no usable history is reported as *missing* through an explicit flag, never as
  a tensor of NaNs and never as a silently-valid zero vector.
* **COE stays months-on-the-time-axis.** ``coe_seq`` is exported as
  ``(window, category, month, channel)``; it is transposed to
  ``(window, month, category*channel)`` so a GRU iterates over months. Concatenating
  the four categories along the time axis would turn a 12-month history into 48 fake
  "months", which is the exact error this function exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.macro_encoders_v3 import contract


@dataclass
class MacroWindows:
    """One stage's unique macro windows plus their provenance."""

    dbs_seq: np.ndarray            # (W, 63, 7) float32
    dbs_mask: np.ndarray           # (W, 63) float32
    dbs_dates: np.ndarray          # (W, 63) int32 YYYYMMDD, 0 = padding
    dbs_asof_date: np.ndarray      # (W,) int32 YYYYMMDD, 0 = no observation
    dbs_scalars: np.ndarray        # (W, 12) float32
    dbs_scalar_mask: np.ndarray    # (W, 12) float32
    coe_months_seq: np.ndarray     # (W, 12, 24) float32  months x (category*channel)
    coe_mask: np.ndarray           # (W, 4, 12) float32
    coe_month_ids: np.ndarray      # (W, 12) int32 YYYYMM, 0 = padding
    coe_scalars: np.ndarray        # (W, 4, 9) float32
    coe_scalar_mask: np.ndarray    # (W, 4, 9) float32
    coe_incomplete: np.ndarray     # (W, 4, 12) float32
    window_cutoff_month: np.ndarray  # (W,) int32
    window_count: np.ndarray       # (W,) int32
    policy: str
    contract_id: str
    split_id: str
    stage: str

    # ---- derived, explicit missing state -------------------------------
    @property
    def n_windows(self) -> int:
        return int(self.dbs_seq.shape[0])

    def dbs_has_history(self) -> np.ndarray:
        """(W,) bool: at least one real DBS observation in the window."""
        return self.dbs_mask.sum(axis=1) > 0

    def coe_has_history(self) -> np.ndarray:
        """(W,) bool: at least one observed category-month."""
        return self.coe_mask.sum(axis=(1, 2)) > 0


def _flatten_coe(coe_seq: np.ndarray) -> np.ndarray:
    """(W, cat, month, chan) -> (W, month, cat*chan).

    The month axis is preserved; categories become parallel channels of the same
    month.  Channel order is category-major: a.premium, a.quota, ..., e.observed.
    """
    if coe_seq.ndim != 4:
        raise ValueError(f"coe_seq must be 4-D, got shape {coe_seq.shape}")
    n, n_cat, n_month, n_chan = coe_seq.shape
    if n_cat != len(contract.COE_CATEGORIES):
        raise ValueError(f"expected {len(contract.COE_CATEGORIES)} categories, got {n_cat}")
    if n_month != contract.COE_LOOKBACK_MONTHS:
        raise ValueError(f"expected {contract.COE_LOOKBACK_MONTHS} months, got {n_month}")
    if n_chan != len(contract.COE_CHANNELS):
        raise ValueError(f"expected {len(contract.COE_CHANNELS)} channels, got {n_chan}")
    # transpose to (W, month, cat, chan) then flatten the last two
    laid_out = np.transpose(coe_seq, (0, 2, 1, 3))
    return np.ascontiguousarray(laid_out.reshape(n, n_month, n_cat * n_chan))


def load_windows(
    manifest: dict,
    split_id: str,
    stage: str,
) -> MacroWindows:
    """Read one stage's macro arrays from the recorded paths."""
    entry = contract.stage_entry(manifest, split_id, stage)
    path = contract.resolve(entry["artifacts"]["macro"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"macro artifact missing: {path}")
    arrays = np.load(path)

    required = ("dbs_seq", "dbs_mask", "dbs_dates", "dbs_asof_date",
                "dbs_scalars", "dbs_scalar_mask", "coe_seq", "coe_mask",
                "coe_months", "coe_scalars", "coe_scalar_mask", "coe_incomplete",
                "window_cutoff_month", "window_count")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{split_id}/{stage}: macro artifact missing keys {missing}")

    windows = MacroWindows(
        dbs_seq=arrays["dbs_seq"],
        dbs_mask=arrays["dbs_mask"],
        dbs_dates=arrays["dbs_dates"],
        dbs_asof_date=arrays["dbs_asof_date"],
        dbs_scalars=arrays["dbs_scalars"],
        dbs_scalar_mask=arrays["dbs_scalar_mask"],
        coe_months_seq=_flatten_coe(arrays["coe_seq"]),
        coe_mask=arrays["coe_mask"],
        coe_month_ids=arrays["coe_months"],
        coe_scalars=arrays["coe_scalars"],
        coe_scalar_mask=arrays["coe_scalar_mask"],
        coe_incomplete=arrays["coe_incomplete"],
        window_cutoff_month=arrays["window_cutoff_month"],
        window_count=arrays["window_count"],
        policy=manifest["policy"],
        contract_id=manifest["contract_id"],
        split_id=split_id,
        stage=stage,
    )
    _validate(windows, entry)
    return windows


def _validate(windows: MacroWindows, entry: dict) -> None:
    """Shape and consistency checks against the manifest and each other."""
    n = windows.n_windows
    if windows.dbs_seq.shape[1:] != (contract.DBS_LOOKBACK_ROWS,
                                     len(contract.DBS_CHANNELS)):
        raise ValueError(f"DBS seq shape {windows.dbs_seq.shape} unexpected")
    if windows.coe_months_seq.shape[1:] != (contract.COE_LOOKBACK_MONTHS,
                                            contract.COE_FLAT_CHANNELS):
        raise ValueError(f"COE flattened shape {windows.coe_months_seq.shape} unexpected")
    if windows.dbs_mask.shape != (n, contract.DBS_LOOKBACK_ROWS):
        raise ValueError("DBS mask shape disagrees with the sequence")
    if windows.window_cutoff_month.shape != (n,):
        raise ValueError("window_cutoff_month shape disagrees with the window count")
    if int(windows.window_count.sum()) != int(entry["n_rows"]):
        raise ValueError(
            f"window_count sums to {int(windows.window_count.sum())} but the manifest "
            f"records {entry['n_rows']} rows")
    # Padding must be exactly zero in both value and date space.
    if np.any(windows.dbs_seq[windows.dbs_mask == 0] != 0):
        raise ValueError("DBS padding is not zero")
    if np.any(windows.dbs_dates[windows.dbs_mask == 0] != 0):
        raise ValueError("DBS padding dates are not zero")
    coe_any = windows.coe_mask.max(axis=1)          # (W, month)
    if np.any(windows.coe_months_seq[coe_any == 0] != 0):
        raise ValueError("COE padding is not zero")


@dataclass
class Rows:
    """Per-supervised-row view: which window, which population."""

    row_window_index: np.ndarray   # (n,) int32
    row_id: np.ndarray             # (n,) int64
    macro_cutoff_month: np.ndarray  # (n,) int32
    population: np.ndarray         # (n,) '<U...'  'fit' | 'holdout' | 'validation'
    horizon_months: np.ndarray     # (n,) int32
    split_id: str
    stage: str
    policy: str


def load_rows(manifest: dict, split_id: str, stage: str) -> Rows:
    """Per-row window mapping plus the sample table's population labels."""
    entry = contract.stage_entry(manifest, split_id, stage)

    features_path = contract.resolve(entry["artifacts"]["features"]["path"])
    features = np.load(features_path)
    samples_path = contract.resolve(entry["artifacts"]["samples"]["path"])
    import pandas as pd
    samples = pd.read_csv(samples_path, usecols=["SOURCE_ROW_ID", "population",
                                                 "horizon_months"])

    row_window = features["row_window_index"]
    if len(samples) != len(row_window):
        raise ValueError(
            f"{split_id}/{stage}: {len(samples)} sample rows vs {len(row_window)} "
            "row_window_index entries")
    if not np.array_equal(samples["SOURCE_ROW_ID"].to_numpy(), features["row_id"]):
        raise ValueError(f"{split_id}/{stage}: row ids disagree between samples and features")

    return Rows(
        row_window_index=row_window,
        row_id=features["row_id"],
        macro_cutoff_month=features["macro_cutoff_month"],
        population=samples["population"].to_numpy().astype(str),
        horizon_months=samples["horizon_months"].to_numpy(dtype=np.int32),
        split_id=split_id,
        stage=stage,
        policy=manifest["policy"],
    )


def population_mask(rows: Rows, population: str) -> np.ndarray:
    """Boolean mask for one population, with a clear error on a typo."""
    known = set(np.unique(rows.population).tolist())
    if population not in known:
        raise KeyError(f"population {population!r} not present; available: {sorted(known)}")
    return rows.population == population
