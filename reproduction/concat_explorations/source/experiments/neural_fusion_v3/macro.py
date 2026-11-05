"""Reader for the v3 macro contract: deduplicated DBS/COE windows plus a row index.

The producer (`experiments/fusion_contract_v3/`) owns the macro data. This module reads it
and never derives a stock or COE statistic of its own: no resampling, no aggregation, no
backfill, no re-derivation of returns.

Layout:

    <contract>/macro/<split>/stage{A,B}.npz
        window_cutoff_month (n_windows,)          last month whose observations are allowed
        row_window_index    (n_rows,)             supervised row -> window
        dbs_seq             (n_windows, 63, 7)    trading-day axis, fixed-unit values
        dbs_mask            (n_windows, 63)
        dbs_dates           (n_windows, 63)       YYYYMMDD
        dbs_scalars         (n_windows, 12)
        coe_seq             (n_windows, 4, 12, 6) category, month, channel
        coe_mask            (n_windows, 4, 12)
        coe_months          (n_windows, 12)       YYYYMM

Two axes that must never be confused: DBS is a **trading-day** axis of 63 steps; COE is a
**calendar-month** axis of 12 steps with four categories as a separate axis. A category is
not a time step.
"""
from __future__ import annotations

import json
import sys as _sys
from dataclasses import dataclass
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import numpy as np

DBS_WINDOW_DAYS = 63
COE_WINDOW_MONTHS = 12
COE_CATEGORIES = ("a", "b", "c", "e")
POLICIES = ("origin_asof", "target_previous_month")


class MacroNotReady(RuntimeError):
    """The macro artifacts (or the requested policy/stage) are absent."""


class MacroInvalid(ValueError):
    """The artifacts exist but violate an invariant this consumer relies on."""


@dataclass(frozen=True)
class StageMacro:
    """Macro arrays for one (split, stage), already validated."""

    split: str
    stage: str
    policy: str
    window_cutoff_month: np.ndarray
    row_window_index: np.ndarray
    dbs_seq: np.ndarray
    dbs_mask: np.ndarray
    dbs_dates: np.ndarray
    dbs_scalars: np.ndarray
    dbs_scalar_mask: np.ndarray
    dbs_asof_date: np.ndarray
    coe_seq: np.ndarray
    coe_mask: np.ndarray
    coe_months: np.ndarray
    coe_scalars: np.ndarray
    coe_scalar_mask: np.ndarray
    coe_incomplete: np.ndarray

    @property
    def n_windows(self) -> int:
        return len(self.window_cutoff_month)

    @property
    def n_rows(self) -> int:
        return len(self.row_window_index)

    @property
    def dbs_channels(self) -> int:
        return int(self.dbs_seq.shape[2])

    @property
    def coe_channels(self) -> int:
        return int(self.coe_seq.shape[3])

    def gather(self, row_index: np.ndarray) -> dict[str, np.ndarray]:
        """Windows for a set of supervised rows, in that row order."""
        window = self.row_window_index[row_index]
        return {
            "window_index": window,
            "dbs_seq": self.dbs_seq[window],
            "dbs_mask": self.dbs_mask[window],
            "dbs_scalars": self.dbs_scalars[window],
            "dbs_scalar_mask": self.dbs_scalar_mask[window],
            "coe_seq": self.coe_seq[window],
            "coe_mask": self.coe_mask[window],
            "coe_scalars": self.coe_scalars[window],
            "coe_scalar_mask": self.coe_scalar_mask[window],
            "macro_cutoff_month": self.window_cutoff_month[window],
        }

    def describe(self) -> dict:
        return {
            "split": self.split, "stage": self.stage, "policy": self.policy,
            "n_windows": self.n_windows, "n_rows": self.n_rows,
            "dbs": {"symbol": "D05.SI", "window_days": int(self.dbs_seq.shape[1]),
                    "channels": self.dbs_channels,
                    "windows_full_lookback": int((self.dbs_mask.sum(axis=1)
                                                  == self.dbs_mask.shape[1]).sum())},
            "coe": {"categories": list(COE_CATEGORIES),
                    "window_months": int(self.coe_seq.shape[2]),
                    "channels": self.coe_channels,
                    "incomplete_month_cells": int(self.coe_incomplete.sum())},
            "cutoff_range": [int(self.window_cutoff_month.min()),
                             int(self.window_cutoff_month.max())],
        }


def contract_dir(root, contract_id: str) -> _Path:
    return _Path(root) / contract_id


def load_manifest(directory) -> dict:
    path = _Path(directory) / "manifest.json"
    if not path.exists():
        raise MacroNotReady(f"v3 contract manifest missing: {path}")
    payload = json.loads(path.read_text())
    for key in ("contract_id", "contract_version", "policy"):
        if key not in payload:
            raise MacroInvalid(f"manifest missing {key!r}")
    return payload


def split_spec(manifest: dict, split: str) -> dict:
    if split not in manifest.get("splits", {}):
        raise MacroInvalid(f"split {split!r} not in manifest.splits")
    spec = manifest["splits"][split].get("split")
    if not isinstance(spec, dict):
        raise MacroInvalid(f"splits.{split}.split missing")
    for key in ("cutoff_index", "val_start_index", "val_end_index", "inner_cutoff_index"):
        if key not in spec:
            raise MacroInvalid(f"splits.{split}.split.{key} missing")
    return spec


def validate_policy(manifest: dict, policy: str) -> dict:
    declared = manifest.get("policy")
    if isinstance(declared, dict):
        declared = declared.get("observation_policy")
    if declared is None:
        raise MacroInvalid("manifest has no 'policy'")
    if declared != policy:
        raise MacroInvalid(
            f"manifest was built under policy {declared!r}, run requests {policy!r}; "
            "policies must not be mixed")
    return {"policy": declared, "strict_fixed_cutoff": declared == "origin_asof",
            "policies_implemented": manifest.get("policies_implemented", [])}


def load_stage(directory, manifest: dict, split: str, stage: str) -> StageMacro:
    directory = _Path(directory)
    spec = split_spec(manifest, split)
    declared = manifest["splits"][split]["stages"].get(stage)
    if declared is None:
        raise MacroInvalid(f"splits.{split}.stages.{stage} missing")
    stage_policy = declared.get("policy")
    manifest_policy = manifest.get("policy")
    if isinstance(manifest_policy, dict):
        manifest_policy = manifest_policy.get("observation_policy")
    if stage_policy is not None and stage_policy != manifest_policy:
        raise MacroInvalid(
            f"stage {stage} declares policy {stage_policy!r} but the manifest says "
            f"{manifest_policy!r}")

    path = directory / "macro" / split / f"stage{stage}.npz"
    if not path.exists():
        raise MacroNotReady(f"macro arrays missing: {path}")
    stored = np.load(path, allow_pickle=False)
    required = ("window_cutoff_month", "row_window_index", "dbs_seq", "dbs_mask", "dbs_dates",
                "dbs_scalars", "dbs_scalar_mask", "dbs_asof_date", "coe_seq", "coe_mask",
                "coe_months", "coe_scalars", "coe_scalar_mask", "coe_incomplete")
    missing = [k for k in required if k not in stored.files]
    if missing:
        raise MacroInvalid(f"{path.name} missing arrays {missing}")

    stage_data = StageMacro(
        split=split, stage=stage, policy=str(manifest_policy),
        window_cutoff_month=stored["window_cutoff_month"],
        row_window_index=stored["row_window_index"],
        dbs_seq=stored["dbs_seq"], dbs_mask=stored["dbs_mask"], dbs_dates=stored["dbs_dates"],
        dbs_scalars=stored["dbs_scalars"], dbs_scalar_mask=stored["dbs_scalar_mask"],
        dbs_asof_date=stored["dbs_asof_date"],
        coe_seq=stored["coe_seq"], coe_mask=stored["coe_mask"], coe_months=stored["coe_months"],
        coe_scalars=stored["coe_scalars"], coe_scalar_mask=stored["coe_scalar_mask"],
        coe_incomplete=stored["coe_incomplete"],
    )
    _validate(stage_data, path, declared, spec)
    return stage_data


def _validate(stage: StageMacro, path: _Path, declared: dict, spec: dict) -> None:
    n_rows_declared = declared.get("n_rows")
    if n_rows_declared is not None and int(n_rows_declared) != stage.n_rows:
        raise MacroInvalid(
            f"{path.name}: row_window_index has {stage.n_rows} rows, manifest declares "
            f"{n_rows_declared}")

    if stage.dbs_seq.ndim != 3 or stage.dbs_seq.shape[1] != DBS_WINDOW_DAYS:
        raise MacroInvalid(
            f"{path.name}: dbs_seq must be (windows, {DBS_WINDOW_DAYS}, channels); "
            f"got {stage.dbs_seq.shape}")
    if stage.coe_seq.ndim != 4 or stage.coe_seq.shape[1] != len(COE_CATEGORIES) \
            or stage.coe_seq.shape[2] != COE_WINDOW_MONTHS:
        raise MacroInvalid(
            f"{path.name}: coe_seq must be (windows, {len(COE_CATEGORIES)}, "
            f"{COE_WINDOW_MONTHS}, channels); got {stage.coe_seq.shape}")
    if stage.dbs_mask.shape != stage.dbs_seq.shape[:2]:
        raise MacroInvalid(f"{path.name}: dbs_mask shape {stage.dbs_mask.shape} != "
                           f"{stage.dbs_seq.shape[:2]}")
    if stage.coe_mask.shape != stage.coe_seq.shape[:3]:
        raise MacroInvalid(f"{path.name}: coe_mask shape {stage.coe_mask.shape} != "
                           f"{stage.coe_seq.shape[:3]}")
    for name, mask in (("dbs", stage.dbs_mask), ("coe", stage.coe_mask)):
        if not np.isin(mask, (0.0, 1.0)).all():
            raise MacroInvalid(f"{path.name}: {name} mask must be 0/1")

    padded_dbs = stage.dbs_mask == 0
    if padded_dbs.any():
        if not np.isfinite(stage.dbs_seq[padded_dbs]).all():
            raise MacroInvalid(f"{path.name}: padded DBS steps are non-finite")
        if np.abs(stage.dbs_seq[padded_dbs]).max(initial=0.0) > 0:
            raise MacroInvalid(f"{path.name}: padded DBS steps are non-zero")
    padded_coe = stage.coe_mask == 0
    if padded_coe.any():
        if not np.isfinite(stage.coe_seq[padded_coe]).all():
            raise MacroInvalid(f"{path.name}: padded COE cells are non-finite")
        if np.abs(stage.coe_seq[padded_coe]).max(initial=0.0) > 0:
            raise MacroInvalid(f"{path.name}: padded COE cells are non-zero")

    if stage.row_window_index.size and (stage.row_window_index.min() < 0
                                        or stage.row_window_index.max() >= stage.n_windows):
        raise MacroInvalid(f"{path.name}: row_window_index out of range")

    if (stage.dbs_mask == 1).any():
        packed = (stage.dbs_dates // 10000) * 100 + (stage.dbs_dates // 100) % 100
        cutoff = stage.window_cutoff_month[:, None]
        cutoff_packed = (2021 + cutoff // 12) * 100 + (cutoff % 12 + 1)
        if (packed[stage.dbs_mask == 1]
                > np.broadcast_to(cutoff_packed, packed.shape)[stage.dbs_mask == 1]).any():
            raise MacroInvalid(f"{path.name}: a DBS observation postdates its window cutoff")

    if (stage.coe_mask == 1).any():
        # coe_months is (windows, months); the mask adds a category axis, so broadcast the
        # month grid across categories before comparing element-wise
        packed = (stage.coe_months // 100) * 100 + stage.coe_months % 100
        packed = np.repeat(packed[:, None, :], stage.coe_mask.shape[1], axis=1)
        cutoff = stage.window_cutoff_month[:, None, None]
        cutoff_packed = np.broadcast_to(
            (2021 + cutoff // 12) * 100 + (cutoff % 12 + 1), packed.shape)
        if (packed[stage.coe_mask == 1] > cutoff_packed[stage.coe_mask == 1]).any():
            raise MacroInvalid(f"{path.name}: a COE month postdates its window cutoff")


def load_split(directory, manifest: dict, split: str) -> dict[str, StageMacro]:
    return {stage: load_stage(directory, manifest, split, stage) for stage in ("A", "B")}


def policy_report(manifest: dict) -> dict:
    declared = manifest.get("policy")
    if isinstance(declared, dict):
        declared = declared.get("observation_policy")
    return {
        "observation_policy": declared,
        "strict_fixed_cutoff_forecasting": declared == "origin_asof",
        "macro_cutoff_field": "window_cutoff_month / macro_cutoff_month",
        "rental_origin_field": "origin_month_index (separate, never conflated)",
        "policies_implemented": manifest.get("policies_implemented", []),
        "description": manifest.get("policy_description"),
    }
