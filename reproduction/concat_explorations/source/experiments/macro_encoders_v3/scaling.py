"""Fit-population-only standardisation state, saved so it can be rebuilt exactly.

Rules encoded here
------------------
* **Only the fitting population's windows are used to fit.** For stage A that is
  ``population == "fit"``; for stage B likewise.  The inner holdout, the validation
  window and the "rows that share a window with a validation row" are never used —
  the state is fitted on *windows that belong to fit rows*, then applied to every
  window.
* **Masks are respected.** A padded position never contributes to a mean, a
  standard deviation or a quantile.  A channel that is observed in one row of a
  window and padded in another therefore still gets a correct statistic.
* **A degenerate channel is marked, not divided by zero.** If a channel has fewer
  than two usable observations, or zero variance, ``scale`` is stored as 1.0 and
  ``degenerate`` is set, so the transform is the identity for that channel and the
  fact is visible in the state rather than hidden in an epsilon.
* **Counts and units stay recognisable.** ``n_used`` is recorded per channel.
  Volumes and premiums are standardised in their logged or native units respectively
  (the loader already supplies the contract's units); nothing here re-logs.

The state is a plain dict of numpy arrays plus JSON-scalar metadata, so
``save``/``load`` round-trips exactly and a reviewer can diff two states.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.macro_encoders_v3 import contract


@dataclass
class ScalerState:
    """Mean/std per channel for one block, plus provenance."""

    block: str                     # "dbs" | "coe" | "dbs_scalar" | "coe_scalar"
    channel_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    degenerate: np.ndarray         # bool, True where scale fell back to 1.0
    n_used: np.ndarray             # int per channel
    fitted_on: str                 # human-readable population description
    split_id: str
    stage: str
    policy: str
    n_windows_used: int
    extra: dict = field(default_factory=dict)

    # ---- transform ------------------------------------------------------
    def transform(self, values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Standardise; positions outside ``mask`` are returned as 0.0.

        Returning 0.0 for padding (rather than NaN) is deliberate: the encoder
        consumes it together with the mask, and a NaN would propagate through the
        LSTM into every downstream token. The mask, not the value, carries the
        missingness.
        """
        values = np.asarray(values, dtype=np.float32)
        mean = self.mean.astype(np.float32)
        scale = self.scale.astype(np.float32)
        if values.shape[-1] != len(self.channel_names):
            raise ValueError(
                f"{self.block}: expected {len(self.channel_names)} channels, "
                f"got {values.shape[-1]}")
        scaled = (values - mean) / scale
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            # A mask may describe positions only (...,), or positions x channels
            # (..., C) when a block has per-channel validity.
            if mask.shape == values.shape[:-1]:
                scaled = np.where(mask[..., None], scaled, 0.0)
            elif mask.shape == values.shape:
                scaled = np.where(mask, scaled, 0.0)
            else:
                raise ValueError(
                    f"{self.block}: mask shape {mask.shape} matches neither "
                    f"positions {values.shape[:-1]} nor values {values.shape}")
        # Replace any non-finite with 0 and let the mask carry the meaning.
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "block": self.block,
            "channel_names": list(self.channel_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "degenerate": self.degenerate.astype(bool).tolist(),
            "n_used": self.n_used.astype(int).tolist(),
            "fitted_on": self.fitted_on,
            "split_id": self.split_id,
            "stage": self.stage,
            "policy": self.policy,
            "n_windows_used": int(self.n_windows_used),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ScalerState":
        return cls(
            block=payload["block"],
            channel_names=list(payload["channel_names"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            degenerate=np.asarray(payload["degenerate"], dtype=bool),
            n_used=np.asarray(payload["n_used"], dtype=np.int64),
            fitted_on=payload["fitted_on"],
            split_id=payload["split_id"],
            stage=payload["stage"],
            policy=payload["policy"],
            n_windows_used=int(payload["n_windows_used"]),
            extra=dict(payload.get("extra") or {}),
        )

    def save(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "ScalerState":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class MacroScalers:
    """The four states a macro encoder needs, saved and loaded as one unit."""

    dbs: ScalerState
    dbs_scalar: ScalerState
    coe: ScalerState
    coe_scalar: ScalerState
    contract_id: str
    split_id: str
    stage: str
    policy: str

    def save(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "dbs": self.dbs.save(directory, "dbs_channels"),
            "dbs_scalar": self.dbs_scalar.save(directory, "dbs_scalars"),
            "coe": self.coe.save(directory, "coe_channels"),
            "coe_scalar": self.coe_scalar.save(directory, "coe_scalars"),
        }
        index = {
            "contract_id": self.contract_id,
            "split_id": self.split_id,
            "stage": self.stage,
            "policy": self.policy,
            "files": {name: path.name for name, path in paths.items()},
        }
        (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        return index

    @classmethod
    def load(cls, directory: Path) -> "MacroScalers":
        index = json.loads((directory / "index.json").read_text())
        files = index["files"]
        return cls(
            dbs=ScalerState.load(directory / files["dbs"]),
            dbs_scalar=ScalerState.load(directory / files["dbs_scalar"]),
            coe=ScalerState.load(directory / files["coe"]),
            coe_scalar=ScalerState.load(directory / files["coe_scalar"]),
            contract_id=index["contract_id"],
            split_id=index["split_id"],
            stage=index["stage"],
            policy=index["policy"],
        )


# ---------------------------------------------------------------- fitting
def _fit_channel(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, bool, int]:
    """mean, scale, degenerate, n_used for one channel over masked entries."""
    usable = np.asarray(mask, dtype=bool) & np.isfinite(values)
    flat = np.asarray(values, dtype=np.float64)[usable]
    n_used = int(flat.size)
    if n_used < 2:
        return 0.0, 1.0, True, n_used
    mean = float(np.mean(flat))
    scale = float(np.std(flat, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        return mean, 1.0, True, n_used
    return mean, scale, False, n_used


def fit_dbs_channels(window_values: np.ndarray, window_mask: np.ndarray,
                     *, split_id: str, stage: str, policy: str,
                     fitted_on: str) -> ScalerState:
    """Fit DBS channel statistics on the supplied (already fit-only) windows."""
    names = list(contract.DBS_CHANNELS)
    mean = np.zeros(len(names))
    scale = np.ones(len(names))
    degenerate = np.zeros(len(names), dtype=bool)
    n_used = np.zeros(len(names), dtype=np.int64)
    for index, name in enumerate(names):
        values = window_values[..., index]
        # The 'observed' channel is a flag: standardising it would be misleading.
        if name == "observed":
            mean[index], scale[index], degenerate[index] = 0.0, 1.0, False
            n_used[index] = int(np.asarray(window_mask).sum())
            continue
        mean[index], scale[index], degenerate[index], n_used[index] = _fit_channel(
            values, window_mask)
    return ScalerState(block="dbs", channel_names=names, mean=mean, scale=scale,
                       degenerate=degenerate, n_used=n_used, fitted_on=fitted_on,
                       split_id=split_id, stage=stage, policy=policy,
                       n_windows_used=int(window_values.shape[0]))


def fit_coe_channels(window_values: np.ndarray, window_mask: np.ndarray,
                     *, split_id: str, stage: str, policy: str,
                     fitted_on: str) -> ScalerState:
    """Fit COE month-channel statistics: ``window_values`` is (W, month, 24)."""
    n_cat = len(contract.COE_CATEGORIES)
    n_chan = len(contract.COE_CHANNELS)
    names = [f"coe_{category}_{channel}"
             for category in contract.COE_CATEGORIES
             for channel in contract.COE_CHANNELS]
    mean = np.zeros(len(names))
    scale = np.ones(len(names))
    degenerate = np.zeros(len(names), dtype=bool)
    n_used = np.zeros(len(names), dtype=np.int64)

    # window_mask is (W, cat, month); broadcast it onto channel space.
    per_month = np.transpose(window_mask, (0, 2, 1))          # (W, month, cat)
    flat_mask = np.repeat(per_month, n_chan, axis=2)          # (W, month, 24)
    for index, name in enumerate(names):
        channel = contract.COE_CHANNELS[index % n_chan]
        if channel == "observed":
            mean[index], scale[index], degenerate[index] = 0.0, 1.0, False
            n_used[index] = int(flat_mask[..., index].sum())
            continue
        mean[index], scale[index], degenerate[index], n_used[index] = _fit_channel(
            window_values[..., index], flat_mask[..., index])
    return ScalerState(block="coe", channel_names=names, mean=mean, scale=scale,
                       degenerate=degenerate, n_used=n_used, fitted_on=fitted_on,
                       split_id=split_id, stage=stage, policy=policy,
                       n_windows_used=int(window_values.shape[0]))


def fit_scalars(values: np.ndarray, mask: np.ndarray, names: list[str],
                block: str, *, split_id: str, stage: str, policy: str,
                fitted_on: str) -> ScalerState:
    """Fit one scalar block: ``values``/``mask`` are (W, channels)."""
    mean = np.zeros(len(names))
    scale = np.ones(len(names))
    degenerate = np.zeros(len(names), dtype=bool)
    n_used = np.zeros(len(names), dtype=np.int64)
    for index in range(len(names)):
        mean[index], scale[index], degenerate[index], n_used[index] = _fit_channel(
            values[..., index], mask[..., index])
    return ScalerState(block=block, channel_names=list(names), mean=mean, scale=scale,
                       degenerate=degenerate, n_used=n_used, fitted_on=fitted_on,
                       split_id=split_id, stage=stage, policy=policy,
                       n_windows_used=int(values.shape[0]))


def fit_macro_scalers(windows, rows) -> MacroScalers:
    """Fit all four states on the windows that fit-population rows point to.

    Only windows referenced by at least one fit row contribute. A window shared by
    a fit row and a validation row is fitted on once — using its own data, which is
    legitimate, because the *window's* observations are all dated at or before the
    cutoff for a fit row and the cutoff is shared.
    """
    import numpy as np  # local import keeps the module import light

    fit_mask = rows.population == "fit"
    if not fit_mask.any():
        raise ValueError(f"{rows.split_id}/{rows.stage}: no fit rows")
    fit_windows = np.unique(rows.row_window_index[fit_mask])
    fitted_on = (f"windows referenced by population=='fit' rows "
                 f"({int(fit_mask.sum())} rows, {len(fit_windows)} unique windows); "
                 f"policy={rows.policy}")

    dbs_values = windows.dbs_seq[fit_windows]
    dbs_masks = windows.dbs_mask[fit_windows] > 0
    coe_values = windows.coe_months_seq[fit_windows]
    coe_masks = windows.coe_mask[fit_windows] > 0

    return MacroScalers(
        dbs=fit_dbs_channels(dbs_values, dbs_masks, split_id=rows.split_id,
                             stage=rows.stage, policy=rows.policy, fitted_on=fitted_on),
        dbs_scalar=fit_scalars(windows.dbs_scalars[fit_windows],
                               windows.dbs_scalar_mask[fit_windows] > 0,
                               list(contract.DBS_SCALARS), "dbs_scalar",
                               split_id=rows.split_id, stage=rows.stage,
                               policy=rows.policy, fitted_on=fitted_on),
        coe=fit_coe_channels(coe_values, coe_masks, split_id=rows.split_id,
                             stage=rows.stage, policy=rows.policy, fitted_on=fitted_on),
        coe_scalar=fit_scalars(
            windows.coe_scalars[fit_windows].reshape(len(fit_windows), -1),
            windows.coe_scalar_mask[fit_windows].reshape(len(fit_windows), -1) > 0,
            [f"coe_{category}_{name}" for category in contract.COE_CATEGORIES
             for name in contract.COE_SCALARS],
            "coe_scalar", split_id=rows.split_id, stage=rows.stage,
            policy=rows.policy, fitted_on=fitted_on),
        contract_id=windows.contract_id,
        split_id=rows.split_id,
        stage=rows.stage,
        policy=rows.policy,
    )
