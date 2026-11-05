"""Tensor assembly: causal panel + static features -> model batches.

A single code path serves training, inner early stopping and outer validation; the
sample frame passed in already carries ``MONTH_INDEX``/``HORIZON``/``ORIGIN``, so no
stage can silently reuse another stage's labels.

Target parameterisation
-----------------------
Rent levels drift upward across the whole sample, so a network given only
``APPROVAL_MONTH_INDEX`` must extrapolate a level it has never seen -- the probe run
did exactly that and diverged.  Every sample therefore carries a *causal origin
anchor*: the last observable median rent of its group at the origin month (for
training rows the origin is ``t - h``, for validation rows it is the split cutoff).
The network predicts the **residual** from that anchor, and predictions are rebuilt
as ``anchor + residual``.  All four models receive the same anchor, so the ablation
isolates whether the monthly *sequence* adds information beyond anchor + static
features rather than whether a model can extrapolate a trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONTEXT_DIM, TARGET, TARGET_SCALE
from panel import Panel, attach_months
from preprocessing import StaticPreprocessor, context_matrix


class Dataset:
    """numpy container with deterministic batching; the target is never a model input."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.arrays = arrays
        lengths = {len(v) for v in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"Ragged dataset: {[(k, len(v)) for k, v in arrays.items()]}")
        self.n = lengths.pop()

    def __len__(self) -> int:
        return self.n

    def subset(self, index: np.ndarray) -> "Dataset":
        return Dataset({key: value[index] for key, value in self.arrays.items()})

    def batches(self, batch_size: int, shuffle: bool, seed: int = 0, with_target: bool = False):
        import torch

        order = np.arange(self.n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, self.n, batch_size):
            index = order[start:start + batch_size]
            features = {key: torch.from_numpy(value[index])
                        for key, value in self.arrays.items()
                        if key not in ("target", "anchor_ksgd", "actual_sgd")}
            if with_target:
                yield features, torch.from_numpy(self.arrays["target"][index])
            else:
                yield features

    # -- target helpers ----------------------------------------------------
    def residual_to_sgd(self, raw: np.ndarray) -> np.ndarray:
        """Rebuild SGD predictions from the network's residual output."""
        return (np.asarray(raw, dtype=np.float64) + self.arrays["anchor_ksgd"]) * TARGET_SCALE

    def actual_sgd(self) -> np.ndarray:
        return self.arrays["actual_sgd"].astype(np.float64)


def assemble(frame: pd.DataFrame, prep: StaticPreprocessor, panel: Panel, with_history: bool) -> Dataset:
    """Build every tensor for one set of sample rows."""
    codes = panel.codes_for(frame)
    channels, mask, summary = panel.windows(codes, frame["ORIGIN"].to_numpy())

    anchor_ksgd = summary["anchor_rent_sgd"].astype(np.float32) / TARGET_SCALE
    actual = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=np.float64)

    arrays = {
        "numeric": prep.numeric_matrix(frame),
        "categorical": prep.categorical_matrix(frame),
        "context": context_matrix(frame["HORIZON"].to_numpy(), frame["MONTH_INDEX"].to_numpy(),
                                  summary, with_history),
        # residual target in kSGD; anchor is a causal input, never a target leak
        "target": (actual / TARGET_SCALE - anchor_ksgd).astype(np.float32),
        "anchor_ksgd": anchor_ksgd.astype(np.float32),
        "actual_sgd": actual.astype(np.float32),
        "history_months": summary["history_months"].astype(np.float32),
    }
    if with_history:
        arrays["panel"] = channels
        arrays["panel_mask"] = mask
    return Dataset(arrays)


def history_stats(dataset: Dataset) -> dict:
    months = dataset.arrays["history_months"]
    return {
        "rows": int(len(months)),
        "mean_history_months": float(months.mean()),
        "rows_lt_6_months": int((months < 6).sum()),
        "rows_zero_history": int((months == 0).sum()),
    }


def load_frame(path) -> pd.DataFrame:
    return attach_months(pd.read_csv(path))


def split_frames(frame: pd.DataFrame, cutoff_month: int, val_end_month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split honouring BOTH the cutoff and the declared validation end."""
    if val_end_month <= cutoff_month:
        raise ValueError("val_end_month must be after cutoff_month")
    train = frame.loc[frame["MONTH_INDEX"] <= cutoff_month].copy()
    valid = frame.loc[(frame["MONTH_INDEX"] > cutoff_month) & (frame["MONTH_INDEX"] <= val_end_month)].copy()
    if train.empty or valid.empty:
        raise ValueError("Empty temporal split")
    if train["MONTH_INDEX"].max() >= valid["MONTH_INDEX"].min():
        raise ValueError("Split boundaries overlap")
    return train, valid


def split_spec(split_id: str) -> dict:
    from config import SPLITS

    return SPLITS[split_id]


def split_bounds(split_id: str) -> tuple[int, int]:
    from config import SPLITS, month_index

    spec = SPLITS[split_id]
    return month_index(spec["cutoff"]), month_index(spec["val_end"])
