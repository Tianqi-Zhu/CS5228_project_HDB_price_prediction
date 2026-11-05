"""Causal four-level monthly rent panel and per-origin feature extraction.

Why prefix sums
---------------
The earlier implementation served rows inside the aggregation loop, and the
serve/absorb ordering silently produced a one-month-short aggregate.  Here the
monthly aggregates are stacked into arrays first and every origin query is a
plain prefix difference ``prefix[b] - prefix[a]``.  There is no loop ordering to
get wrong, so "inclusive of month ``o``, exclusive of ``o+1``" is enforced by
construction rather than by careful sequencing.

Information boundary
--------------------
* A training sample with target month ``t`` and horizon ``h`` has origin
  ``o = t - h`` and reads months ``0..o`` inclusive.
* A validation sample reads months ``0..cutoff`` inclusive at every horizon.
* Nothing after the boundary is ever read, and early months are never backfilled.

The panel holds only months ``<= fit_end`` where ``fit_end`` is the stage's own
last fitting month (``cutoff`` for stage B, ``cutoff - INNER_HOLDOUT_MONTHS`` for
stage A).  Stage-A holdout rows may have ``origin > fit_end``; their boundary is
clamped to ``fit_end`` so that inner-holdout labels cannot enter the stage-A
state.  That clamp applies to early stopping only, never to a reported score.

Aggregation is a *mean* (sum/count) with empirical-Bayes shrinkage across levels,
which is what both the tree and the network receive.  The plan requires identical
statistics for both, so there is exactly one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.fusion_contract_v2 import contract

UNSEEN = -1


def _key_series(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if not columns:
        return pd.Series(["__all__"] * len(frame), index=frame.index)
    first = frame[columns[0]].fillna("__missing__").astype(str)
    for column in columns[1:]:
        first = first.str.cat(frame[column].fillna("__missing__").astype(str), sep="|")
    return first


LEVEL_KEYS: dict[str, tuple[str, ...]] = {
    "global": (),
    "flat_type": ("FLAT_TYPE_CLEAN",),
    "town_ft": ("TOWN_CLEAN", "FLAT_TYPE_CLEAN"),
    "block_ft": ("BLOCK_ID", "FLAT_TYPE_CLEAN"),
}


@dataclass
class Panel:
    """Per-month and cumulative aggregates for every history level."""

    n_months: int                 # months 0..fit_end inclusive
    fit_end: int
    codes: dict[str, np.ndarray]  # level -> per-row code (UNSEEN allowed)
    sizes: dict[str, int]
    month_n: dict[str, np.ndarray]      # (size, n_months) counts
    month_sum: dict[str, np.ndarray]    # (size, n_months) rent sums
    month_sum_psm: dict[str, np.ndarray]
    cum_n: dict[str, np.ndarray]        # (size, n_months+1) prefix
    cum_sum: dict[str, np.ndarray]
    cum_sum_psm: dict[str, np.ndarray]
    cum_months: dict[str, np.ndarray]   # (size, n_months+1) observed months
    cum_last: dict[str, np.ndarray]     # (size, n_months+1) last observed month
    prior: dict[str, float]
    level_maps: dict[str, list]

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, frame: pd.DataFrame, fit_end: int, prior: dict[str, float] | None = None) -> "Panel":
        if fit_end < 0:
            raise ValueError("fit_end must be a non-negative month index")
        prior = dict(contract.PRIOR_ROWS if prior is None else prior)
        month = frame[contract.MONTH_COL].to_numpy(dtype=np.int64)
        fitting = month <= fit_end
        if not fitting.any():
            raise ValueError("no fitting rows for the panel")

        rent = frame[contract.TARGET].to_numpy(dtype=float)
        area = frame["FLOOR_AREA_SQM"].to_numpy(dtype=float)
        psm = np.divide(rent, area, out=np.full_like(rent, np.nan), where=area > 0)

        n_months = fit_end + 1
        codes: dict[str, np.ndarray] = {}
        sizes: dict[str, int] = {}
        level_maps: dict[str, list] = {}
        month_n, month_sum, month_sum_psm = {}, {}, {}
        cum_n, cum_sum, cum_sum_psm, cum_months, cum_last = {}, {}, {}, {}, {}

        for level, key_columns in LEVEL_KEYS.items():
            series = _key_series(frame, key_columns)
            values, uniques = pd.factorize(series, sort=True)
            code = np.asarray(values, dtype=np.int64)
            codes[level] = code
            size = len(uniques)
            sizes[level] = size
            level_maps[level] = [str(item) for item in uniques]

            counts = np.zeros((size, n_months))
            sums = np.zeros((size, n_months))
            sums_psm = np.zeros((size, n_months))
            for month_value in range(n_months):
                mask = fitting & (month == month_value)
                if not mask.any():
                    continue
                local = code[mask]
                counts[:, month_value] = np.bincount(local, minlength=size)
                finite = np.isfinite(rent[mask])
                sums[:, month_value] = np.bincount(
                    local[finite], weights=rent[mask][finite], minlength=size)
                finite_psm = np.isfinite(psm[mask])
                sums_psm[:, month_value] = np.bincount(
                    local[finite_psm], weights=psm[mask][finite_psm], minlength=size)

            month_n[level] = counts
            month_sum[level] = sums
            month_sum_psm[level] = sums_psm

            # Prefix arrays: index p covers months 0..p-1.
            pad = np.zeros((size, 1))
            c_n = np.concatenate([pad, np.cumsum(counts, axis=1)], axis=1)
            c_sum = np.concatenate([pad, np.cumsum(sums, axis=1)], axis=1)
            c_psm = np.concatenate([pad, np.cumsum(sums_psm, axis=1)], axis=1)
            observed = (counts > 0).astype(np.float64)
            c_months = np.concatenate([np.zeros((size, 1)), np.cumsum(observed, axis=1)], axis=1)
            last = np.full((size, n_months + 1), -1.0)
            running = np.full(size, -1.0)
            for p in range(n_months):
                running = np.where(counts[:, p] > 0, float(p), running)
                last[:, p + 1] = running

            cum_n[level] = c_n
            cum_sum[level] = c_sum
            cum_sum_psm[level] = c_psm
            cum_months[level] = c_months
            cum_last[level] = last

        return cls(
            n_months=n_months, fit_end=fit_end, codes=codes, sizes=sizes,
            month_n=month_n, month_sum=month_sum, month_sum_psm=month_sum_psm,
            cum_n=cum_n, cum_sum=cum_sum, cum_sum_psm=cum_sum_psm,
            cum_months=cum_months, cum_last=cum_last, prior=prior,
            level_maps=level_maps,
        )

    # -- scalar statistics at a boundary ---------------------------------
    def _window(self, level: str, code: np.ndarray, boundary: np.ndarray):
        """Prefix difference over months ``0..boundary`` (inclusive)."""
        valid = (code >= 0) & (boundary >= 0)
        safe_code = np.where(valid, code, 0)
        high = np.clip(boundary + 1, 0, self.n_months)
        low = np.zeros_like(high)
        n = self.cum_n[level][safe_code, high] - self.cum_n[level][safe_code, low]
        total = self.cum_sum[level][safe_code, high] - self.cum_sum[level][safe_code, low]
        total_psm = self.cum_sum_psm[level][safe_code, high] - self.cum_sum_psm[level][safe_code, low]
        months = self.cum_months[level][safe_code, high] - self.cum_months[level][safe_code, low]
        last = self.cum_last[level][safe_code, high]
        n = np.where(valid, n, 0.0)
        total = np.where(valid, total, 0.0)
        total_psm = np.where(valid, total_psm, 0.0)
        months = np.where(valid, months, 0.0)
        last = np.where(valid, last, -1.0)
        return n, total, total_psm, months, last

    def stats(self, boundary: np.ndarray, raw_codes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Smoothed rent/psm plus support at all four levels, plus the anchor.

        ``raw_codes`` maps level -> per-row code (already resolved against this
        panel's vocabularies; UNSEEN for new keys).
        """
        boundary = np.asarray(boundary, dtype=np.int64)
        if boundary.ndim != 1:
            raise ValueError(f"boundary must be 1-D, got shape {boundary.shape}")
        for level, array in raw_codes.items():
            array = np.asarray(array)
            if array.shape != boundary.shape:
                raise ValueError(
                    f"raw_codes[{level!r}] has shape {array.shape} but boundary has "
                    f"{boundary.shape}; both must describe the same rows")
        out: dict[str, np.ndarray] = {}

        n, total, total_psm, months, last = self._window("global", np.zeros(len(boundary), dtype=np.int64), boundary)
        global_rent = np.divide(total, n, out=np.full(len(boundary), np.nan), where=n > 0)
        global_psm = np.divide(total_psm, n, out=np.full(len(boundary), np.nan), where=n > 0)
        out["hist_rent__global"] = global_rent
        out["hist_psm__global"] = global_psm
        out["hist_n__global"] = n
        out["hist_months__global"] = months
        out["hist_recency__global"] = np.where(last >= 0, boundary - last, np.nan)
        out["hist_exact__global"] = (n > 0).astype(float)

        # Empirically-Bayes chain: flat_type -> town_ft -> block_ft, each shrinking
        # towards its already-smoothed parent.
        parent_rent = global_rent
        parent_psm = global_psm
        for level in ("flat_type", "town_ft", "block_ft"):
            code = raw_codes[level]
            n_l, total_l, total_psm_l, months_l, last_l = self._window(level, code, boundary)
            prior = self.prior.get(level, 10.0)
            rent_l = np.divide(total_l + prior * parent_rent, n_l + prior,
                               out=np.full(len(boundary), np.nan), where=(n_l + prior) > 0)
            psm_l = np.divide(total_psm_l + prior * parent_psm, n_l + prior,
                              out=np.full(len(boundary), np.nan), where=(n_l + prior) > 0)
            out[f"hist_rent__{level}"] = rent_l
            out[f"hist_psm__{level}"] = psm_l
            out[f"hist_n__{level}"] = n_l
            out[f"hist_months__{level}"] = months_l
            out[f"hist_recency__{level}"] = np.where(last_l >= 0, boundary - last_l, np.nan)
            out[f"hist_exact__{level}"] = (n_l > 0).astype(float)
            # A level with no observations of its own still inherits the parent
            # value through the prior; only use it as the next parent if finite.
            parent_rent = np.where(np.isfinite(rent_l), rent_l, parent_rent)
            parent_psm = np.where(np.isfinite(psm_l), psm_l, parent_psm)

        # ---- causal anchor: the finest *observed* level at the boundary ----
        anchor_rent = np.full(len(boundary), np.nan)
        anchor_psm = np.full(len(boundary), np.nan)
        anchor_n = np.zeros(len(boundary))
        anchor_level = np.full(len(boundary), -1, dtype=np.int64)
        for rank, level in enumerate(("block_ft", "town_ft", "flat_type", "global")):
            n_l = out[f"hist_n__{level}"]
            take = (anchor_level < 0) & (n_l > 0)
            anchor_rent = np.where(take, out[f"hist_rent__{level}"], anchor_rent)
            anchor_psm = np.where(take, out[f"hist_psm__{level}"], anchor_psm)
            anchor_n = np.where(take, n_l, anchor_n)
            anchor_level = np.where(take, rank, anchor_level)
        out["anchor_rent_sgd"] = anchor_rent
        out["anchor_psm_sgd"] = anchor_psm
        out["anchor_n"] = anchor_n
        out["anchor_level_code"] = anchor_level
        out["anchor_missing"] = (~np.isfinite(anchor_rent)).astype(float)
        return out

    # -- chronological tensors -------------------------------------------
    def sequences(self, origin: np.ndarray, raw_codes: dict[str, np.ndarray]):
        """(n_rows, n_levels, HISTORY_LEN, n_channels) plus a validity mask.

        Position ``k`` of the window is the absolute month
        ``origin - (HISTORY_LEN - 1) + k``.  Months before 2021-01 (negative
        index) or after ``fit_end`` are masked out with zeros, so padding can
        never be mistaken for observed history.
        """
        origin = np.asarray(origin, dtype=np.int64)
        if origin.ndim != 1:
            raise ValueError(f"origin must be 1-D, got shape {origin.shape}")
        for level, array in raw_codes.items():
            array = np.asarray(array)
            if array.shape != origin.shape:
                raise ValueError(
                    f"raw_codes[{level!r}] has shape {array.shape} but origin has "
                    f"{origin.shape}; both must describe the same rows")
        length = contract.HISTORY_LEN
        levels = contract.HISTORY_LEVELS
        channels = contract.PANEL_CHANNELS
        n = len(origin)
        seq = np.zeros((n, len(levels), length, len(channels)), dtype=np.float32)
        mask = np.zeros((n, len(levels), length), dtype=np.float32)

        offsets = np.arange(length)[None, :]
        months = origin[:, None] - (length - 1) + offsets          # (n, length)
        in_range = (months >= 0) & (months <= self.fit_end)

        for level_index, level in enumerate(levels):
            code = raw_codes[level]
            valid_level = (code >= 0)[:, None] & in_range
            safe_code = np.where(code >= 0, code, 0)
            safe_month = np.clip(months, 0, self.n_months - 1)
            counts = self.month_n[level][safe_code[:, None], safe_month]
            sums = self.month_sum[level][safe_code[:, None], safe_month]
            sums_psm = self.month_sum_psm[level][safe_code[:, None], safe_month]
            mean_rent = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            mean_psm = np.divide(sums_psm, counts, out=np.zeros_like(sums_psm), where=counts > 0)
            observed = (counts > 0) & valid_level
            phase = 2.0 * np.pi * np.mod(safe_month, 12) / 12.0

            block = np.stack([
                np.where(observed, np.log1p(np.maximum(mean_rent, 0.0)), 0.0),
                np.where(observed, np.log1p(np.maximum(mean_psm, 0.0)), 0.0),
                np.log1p(counts),
                np.sin(phase), np.cos(phase),
                observed.astype(np.float32),
            ], axis=-1)
            seq[:, level_index] = np.where(valid_level[..., None], block, 0.0).astype(np.float32)
            mask[:, level_index] = valid_level.astype(np.float32)
        return seq, mask
