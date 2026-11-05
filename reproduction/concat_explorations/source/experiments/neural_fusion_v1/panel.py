"""Causal monthly panel of shared ``town x flat-type`` rental histories.

Design constraints enforced here (docs/plans/2026-10-24-parallel-rent-experiments.md):

* A panel is always built with an explicit fitting cutoff ``c``.  Only rows with
  ``approval month <= c`` contribute to any aggregate, so no validation-period
  rental label can reach a feature at any horizon.
* Each sample keeps a *sequence of monthly group aggregates*, never a sequence of
  arbitrary transaction rows: 150k rows are not treated as 150k market histories.
* For a training row at month ``t`` and horizon ``h`` the window ends at origin
  ``o = t - h`` and covers ``o-H+1 .. o``, so the row's own month and every later
  label stay outside the input.
* For validation rows the window always ends at ``c`` regardless of ``h``.
* Channel standardisation uses only months ``<= c - HISTORY_LEN`` so the inner
  early-stopping holdout cannot be described by its own statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    FALLBACK_MIN_COUNT,
    HISTORY_LEN,
    KEY_COLS,
    PANEL_CHANNELS,
    STANDARDISED_CHANNELS,
    TARGET,
)

LEVEL_NAMES = ["exact_group", "town", "flat_type", "global"]
UNSEEN = -1


@dataclass
class Panel:
    """Monthly aggregates for four fallback levels, restricted to months <= cutoff."""

    n: list[np.ndarray]            # per level: (n_codes, n_months) transaction counts
    rent: list[np.ndarray]         # median monthly rent
    psm: list[np.ndarray]          # median rent per square metre
    n_months: int
    cutoff_month: int
    level_maps: list = field(default_factory=list, repr=False)
    channel_mean: np.ndarray = field(default=None, repr=False)
    channel_std: np.ndarray = field(default=None, repr=False)
    scale_until_month: int = -1

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, frame: pd.DataFrame, cutoff_month: int) -> "Panel":
        fit = frame.loc[frame["MONTH_INDEX"] <= cutoff_month]
        if fit.empty:
            raise ValueError("Panel needs at least one fitting month")
        n_months = cutoff_month + 1
        fit = fit.assign(PSM=fit[TARGET] / fit["FLOOR_AREA_SQM"].replace(0, np.nan))

        specs = [
            (fit.assign(_k=fit[KEY_COLS[0]] + "|" + fit[KEY_COLS[1]]), "_k"),
            (fit, KEY_COLS[0]),
            (fit, KEY_COLS[1]),
            (fit.assign(_k="__all__"), "_k"),
        ]
        n_tables, rent_tables, psm_tables, level_maps = [], [], [], []
        for source, key in specs:
            grouped = source.groupby([key, "MONTH_INDEX"], observed=True).agg(
                n=(TARGET, "size"), rent=(TARGET, "median"), psm=("PSM", "median")
            )
            categories = sorted(grouped.index.get_level_values(0).unique())
            mapping = {name: i for i, name in enumerate(categories)}
            level_maps.append(mapping)
            shape = (len(categories), n_months)
            n_tab = np.zeros(shape, dtype=np.float32)
            rent_tab = np.full(shape, np.nan, dtype=np.float32)
            psm_tab = np.full(shape, np.nan, dtype=np.float32)
            idx = grouped.index
            rows = np.fromiter((mapping[k] for k in idx.get_level_values(0)), dtype=np.int32, count=len(idx))
            cols = idx.get_level_values(1).to_numpy(dtype=np.int32)
            n_tab[rows, cols] = grouped["n"].to_numpy(dtype=np.float32)
            rent_tab[rows, cols] = grouped["rent"].to_numpy(dtype=np.float32)
            psm_tab[rows, cols] = grouped["psm"].to_numpy(dtype=np.float32)
            n_tables.append(n_tab)
            rent_tables.append(rent_tab)
            psm_tables.append(psm_tab)

        panel = cls(n=n_tables, rent=rent_tables, psm=psm_tables,
                    n_months=n_months, cutoff_month=cutoff_month, level_maps=level_maps)
        panel._fit_standardisation()
        return panel

    # -- per-row level codes ----------------------------------------------
    def codes_for(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Level codes for arbitrary rows.

        A category that never appeared during fitting maps to ``UNSEEN`` instead of
        raising, so ``windows`` can fall back to a coarser level.  This is the
        documented behaviour for genuinely new town x flat-type combinations.
        """
        exact = (frame[KEY_COLS[0]] + "|" + frame[KEY_COLS[1]]).to_numpy()
        return {
            "exact_group": _map(exact, self.level_maps[0]),
            "town": _map(frame[KEY_COLS[0]].to_numpy(), self.level_maps[1]),
            "flat_type": _map(frame[KEY_COLS[1]].to_numpy(), self.level_maps[2]),
            "global": np.zeros(len(frame), dtype=np.int32),
        }

    def unseen_report(self, frame: pd.DataFrame) -> dict:
        exact = (frame[KEY_COLS[0]] + "|" + frame[KEY_COLS[1]]).to_numpy()
        town = frame[KEY_COLS[0]].to_numpy()
        flat = frame[KEY_COLS[1]].to_numpy()
        return {
            "rows": int(len(frame)),
            "unseen_exact_group": int(sum(v not in self.level_maps[0] for v in exact)),
            "unseen_town": int(sum(v not in self.level_maps[1] for v in town)),
            "unseen_flat_type": int(sum(v not in self.level_maps[2] for v in flat)),
        }

    # -- standardisation --------------------------------------------------
    def _fit_standardisation(self) -> None:
        k = len(STANDARDISED_CHANNELS)
        self.channel_mean = np.zeros(k, dtype=np.float32)
        self.channel_std = np.ones(k, dtype=np.float32)
        stop = max(self.cutoff_month - HISTORY_LEN, 0) + 1     # exclusive
        for j, name in enumerate(STANDARDISED_CHANNELS):
            values = {
                "log_median_rent": np.log1p(self.rent[0]),
                "log_median_psm": np.log1p(self.psm[0]),
                "log_count": np.log1p(self.n[0]),
            }[name][:, :stop]
            observed = values[self.n[0][:, :stop] > 0]
            if observed.size < 2:
                continue
            self.channel_mean[j] = float(np.nanmean(observed))
            self.channel_std[j] = float(np.nanstd(observed)) or 1.0
        self.scale_until_month = stop - 1

    # -- window extraction -------------------------------------------------
    def windows(self, codes: dict[str, np.ndarray], end_month: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (channels, mask, summary) for one window per sample.

        Windows end at ``end_month`` and are left-aligned: a short history occupies
        the leading slots and padding trails, so chronological order is preserved
        and ``run_lstm`` can use ``pack_padded_sequence`` directly.  Padding is
        always encoded as zeros and never as a real aggregate.
        """
        n_samples = len(end_month)
        start = np.maximum(0, end_month - (HISTORY_LEN - 1))
        pos = np.arange(HISTORY_LEN)[None, :]
        months = start[:, None] + pos
        month_valid = months <= end_month[:, None]
        clipped = np.clip(months, 0, self.n_months - 1)

        chosen = np.full(months.shape, len(LEVEL_NAMES) - 1, dtype=np.int8)
        n_choice = np.zeros(months.shape, dtype=np.float32)
        for level in (2, 1, 0):            # ascending priority: flat -> town -> exact
            level_codes = codes[LEVEL_NAMES[level]]
            valid_code = level_codes >= 0
            safe = np.where(valid_code, level_codes, 0)
            n_level = self.n[level][safe[:, None], clipped]
            usable = month_valid & valid_code[:, None] & (n_level >= FALLBACK_MIN_COUNT)
            chosen[usable] = level
            n_choice[usable] = n_level[usable]
        base = month_valid & (n_choice == 0)
        if base.any():                      # nothing passed the floor: use global counts
            n_choice[base] = self.n[3][0, clipped[base]]
        mask = month_valid & (n_choice > 0)

        rent = np.zeros(months.shape, dtype=np.float32)
        psm = np.zeros(months.shape, dtype=np.float32)
        for level in range(len(LEVEL_NAMES)):
            sel = chosen == level
            if not sel.any():
                continue
            level_codes = codes[LEVEL_NAMES[level]]
            safe = np.where(level_codes >= 0, level_codes, 0)
            rent[sel] = self.rent[level][safe[:, None], clipped][sel]
            psm[sel] = self.psm[level][safe[:, None], clipped][sel]

        channels = np.zeros(months.shape + (len(PANEL_CHANNELS),), dtype=np.float32)
        month_clipped = np.clip(months, 0, None)
        channels[..., 0] = np.log1p(rent)
        channels[..., 1] = np.log1p(psm)
        channels[..., 2] = np.log1p(n_choice)
        channels[..., 3] = chosen / (len(LEVEL_NAMES) - 1)
        channels[..., 4] = np.sin(2 * np.pi * (month_clipped % 12) / 12)
        channels[..., 5] = np.cos(2 * np.pi * (month_clipped % 12) / 12)
        channels[..., 6] = (pos + 1) / HISTORY_LEN
        channels[..., 7] = (chosen == 0).astype(np.float32)
        for j, name in enumerate(STANDARDISED_CHANNELS):
            k = PANEL_CHANNELS.index(name)
            channels[..., k] = (channels[..., k] - self.channel_mean[j]) / self.channel_std[j]
            channels[..., k] *= mask
        channels[~mask] = 0.0

        valid_counts = mask.sum(axis=1)
        last = np.clip(valid_counts - 1, 0, HISTORY_LEN - 1)
        rows = np.arange(n_samples)
        any_valid = valid_counts > 0
        anchor_rent = np.where(any_valid, rent[rows, last], 0.0)
        anchor_psm = np.where(any_valid, psm[rows, last], 0.0)
        anchor_level = np.where(any_valid, chosen[rows, last], len(LEVEL_NAMES) - 1)
        summary = {
            "history_months": valid_counts.astype(np.int16),
            "mean_log_count": np.where(any_valid, channels[..., 2].sum(axis=1) / np.maximum(valid_counts, 1), 0.0).astype(np.float32),
            "exact_frac": np.where(any_valid, (channels[..., 7] * mask).sum(axis=1) / np.maximum(valid_counts, 1), 0.0).astype(np.float32),
            # last observable group level at the origin, in original units
            "anchor_rent_sgd": anchor_rent.astype(np.float32),
            "anchor_psm_sgd": anchor_psm.astype(np.float32),
            "anchor_level": anchor_level.astype(np.int8),
            "anchor_missing": (~any_valid).astype(np.float32),
        }
        return channels, mask.astype(np.float32), summary


def _map(values: np.ndarray, mapping: dict) -> np.ndarray:
    return np.array([mapping.get(v, UNSEEN) for v in values], dtype=np.int32)


def attach_months(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m", errors="raise")
    out = frame.copy()
    out["MONTH_INDEX"] = ((parsed.dt.year - 2021) * 12 + parsed.dt.month - 1).to_numpy(dtype=np.int32)
    return out


def training_samples(frame: pd.DataFrame, cutoff_month: int, rng: np.random.Generator, max_horizon: int) -> pd.DataFrame:
    """Fitting-period rows with a sampled horizon and the origin it implies."""
    fit = frame.loc[frame["MONTH_INDEX"] <= cutoff_month].copy()
    horizons = rng.integers(1, max_horizon + 1, size=len(fit))
    fit["HORIZON"] = horizons
    fit["ORIGIN"] = fit["MONTH_INDEX"] - fit["HORIZON"]
    return fit.loc[fit["ORIGIN"] >= 0]


def validation_samples(frame: pd.DataFrame, cutoff_month: int, val_end_month: int | None = None) -> pd.DataFrame:
    """Validation rows; every history window is frozen at ``cutoff_month``."""
    upper = cutoff_month if val_end_month is None else val_end_month
    valid = frame.loc[
        (frame["MONTH_INDEX"] > cutoff_month) & (frame["MONTH_INDEX"] <= upper)
    ].copy()
    valid["HORIZON"] = valid["MONTH_INDEX"] - cutoff_month
    valid["ORIGIN"] = cutoff_month
    return valid


def save_panel_state(panel: "Panel", directory, name: str) -> dict:
    """Persist the fitted aggregates so checkpoint inference is reproducible."""
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_months": panel.n_months,
        "cutoff_month": panel.cutoff_month,
        "levels": LEVEL_NAMES,
        "level_maps": [{str(k): int(v) for k, v in mapping.items()} for mapping in panel.level_maps],
        "channel_mean": panel.channel_mean.tolist(),
        "channel_std": panel.channel_std.tolist(),
        "scale_until_month": int(panel.scale_until_month),
        "fallback_min_count": FALLBACK_MIN_COUNT,
        "history_len": HISTORY_LEN,
        "unseen_policy": "unseen category -> UNSEEN code -> finer level skipped, coarser level used",
    }
    arrays = {}
    for level_name in LEVEL_NAMES:
        idx = LEVEL_NAMES.index(level_name)
        arrays[f"n_{level_name}"] = panel.n[idx]
        arrays[f"rent_{level_name}"] = panel.rent[idx]
        arrays[f"psm_{level_name}"] = panel.psm[idx]
    np.savez_compressed(directory / f"{name}.npz", **arrays)
    with open(directory / f"{name}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def load_panel_state(directory, name: str) -> "Panel":
    import json
    from pathlib import Path

    directory = Path(directory)
    with open(directory / f"{name}.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    stored = np.load(directory / f"{name}.npz")
    panel = Panel(
        n=[stored[f"n_{lvl}"] for lvl in LEVEL_NAMES],
        rent=[stored[f"rent_{lvl}"] for lvl in LEVEL_NAMES],
        psm=[stored[f"psm_{lvl}"] for lvl in LEVEL_NAMES],
        n_months=int(payload["n_months"]),
        cutoff_month=int(payload["cutoff_month"]),
        level_maps=[{k: int(v) for k, v in m.items()} for m in payload["level_maps"]],
        scale_until_month=int(payload.get("scale_until_month", -1)),
    )
    panel.channel_mean = np.array(payload["channel_mean"], dtype=np.float32)
    panel.channel_std = np.array(payload["channel_std"], dtype=np.float32)
    return panel
