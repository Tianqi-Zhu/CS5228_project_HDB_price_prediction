"""v4 temporal adapters: repack the contract's (levels, months, channels) window.

Two different readings of the same tensor, not interchangeable:

`legacy_48_step`
    v2's reading: month-major flatten of (month, level) into 48 successive steps. Four
    hierarchy levels become four successive timesteps even though they describe the same
    month. Kept so the padding defect can be reproduced against it.

`monthly_12_step`
    Four levels are *channels* of one month: 12 calendar steps, 24 channels each.
    Channel and level names are preserved so ordering is provable, not asserted.

Mask vs observed — these are NOT the same thing
-----------------------------------------------
The producer (`fusion_contract_v2/panel.py:275,294`) sets

    valid_level = (code >= 0) & in_range
    mask[:, level] = valid_level

so `sequence_mask` means **"this level is valid and its calendar month is inside the
panel window"** — it says nothing about whether a transaction happened. The transaction
flag is a separate channel: `PANEL_CHANNELS[5]`, written as
`observed = (counts > 0) & valid_level` (panel.py:283).

Consequently:

* `mask == 0`  -> genuine padding (month outside the panel, or invalid level). Dropped.
* `mask == 1, observed == 0` -> a **real calendar month with no transactions**. Retained
  as a timestep; reporting it as "has transactions" would be wrong.

`month_presence_report` therefore counts `observed` from the channel, never from the
keep-mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PANEL_START_MONTH = 0        # month index 0 is 2021-01 in the v2.1 contract
OBSERVED_CHANNEL_INDEX = 5   # panel.py:283 sets `observed` as the sixth channel


class AdapterError(ValueError):
    pass


@dataclass(frozen=True)
class WindowRep:
    """A repacked window.

    ``values``   (n, T, C) float32
    ``mask``     (n, T) float32, 1 = real timestep (valid level, calendar in range)
    ``observed`` (n, T) float32, 1 = at least one level actually saw a transaction
    ``kept``     (n, T) bool, True where the timestep enters the recurrence
    """

    values: np.ndarray
    mask: np.ndarray
    observed: np.ndarray
    kept: np.ndarray
    channel_names: list[str]
    step_names: list[str]
    kind: str
    calendar: np.ndarray | None = None


def _validate_order(sequence_order: list[str], n_levels: int, n_channels: int) -> None:
    if len(sequence_order) != n_levels * n_channels:
        raise AdapterError(
            f"sequence_order has {len(sequence_order)} entries but levels*channels = "
            f"{n_levels * n_channels}")
    for name in sequence_order:
        if "::" not in name:
            raise AdapterError(f"channel name {name!r} is not 'level::channel'")


def _names(sequence_order: list[str], n_levels: int, n_channels: int) -> list[str]:
    """`level::channel` for every level-major channel position."""
    _validate_order(sequence_order, n_levels, n_channels)
    return list(sequence_order)


def level_names(sequence_order: list[str], n_levels: int, n_channels: int) -> list[str]:
    _validate_order(sequence_order, n_levels, n_channels)
    return [sequence_order[level * n_channels].split("::", 1)[0]
            for level in range(n_levels)]


def channel_names(sequence_order: list[str], n_levels: int, n_channels: int) -> list[str]:
    """Per-level channel names (identical across levels), in order."""
    _validate_order(sequence_order, n_levels, n_channels)
    return [name.split("::", 1)[1] for name in sequence_order[:n_channels]]


# ---------------------------------------------------------------- legacy reading
def legacy_48_step(sequence: np.ndarray, mask: np.ndarray,
                   sequence_order: list[str]) -> WindowRep:
    """v2's month-major flatten: 48 steps of (month, level), 6 channels each.

    Here one step *is* one (month, level) cell, so its keep-mask and its observed flag
    come from the same cell.
    """
    if sequence.ndim != 4:
        raise AdapterError(f"sequence must be 4-D, got {sequence.shape}")
    n, n_levels, n_months, n_channels = sequence.shape
    if mask.shape != (n, n_levels, n_months):
        raise AdapterError(f"mask {mask.shape} != {(n, n_levels, n_months)}")

    ordered = np.transpose(sequence, (0, 2, 1, 3))            # n, months, levels, ch
    flat = ordered.reshape(n, n_months * n_levels, n_channels)
    flat_mask = np.transpose(mask, (0, 2, 1)).reshape(n, n_months * n_levels)

    steps = [f"m{month}/l{level}"
             for month in range(n_months) for level in range(n_levels)]
    # `ordered` is already (n, months, levels, channels), and `flat` flattened it to
    # (n, months*levels, channels), so the observed flag is simply its channel 5. An
    # earlier version transposed again here, which attached each flag to the wrong step.
    flat_observed = flat[..., OBSERVED_CHANNEL_INDEX]
    return WindowRep(values=flat.astype(np.float32),
                     mask=flat_mask.astype(np.float32),
                     observed=(flat_observed > 0).astype(np.float32),
                     kept=flat_mask > 0,
                     channel_names=_names(sequence_order, n_levels, n_channels),
                     step_names=steps, kind="legacy_48_step")


# ---------------------------------------------------------------- monthly reading
def monthly_12_step(sequence: np.ndarray, mask: np.ndarray, sequence_order: list[str],
                    origin_month: np.ndarray,
                    panel_start: int = PANEL_START_MONTH,
                    panel_end: int | None = None) -> WindowRep:
    """12 calendar steps, 24 channels each, pre-panel padding separated out.

    The calendar month of window slot `s` is `origin - (n_months - 1) + s`.

    `mask == 0` marks padding. An observation *outside* the panel calendar is a
    contradiction in the source data and raises rather than being retained, because
    keeping it would silently widen the available history.
    """
    if sequence.ndim != 4:
        raise AdapterError(f"sequence must be 4-D, got {sequence.shape}")
    n, n_levels, n_months, n_channels = sequence.shape
    if mask.shape != (n, n_levels, n_months):
        raise AdapterError(f"mask {mask.shape} != {(n, n_levels, n_months)}")
    origin_month = np.asarray(origin_month).reshape(-1)
    if origin_month.shape[0] != n:
        raise AdapterError("origin_month must have one entry per row")

    ordered = np.transpose(sequence, (0, 2, 1, 3)).reshape(
        n, n_months, n_levels * n_channels)
    level_kept = np.transpose(mask, (0, 2, 1)) > 0                          # (n,months,lv)

    offset = np.arange(n_months)[None, :] - (n_months - 1)
    calendar = origin_month[:, None] + offset                              # (n, months)
    in_panel = calendar >= panel_start
    if panel_end is not None:
        in_panel &= calendar <= int(panel_end)

    # observed comes from the channel, never from the keep-mask. `ordered` is
    # (n, months, levels*channels), so channel 5 of level L sits at L*n_channels + 5.
    observed_positions = [level * n_channels + OBSERVED_CHANNEL_INDEX
                          for level in range(n_levels)]
    observed_cells = ordered[:, :, observed_positions] > 0                  # (n,months,lv)
    observed_month = (observed_cells & level_kept).any(axis=2)

    contradiction = (~in_panel) & observed_month
    if contradiction.any():
        rows = np.argwhere(contradiction)
        first = rows[0]
        raise AdapterError(
            f"{int(contradiction.sum())} observed slot(s) fall outside the panel calendar "
            f"[{panel_start}, {panel_end}]; e.g. row {int(first[0])} slot {int(first[1])} "
            f"has calendar month {int(calendar[first[0], first[1]])}. Refusing to retain "
            f"observations the origin record says cannot exist.")

    kept = (level_kept.any(axis=2) | observed_month)
    # mask==0 must coincide with "outside the panel"; anything else would mean a valid
    # level was dropped for a month inside the panel
    mismatch = int(((~kept) != (~in_panel)).sum())
    if mismatch:
        raise AdapterError(
            f"{mismatch} slot(s) disagree between the source mask and the panel calendar; "
            f"mask==0 must mean 'outside the panel calendar'")

    steps = [f"slot{s}" for s in range(n_months)]
    return WindowRep(values=ordered.astype(np.float32),
                     mask=kept.astype(np.float32),
                     observed=observed_month.astype(np.float32),
                     kept=kept,
                     channel_names=_names(sequence_order, n_levels, n_channels),
                     step_names=steps, kind="monthly_12_step",
                     calendar=calendar.astype(np.int32))


def month_presence_report(rep: WindowRep) -> dict:
    """Separate padding from kept-but-unobserved months.

    `kept_unobserved_months` counts steps that are real timesteps but saw no
    transaction — the case the keep-mask deliberately retains, and the case a
    mask-based count would always report as zero.
    """
    if rep.calendar is None:
        raise AdapterError("month presence needs the monthly representation")
    kept = rep.kept
    observed = rep.observed > 0
    return {
        "padded_slots": int((~kept).sum()),
        "kept_slots": int(kept.sum()),
        "kept_unobserved_months": int((kept & ~observed).sum()),
        "observed_months": int((kept & observed).sum()),
        "rows_with_no_kept_month": int((~kept).all(axis=1).sum()),
        "calendar_min": int(rep.calendar[kept].min()) if kept.any() else None,
        "calendar_max": int(rep.calendar[kept].max()) if kept.any() else None,
    }
