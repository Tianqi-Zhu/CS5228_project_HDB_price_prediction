"""Causal recent-rent features from an observed monthly panel.

Everything is built from months `<= origin`, where `origin` is the sample's own origin
month index. `build_panel` refuses a frame containing any month beyond `fit_end`, and
`features_for` refuses an origin beyond it, so a future label cannot enter a feature by
construction rather than by care.

Two statistics must never be confused:

* **cumulative** — mean over every month from the panel start to the origin. That is what
  the frozen v2 history block already provides (`panel._window` with `low = zeros`).
* **rolling** — mean over the last 3 / 6 / 12 calendar months ending at the origin. That is
  what this module builds. In a moving market the two differ a lot, which is the whole
  point of the experiment.

Windows are **calendar months**, not the last N transactions. An empty month stays
distinguishable from a real zero: `n3`/`n12` count transactions, `valid6` counts months
with any observation, and `fallback` records whether the value came from this group, a
parent level, or nowhere.

Levels are rent per square metre, because a raw rent mean moves with transaction mix (unit
size, flat type, town) as well as price. Per-sqm plus stratification reduces that but does
not remove quality/area/location effects; the honest label is "group-level proxy", never
"market index".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEVELS = ("global", "flat_type", "town_ft")
WINDOWS = (3, 6, 12)

# fallback provenance
FALLBACK_GROUP = 0.0
FALLBACK_PARENT = 1.0
FALLBACK_MISSING = 2.0


class MarketDynamicsError(ValueError):
    pass


@dataclass(frozen=True)
class LevelTable:
    """Per-group, per-month sums for one hierarchy level."""

    level: str
    keys: list                       # original key tuples, index == group index
    index_of: dict                   # key tuple -> group index
    count: np.ndarray                # (n_groups, n_months)
    sum_rent: np.ndarray
    sum_psm: np.ndarray
    # Per-cell aggregates for fixed-composition means, present only when a cell column was
    # supplied. A fixed-composition index must average WITHIN each cell first and only then
    # combine cells with fixed shares; multiplying every transaction by a share instead
    # leaves the current transaction proportions inside the average (see
    # docs/reviews/2026-10-26-fixed-composition-audit.json).
    cell_count: np.ndarray | None = None       # (n_groups, n_cells, n_months)
    cell_sum_psm: np.ndarray | None = None     # (n_groups, n_cells, n_months)
    cell_index_of: dict | None = None          # cell key -> cell index

    @property
    def n_groups(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class Panel:
    months: int
    fit_end: int
    levels: dict                     # level -> LevelTable
    parents: dict = field(default_factory=dict)   # (child_level, key) -> parent key
    # Cumulative transaction counts per cell, (n_months, n_cells). Row `origin` holds the
    # count over months 0..origin only, so a share vector derived from it can never read a
    # month after the origin. Fixed-composition weights are defined from this, which is why
    # an origin before the base window still gets a causal, well-defined share vector.
    cell_cumulative: np.ndarray | None = None
    cell_keys: list | None = None


@dataclass(frozen=True)
class FeatureSpec:
    levels: tuple = LEVELS
    windows: tuple = WINDOWS

    def names(self) -> list[str]:
        """Feature order. The count is derived, never asserted against a magic number:
        the trailing/fixed fields plus two provenance codes per level."""
        out = []
        for level in self.levels:
            for window in self.windows:
                out.append(f"recent{window}_logpsm__{level}")
            out.append(f"recent3_over_prior3_log__{level}")
            out.append(f"slope6_logpsm__{level}")
            out.append(f"n3__{level}")
            out.append(f"n12__{level}")
            out.append(f"valid6__{level}")
            out.append(f"fallback__{level}")
            # which level actually supplied each compound statistic; a ratio or slope
            # that silently mixed bases would otherwise be invisible
            out.append(f"ratio_source__{level}")
            out.append(f"slope_source__{level}")
        return out


def build_panel(frame, *, level_keys: dict, rent_col: str, area_col: str,
                month_col: str, fit_end: int, parents: dict | None = None,
                cell_column: str | None = None) -> Panel:
    """Aggregate observed rows into per-month, per-group sums.

    `level_keys` maps level name -> list of key tuples, one per row (higher levels repeat
    the parent key for every row, which is what makes the parent lookup exact rather than
    guessed). `frame` must already exclude months after `fit_end`.
    """
    months_present = frame[month_col].to_numpy()
    if months_present.size and int(months_present.max()) > fit_end:
        raise MarketDynamicsError(
            f"panel frame contains month {int(months_present.max())} > fit_end {fit_end}; "
            "future observations must be excluded before aggregation")

    n_months = int(fit_end) + 1
    rent = frame[rent_col].to_numpy(dtype=np.float64)
    area = frame[area_col].to_numpy(dtype=np.float64)
    if np.any(area <= 0):
        raise MarketDynamicsError("non-positive floor area in the panel frame")
    psm = rent / area
    month = frame[month_col].to_numpy(dtype=np.int64)

    levels = {}
    for level, keys_per_row in level_keys.items():
        unique = sorted(set(keys_per_row))
        index_of = {key: i for i, key in enumerate(unique)}
        encoded = np.fromiter((index_of[key] for key in keys_per_row),
                              dtype=np.int64, count=len(keys_per_row))
        shape = (len(unique), n_months)
        count = np.zeros(shape)
        sum_rent = np.zeros(shape)
        sum_psm = np.zeros(shape)
        np.add.at(count, (encoded, month), 1.0)
        np.add.at(sum_rent, (encoded, month), rent)
        np.add.at(sum_psm, (encoded, month), psm)

        cell_count = cell_sum_psm = cell_index_of = None
        if cell_column is not None:
            cell_values = frame[cell_column].astype(str).to_numpy()
            cell_keys = sorted(set(cell_values))
            cell_index_of = {key: i for i, key in enumerate(cell_keys)}
            cell_encoded = np.fromiter((cell_index_of[v] for v in cell_values),
                                       dtype=np.int64, count=len(cell_values))
            # Counts and per-sqm sums per (group, cell, month). A fixed-composition mean is
            # built from these by averaging inside each cell first.
            cell_count = np.zeros((len(unique), len(cell_keys), n_months))
            cell_sum_psm = np.zeros((len(unique), len(cell_keys), n_months))
            np.add.at(cell_count, (encoded, cell_encoded, month), 1.0)
            np.add.at(cell_sum_psm, (encoded, cell_encoded, month), psm)

        levels[level] = LevelTable(level=level, keys=unique, index_of=index_of,
                                   count=count, sum_rent=sum_rent, sum_psm=sum_psm,
                                   cell_count=cell_count, cell_sum_psm=cell_sum_psm,
                                   cell_index_of=cell_index_of)

    # Cumulative per-cell counts, one row per origin. Yearly cumulative is causal by
    # construction: row `origin` only sums months 0..origin.
    cell_cumulative = cell_keys_out = None
    if cell_column is not None:
        cell_values = frame[cell_column].astype(str).to_numpy()
        cell_keys_out = sorted(set(cell_values))
        cell_encoded = np.fromiter(
            ({k: i for i, k in enumerate(cell_keys_out)}[v] for v in cell_values),
            dtype=np.int64, count=len(cell_values))
        per_month = np.zeros((n_months, len(cell_keys_out)))
        np.add.at(per_month, (month, cell_encoded), 1.0)
        cell_cumulative = np.cumsum(per_month, axis=0)

    return Panel(months=n_months, fit_end=int(fit_end), levels=levels,
                 parents=dict(parents or {}),
                 cell_cumulative=cell_cumulative, cell_keys=cell_keys_out)


def cell_shares(panel: Panel, origin: int) -> dict | None:
    """Composition shares from months 0..origin only, i.e. causal at this origin.

    Defined per origin rather than as one global base vector, because a sample whose origin
    falls before the base window would otherwise have no legal weights — and taking the base
    window's counts would read months after that origin. The share vector is then held
    fixed across all comparison windows at that origin.
    """
    if panel.cell_cumulative is None:
        return None
    counts = panel.cell_cumulative[max(0, min(origin, panel.months - 1))]
    total = float(counts.sum())
    if total <= 0:
        return None
    return {key: float(counts[i]) / total for i, key in enumerate(panel.cell_keys)}


def _fixed_value(table: LevelTable, group: int, origin: int, window: int,
                 shares: dict | None):
    """Fixed-composition mean: average within each cell, then combine cells with `q`.

    This is the corrected estimator. Multiplying transactions by a share (the earlier
    implementation) leaves the current transaction proportions inside the average, so a
    market whose mix moves still moves the index even when every cell's price is constant.

    Cells with no transaction in the window contribute nothing, and the shares of the
    absent cells are redistributed over the cells that did trade — the coverage loss is
    returned so callers can report it rather than hide it.
    """
    if shares is None or table.cell_count is None:
        raise MarketDynamicsError(
            "fixed composition needs a cell column at build_panel time")
    low = max(0, origin - window + 1)
    high = origin + 1
    if low >= high:
        return None, 0.0, 0, 0.0
    cell_counts = table.cell_count[group, :, low:high].sum(axis=1)     # (n_cells,)
    cell_sums = table.cell_sum_psm[group, :, low:high].sum(axis=1)
    present = cell_counts > 0
    if not present.any():
        return None, 0.0, 0, 0.0

    cells = table.cell_index_of.keys() if table.cell_index_of else []
    q = np.zeros(len(cell_counts))
    for cell, idx in (table.cell_index_of or {}).items():
        q[idx] = shares.get(cell, 0.0)
    q_present = q * present
    denom = float(q_present.sum())
    covered_fraction = float(q_present.sum())
    if denom <= 0:
        # no cell with support carries a positive share: fall back to equal weights over
        # the cells that did trade, and report zero coverage so the caller sees it
        means = cell_sums[present] / cell_counts[present]
        return float(means.mean()), float(cell_counts.sum()), int(present.sum()), 0.0
    means = cell_sums[present] / cell_counts[present]
    value = float((q_present[present] * means).sum() / denom)
    return value, float(cell_counts.sum()), int(present.sum()), covered_fraction


def _window(table: LevelTable, group: int, origin: int, window: int,
            weighting: str = "transactions"):
    """Count, per-sqm sum and observed-month count over `window` months ending at origin.

    With `weighting="fixed"` the numerator and denominator are the base-mix-weighted sums,
    so the returned mean is a fixed-composition average rather than a transaction average.
    """
    low = max(0, origin - window + 1)
    high = origin + 1
    if low >= high:
        return 0.0, 0.0, 0, 0.0
    counts = table.count[group, low:high]
    valid = int((counts > 0).sum())
    return float(counts.sum()), float(table.sum_psm[group, low:high].sum()), valid, 0.0


def _value(table: LevelTable, group: int, origin: int, window: int,
           shares: dict | None = None):
    """Window mean. `shares` None = transaction-weighted, dict = fixed composition."""
    if shares is not None:
        return _fixed_value(table, group, origin, window, shares)[:3]
    n, total_psm, valid, _weight = _window(table, group, origin, window, "transactions")
    if n <= 0 or total_psm <= 0:
        return None, n, valid
    return total_psm / n, n, valid


SOURCE_CODE = {"global": 0.0, "flat_type": 1.0, "town_ft": 2.0}


def _resolve(panel: Panel, level: str, key, origin: int, window: int,
             shares: dict | None = None):
    """Group value first, then an explicit parent cascade, keeping provenance."""
    table = panel.levels[level]
    group = table.index_of.get(key)
    if group is not None:
        value, n, valid = _value(table, group, origin, window, shares)
        if value is not None:
            return value, n, valid, FALLBACK_GROUP, level

    parent = panel.parents.get((level, key))
    if parent is not None:
        for parent_level in ("flat_type", "global"):
            parent_table = panel.levels.get(parent_level)
            if parent_table is None:
                continue
            parent_key = parent.get(parent_level)
            if parent_key is None:
                continue
            parent_group = parent_table.index_of.get(parent_key)
            if parent_group is None:
                continue
            value, n, valid = _value(parent_table, parent_group, origin, window, shares)
            if value is not None:
                return value, n, valid, FALLBACK_PARENT, parent_level
    return None, 0.0, 0, FALLBACK_MISSING, None


def level_and_slope(panel: Panel, level: str, key, origin: int,
                    shares: dict | None = None):
    """Trailing-3 per-sqm level and six-month slope measured on **one shared basis**.

    Direction 2 needs a level and a slope that describe the *same* series, because the
    forecast multiplies them: `psm * exp(slope * h * damp)`. If the level came from the
    group and the slope from a parent, the product mixes two populations and the shift
    would include a level gap that has nothing to do with time.

    The basis is chosen once, most specific first, and must support BOTH statistics: a
    trailing-3 level (at least one transaction) and a fittable slope (at least three
    observed months in the trailing six).

    Returns `(level_psm, slope_per_month, basis_level, valid_months, level_n)`; when no
    basis supports both, `(None, nan, None, 0, 0.0)` — never a partial pair.
    """
    for basis_level, basis_key in _basis_candidates(panel, level, key):
        table = panel.levels.get(basis_level)
        if table is None:
            continue
        group = table.index_of.get(basis_key)
        if group is None:
            continue
        level_value, level_n, _valid = _value(table, group, origin, 3, shares)
        if level_value is None:
            continue
        slope, valid_months, slope_source, _n = _slope6(
            panel, basis_level, basis_key, origin, shares)
        if not np.isfinite(slope) or slope_source != basis_level:
            continue
        return level_value, slope, basis_level, valid_months, level_n
    return None, float("nan"), None, 0, 0.0


def _slope6(panel: Panel, level: str, key, origin: int,
            shares: dict | None = None):
    """Least-squares slope of log(monthly per-sqm) over the trailing six months.

    **One basis for the whole series.** Resolving each month independently would let the
    early months come from the group and the later months from a parent; the level gap
    between those two populations would then be read as a trend, and recording "the last
    level used" would not repair it. So a single basis is chosen up front: the most
    specific level that observes at least three of the six months. Otherwise the entire
    series drops to the next coarser level and the test is repeated there. Missing months
    inside the chosen basis are skipped and the regression uses the real time gaps, so a
    series observed in months 0, 1, 5 is fitted on those three x-positions, not on 0, 1, 2.

    Returns `(slope, valid_months, source_level, n_transactions)`; fewer than three
    observed months at every level gives `(nan, 0, None, 0.0)`.
    """
    for basis_level, basis_key in _basis_candidates(panel, level, key):
        table = panel.levels.get(basis_level)
        if table is None:
            continue
        group = table.index_of.get(basis_key)
        if group is None:
            continue
        xs, ys, n_total = [], [], 0.0
        for offset in range(5, -1, -1):
            month = origin - offset
            if month < 0:
                continue
            # a single-month window at that month, resolved under the same weighting
            value, count, _valid = _value(table, group, month, 1, shares)
            if value is None or count <= 0 or value <= 0:
                continue
            xs.append(float(origin - month))
            ys.append(float(np.log(value)))
            n_total += count
        if len(xs) < 3:
            continue
        return (float(np.polyfit(-np.asarray(xs), np.asarray(ys), 1)[0]),
                len(xs), basis_level, n_total)
    return float("nan"), 0, None, 0.0


def _basis_candidates(panel: Panel, level: str, key) -> list:
    """Candidate bases from most to least specific: this group, then its parents."""
    candidates = [(level, key)]
    parent = panel.parents.get((level, key), {})
    for parent_level in ("town_ft", "flat_type", "global"):
        if parent_level == level:
            continue
        parent_key = parent.get(parent_level)
        if parent_key is not None:
            candidates.append((parent_level, parent_key))
    return candidates


def _shared_pair(panel: Panel, level: str, key, origin: int,
                 shares: dict | None = None, window: int = 3):
    """Trailing-3 and prior-3 measured on **one** shared basis.

    Resolving each interval independently lets the recent window come from the group while
    the prior window falls back to a parent. The ratio then mixes two different populations
    and reads the level gap as a price change. So the basis is chosen once — most specific
    level where *both* intervals have support — and both intervals are measured there. If
    no level supports both, the ratio is missing and says so.

    Returns `(recent_value, prior_value, basis_level, recent_n, prior_n)`.
    """
    prior_origin = origin - window
    for basis_level, basis_key in _basis_candidates(panel, level, key):
        table = panel.levels.get(basis_level)
        if table is None:
            continue
        group = table.index_of.get(basis_key)
        if group is None:
            continue
        recent, recent_n, _ = _value(table, group, origin, window, shares)
        if recent is None:
            continue
        if prior_origin < 0:
            continue
        prior, prior_n, _ = _value(table, group, prior_origin, window, shares)
        if prior is None:
            continue
        return recent, prior, basis_level, recent_n, prior_n
    return None, None, None, 0.0, 0.0


def features_for(panel: Panel, origin: int, *, row_keys: dict,
                 spec: FeatureSpec = FeatureSpec(),
                 fixed: bool = False) -> np.ndarray:
    """Feature vector for one row. Only months <= origin are read."""
    if origin > panel.fit_end:
        raise MarketDynamicsError(
            f"origin {origin} is beyond the panel fit_end {panel.fit_end}; "
            "features would require future observations")
    # one causal share vector per origin, held fixed across every window compared here
    shares = cell_shares(panel, origin) if fixed else None
    if fixed and shares is None:
        raise MarketDynamicsError(
            f"fixed composition requested at origin {origin} but no cell history exists; "
            "build the panel with cell_column and ensure the origin has transactions")

    missing = [level for level in spec.levels if level not in row_keys]
    if missing:
        raise MarketDynamicsError(
            f"row_keys is missing {missing}; every level in the spec needs its key "
            f"(got {sorted(row_keys)})")
    values: list[float] = []
    for level in spec.levels:
        key = row_keys[level]
        by_window = {}
        for window in spec.windows:
            value, n, valid, fallback, _src = _resolve(
                panel, level, key, origin, window, shares)
            by_window[window] = (value, n, valid, fallback)
            values.append(float(np.log(value)) if value and value > 0 else np.nan)

        # the trailing-vs-prior comparison always uses the shortest declared window
        recent, prior, basis, recent_n, prior_n = _shared_pair(
            panel, level, key, origin, shares, spec.windows[0])
        if recent is not None and prior is not None and recent > 0 and prior > 0:
            values.append(float(np.log(recent) - np.log(prior)))
        else:
            values.append(np.nan)

        slope, valid_months, slope_source, _n = _slope6(
            panel, level, key, origin, shares)
        values.append(slope)
        # The scalars below are tied to specific windows. Use the spec's own windows when
        # present and fall back to an explicit short/medium/long read otherwise, so a spec
        # that omits one of them degrades to NaN rather than raising a KeyError.
        def _at(window, field):
            entry = by_window.get(window)
            return float(entry[field]) if entry is not None else float("nan")

        values.append(_at(spec.windows[0], 1) if spec.windows else float("nan"))
        values.append(_at(spec.windows[-1], 1) if spec.windows else float("nan"))
        # support for the *slope's own* series, so the count and the slope share a basis
        values.append(float(valid_months))
        values.append(_at(6 if 6 in by_window else (spec.windows[-1] if spec.windows else 6), 3))
        values.append(SOURCE_CODE.get(basis, FALLBACK_MISSING) if basis else -1.0)
        values.append(SOURCE_CODE.get(slope_source, FALLBACK_MISSING)
                      if slope_source else -1.0)
    return np.asarray(values, dtype=np.float64)
