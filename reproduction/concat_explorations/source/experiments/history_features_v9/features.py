"""History summary features v9: seven grouping levels x seven statistics, plus cutoff age.

The point of this package is a **strictly causal, per-row** rental history: for a row whose
target month is `t`, the cutoff is `c = t - 3` (training) or `c = min(26, t - 3)`
(validation), and only months `<= c` may enter. The row's own label is never in its own
history, and no validation label is ever read.

Two details that are easy to get wrong and are therefore encoded explicitly:

* **`recency` is measured from `t`, not from `c`.** `c` is a policy choice (three months
  earlier); the *record age* the model should see is how old the newest observation is
  relative to the month being predicted. The companion `history_cutoff_age = t - c` tells
  the model what the cutoff policy was, so it can distinguish "no recent data" from
  "data was deliberately withheld".
* **A group with no observation is not a zero.** `count = 0`, means/std are NaN, and
  `recency = -1`. Filling those with 0 would claim a real rent of zero was observed.

Groupings (item numbers follow the plan):

1. `flat_type`
2. `town + flat_type`
3. `town + street + flat_type`
4. `town + subzone + flat_type`
5. `town + street + block + flat_type`   (the street is included: the same block number
   recurs on different streets, so `block` alone would merge unrelated addresses)
6. observable dwelling signature: `town + street + block + flat_type + flat_model +
   area@fixed precision + lease_commence`
7. `flat_type + flat_model + area band`

Level 6 is a *signature match*, not the same rented unit — nothing in this data identifies a
dwelling, and the schema says so. Levels 5 and 6 use the cleaned street key so that a block
number cannot silently join across streets.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- grouping levels
LEVELS = [
    "flat_type",
    "town_ft",
    "town_street_ft",
    "town_subzone_ft",
    "town_street_block_ft",
    "signature",
    "ft_model_areaband",
]

LEVEL_DESCRIPTIONS = {
    "flat_type": "FLAT_TYPE_CLEAN",
    "town_ft": "TOWN_CLEAN + FLAT_TYPE_CLEAN",
    "town_street_ft": "TOWN_CLEAN + STREET_CLEAN + FLAT_TYPE_CLEAN",
    "town_subzone_ft": "TOWN_CLEAN + SUBZONE + FLAT_TYPE_CLEAN",
    "town_street_block_ft": ("TOWN_CLEAN + STREET_CLEAN + BLOCK_CLEAN + FLAT_TYPE_CLEAN "
                             "(street included so a block number never joins across "
                             "different streets)"),
    "signature": ("TOWN_CLEAN + STREET_CLEAN + BLOCK_CLEAN + FLAT_TYPE_CLEAN + "
                  "FLAT_MODEL_CLEAN + FLOOR_AREA_SQM at fixed precision + "
                  "LEASE_COMMENCE year. An OBSERVABLE SIGNATURE match, not proof of the "
                  "same rented unit — this dataset has no unit identifier."),
    "ft_model_areaband": ("FLAT_TYPE_CLEAN + FLAT_MODEL_CLEAN + floor(FLOOR_AREA_SQM/10). "
                          "The band uses a fixed 10 sqm width, not a data quantile, so it "
                          "does not depend on the observed distribution."),
}

# --------------------------------------------------------------------------- statistics
STATS = [
    "count",
    "mean",
    "ew_mean",
    "sample_std",
    "last_month_mean",
    "last_month_median",
    "recency",
]

EW_HALF_LIFE_MONTHS = 12.0

STAT_DEFINITIONS = {
    "count": "number of transactions in the group with month <= c",
    "mean": "arithmetic mean of MONTHLY_RENT/1000 over those transactions",
    "ew_mean": ("exponentially weighted mean with weight 2**(-(c - record_month)/12), "
                "using only months <= c"),
    "sample_std": "sample standard deviation (ddof=1); NaN when count < 2",
    "last_month_mean": "mean rent/1000 among transactions in the newest observed month",
    "last_month_median": "median rent/1000 in that newest observed month",
    "recency": ("t - last_observed_month, i.e. age of the newest observation relative to "
                "the TARGET month (not relative to c); -1 when the group has no history"),
}

MISSING_POLICY = {
    "count": "0 means the group has no history; it is transformed with log1p",
    "mean": "NaN when count == 0",
    "ew_mean": "NaN when count == 0",
    "sample_std": "NaN when count < 2; a NaN std means 'not estimable', it is never a "
                  "claim that the observed dispersion is zero",
    "last_month_mean": "NaN when count == 0",
    "last_month_median": "NaN when count == 0",
    "recency": "-1 when count == 0 (a sentinel, not a real age)",
    "imputation": ("fit-only median imputation then standardisation; the imputation is "
                   "applied after masking, so an imputed std is never presented as an "
                   "observed 0"),
}


def feature_names() -> list[str]:
    """50 numeric names in a fixed order: 7 levels x 7 stats, then the cutoff age."""
    names = []
    for level in LEVELS:
        for stat in STATS:
            names.append(f"hist_{level}__{stat}")
    names.append("history_cutoff_age")
    return names


RAW_NAMES = feature_names()

# --------------------------------------------------------------------------- keys
UNKNOWN = "unknown"


def _clean(series):
    """String key with an explicit 'unknown' placeholder for a missing geography key."""
    return series.astype(str).str.strip().str.lower().replace(
        {"nan": UNKNOWN, "": UNKNOWN, "none": UNKNOWN})


def build_keys(frame, area_precision: int = 1) -> dict:
    """Per-level key tuples, one per row. Missing keys become 'unknown', never merged away."""
    town = _clean(frame["TOWN_CLEAN"])
    street = _clean(frame["STREET_CLEAN"])
    block = _clean(frame["BLOCK_CLEAN"])
    subzone = _clean(frame["SUBZONE"])
    flat_type = _clean(frame["FLAT_TYPE_CLEAN"])
    flat_model = _clean(frame["FLAT_MODEL_CLEAN"])

    area = frame["FLOOR_AREA_SQM"].to_numpy(dtype=float)
    # fixed decimal precision, so the signature does not depend on float formatting drift
    area_key = np.asarray([f"{a:.{area_precision}f}" if np.isfinite(a) else UNKNOWN
                           for a in area])
    area_band = np.asarray([str(int(np.floor(a / 10.0))) if np.isfinite(a) else UNKNOWN
                            for a in area])

    # LEASE_COMMENCE_DATE holds a year in this dataset; keep it as a string key so a real
    # date would also key distinctly rather than being silently truncated.
    lease = _clean(frame["LEASE_COMMENCE_DATE"])

    return {
        "flat_type": list(zip(flat_type)),
        "town_ft": list(zip(town, flat_type)),
        "town_street_ft": list(zip(town, street, flat_type)),
        "town_subzone_ft": list(zip(town, subzone, flat_type)),
        "town_street_block_ft": list(zip(town, street, block, flat_type)),
        "signature": list(zip(town, street, block, flat_type, flat_model, area_key, lease)),
        "ft_model_areaband": list(zip(flat_type, flat_model, area_band)),
    }


# --------------------------------------------------------------------------- cutoffs
def cutoff_for(target_month: int, *, train_end_month: int) -> int:
    """History cutoff for one row.

    Training rows use `t - 3`. Validation rows use `min(train_end, t - 3)`, so a validation
    row can never look past the training boundary even when `t - 3` would reach further.
    """
    natural = int(target_month) - 3
    return int(min(natural, train_end_month))


def cutoff_age(target_month: int, cutoff: int) -> int:
    """`t - c`: how many months before the target the cutoff sits (the declared policy)."""
    return int(target_month) - int(cutoff)


# --------------------------------------------------------------------------- aggregation
class GroupHistory:
    """Per-(level, key) records sorted by month, for causal prefix queries.

    A single aggregation pass builds every group once; a query for a given cutoff then walks
    only that group's records up to `c`, so the cost does not scale with the training table
    size per row.
    """

    def __init__(self, frame, month_index: np.ndarray, rent_ksgd: np.ndarray):
        self.month_index = np.asarray(month_index, dtype=np.int64)
        self.rent = np.asarray(rent_ksgd, dtype=np.float64)
        keys = build_keys(frame)
        self.tables = {}
        for level in LEVELS:
            level_keys = keys[level]
            buckets = {}
            for i, key in enumerate(level_keys):
                buckets.setdefault(key, []).append(i)
            table = {}
            for key, rows in buckets.items():
                rows = np.asarray(rows, dtype=np.int64)
                order = np.argsort(self.month_index[rows], kind="stable")
                rows = rows[order]
                table[key] = rows                     # sorted by month, ascending
            self.tables[level] = table

    def summarise(self, level: str, key, target_month: int, cutoff: int) -> np.ndarray:
        """The seven statistics for one (level, key) at one cutoff. Strictly `month <= c`."""
        rows = self.tables[level].get(key)
        values = np.full(len(STATS), np.nan)
        if rows is None:
            values[STATS.index("count")] = 0.0
            values[STATS.index("recency")] = -1.0
            return values

        months = self.month_index[rows]
        # causal slice: everything at or before the cutoff, nothing after
        stop = int(np.searchsorted(months, cutoff, side="right"))
        if stop == 0:
            values[STATS.index("count")] = 0.0
            values[STATS.index("recency")] = -1.0
            return values

        used_months = months[:stop]
        used_rent = self.rent[rows[:stop]]
        n = int(stop)

        values[STATS.index("count")] = float(n)
        values[STATS.index("mean")] = float(used_rent.mean())
        weights = np.power(2.0, -((cutoff - used_months) / EW_HALF_LIFE_MONTHS))
        values[STATS.index("ew_mean")] = float((used_rent * weights).sum() / weights.sum())
        values[STATS.index("sample_std")] = float(used_rent.std(ddof=1)) if n >= 2 else np.nan

        last_month = int(used_months[-1])
        last_mask = used_months == last_month
        values[STATS.index("last_month_mean")] = float(used_rent[last_mask].mean())
        values[STATS.index("last_month_median")] = float(np.median(used_rent[last_mask]))
        # recency is measured to the TARGET month, not to the cutoff
        values[STATS.index("recency")] = float(int(target_month) - last_month)
        return values


# --------------------------------------------------------------------------- scaling
class HistoryScaler:
    """Fit-only transform: log1p the count, median-impute then standardise the rest.

    The imputation happens **after** masking, and the observation masks are preserved
    alongside, so an imputed value is never mistaken for an observed one.
    """

    def __init__(self, median, mean, std, names):
        self.median = np.asarray(median, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.names = list(names)

    @classmethod
    def fit(cls, raw: np.ndarray, names: list[str]) -> "HistoryScaler":
        block = np.asarray(raw, dtype=np.float64).copy()
        count_cols = [i for i, n in enumerate(names) if n.endswith("__count")]
        for c in count_cols:
            block[:, c] = np.log1p(np.clip(block[:, c], 0, None))
        median = np.zeros(block.shape[1])
        mean = np.zeros(block.shape[1])
        std = np.ones(block.shape[1])
        for c in range(block.shape[1]):
            column = block[:, c]
            valid = column[np.isfinite(column)]
            if valid.size:
                median[c] = float(np.median(valid))
                mean[c] = float(valid.mean())
                std[c] = float(valid.std(ddof=0)) or 1.0
        return cls(median, mean, std, names)

    def transform(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scaled, observed_mask). Missing becomes the imputed value, not 0."""
        block = np.asarray(raw, dtype=np.float64).copy()
        observed = np.isfinite(block)
        count_cols = [i for i, n in enumerate(self.names) if n.endswith("__count")]
        for c in count_cols:
            block[:, c] = np.log1p(np.clip(block[:, c], 0, None))
            observed[:, c] = True          # a count of 0 is observed information
        block = np.where(np.isfinite(block), block, self.median)
        scaled = (block - self.mean) / self.std
        return scaled.astype(np.float32), observed
