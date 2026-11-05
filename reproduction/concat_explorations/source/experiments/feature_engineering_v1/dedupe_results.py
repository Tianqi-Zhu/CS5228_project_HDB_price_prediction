"""De-duplicate result CSVs without ever discarding a conflicting score.

Two different things look like duplicates:

1. **Cross-file repetition** - the same run is recorded by more than one stage
   (e.g. the baseline appears in the ablation, splits, focused and seeds files).
   Its key AND its scores are identical, so it is harmless.  Left untouched.
2. **Within-file repetition** - the same stage was appended twice (e.g. an
   aborted run followed by a rerun).  Key and scores are identical, so it only
   inflates row counts.  Collapsed here, with the count recorded.

Rows that share a key but disagree on any metric are **never** collapsed: they
are written to ``metrics_conflicts.csv`` and the file is left un-deduplicated so
the disagreement stays visible.

Run: ``./run.sh dedupe_results.py``   (``--dry-run`` to only report)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

KEY = ["model", "split_id", "seed", "parameters", "feature_groups"]
METRICS = ["rmse_sgd", "mae_sgd", "bias_sgd", "n_rows"]
FILES = [
    "metrics_ablation.csv",
    "metrics_splits.csv",
    "metrics_focused.csv",
    "metrics_catboost.csv",
    "metrics_smoothing.csv",
    "metrics_seeds.csv",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    base = config.REPORT_DIR
    log: list[dict] = []
    conflicts: list[pd.DataFrame] = []

    for name in FILES:
        path = base / name
        if not path.exists():
            print(f"MISSING {name}")
            continue
        frame = pd.read_csv(path)
        before = len(frame)

        # Detect conflicting keys *before* deduplicating anything.
        spread = frame.groupby(KEY, dropna=False)[METRICS].nunique(dropna=False)
        bad = spread[(spread > 1).any(axis=1)]
        if len(bad):
            bad_keys = bad.reset_index()[KEY].drop_duplicates()
            merged = frame.merge(bad_keys, on=KEY, how="inner")
            merged["conflict_in_file"] = name
            conflicts.append(merged)
            print(f"{name}: {len(bad)} conflicting key(s) kept as-is")
            # Conflicting files are not collapsed at all.
            log.append({"file": name, "rows_before": before, "rows_after": before,
                        "exact_dupes_removed": 0, "conflicting_keys": len(bad),
                        "action": "left untouched (conflict)"})
            continue

        deduped = frame.drop_duplicates(subset=KEY + METRICS, keep="first")
        removed = before - len(deduped)
        if removed and not args.dry_run:
            deduped.to_csv(path, index=False)
        log.append({"file": name, "rows_before": before, "rows_after": len(deduped),
                    "exact_dupes_removed": removed, "conflicting_keys": 0,
                    "action": "collapsed exact dupes" if removed else "no change"})
        print(f"{name}: {before} -> {len(deduped)} rows ({removed} exact dupes removed)")

    report = pd.DataFrame(log)
    print()
    print(report.to_string(index=False))
    if not args.dry_run:
        report.to_csv(base / "dedupe_report.csv", index=False)
        if conflicts:
            pd.concat(conflicts, ignore_index=True).to_csv(
                base / "metrics_conflicts.csv", index=False)
            print(f"\nWROTE metrics_conflicts.csv ({sum(len(c) for c in conflicts)} rows)")
        else:
            print("\nno conflicting keys in any file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
