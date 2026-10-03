"""Zero-dependency rental baseline: python3 baseline.py.

Uses recent six-month mean rents by town and flat type, with fallbacks.
Validation uses the last three observed months; these are local metrics,
not a claim about the course's official evaluation metric.
Cleaning and IQR filtering are shared with data_quality.py (run it for audit files).
IQR bounds are learned from fitting data only; validation/test rows stay intact.
Use --raw to compare with the original baseline on the same time split.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from data_quality import baseline_eligible, clean_rows, iqr_bounds, read_csv, remove_outliers

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / "data" / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def keys(row):
    town = row["TOWN"].strip().lower()
    flat = " ".join(row["FLAT_TYPE"].lower().replace("-", " ").split())
    return (town, flat), (flat,), ()


def fit(rows):
    months = sorted({r["RENT_APPROVAL_DATE"] for r in rows})
    cutoff = months[-min(6, len(months))]
    groups = defaultdict(list)
    for row in rows:
        if row["RENT_APPROVAL_DATE"] >= cutoff:
            for key in keys(row):
                groups[key].append(float(row["MONTHLY_RENT"]))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def predict(model, row):
    return next(model[key] for key in keys(row) if key in model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', action='store_true', help='Reproduce baseline before shared cleaning')
    args = parser.parse_args()
    train = read_csv(ROOT / 'data/train.csv')
    test = read_csv(ROOT / 'data/test.csv', training=False)
    if not args.raw:
        train, train_issues = clean_rows(train)
        test, _ = clean_rows(test)
        eligible = [row for row in train if baseline_eligible(row)]
        print(f"Cleaning: {len(train_issues):,} training cell issues; "
              f"excluded {len(train) - len(eligible):,} unusable training rows")
        train = eligible
    if not train:
        raise ValueError('No usable training rows')
    months = sorted({r["RENT_APPROVAL_DATE"] for r in train})
    if len(months) < 4:
        raise ValueError('Need at least four observed months for time validation')
    cutoff = months[-3]
    earlier = [r for r in train if r["RENT_APPROVAL_DATE"] < cutoff]
    valid = [r for r in train if r["RENT_APPROVAL_DATE"] >= cutoff]
    if not args.raw:
        bounds = iqr_bounds(earlier)
        before = len(earlier)
        earlier = remove_outliers(earlier, bounds)
        print(f'Validation-fit IQR: {bounds}; removed {before - len(earlier):,} training rows')
        if not earlier:
            raise ValueError('No training rows after IQR filtering')
    model = fit(earlier)
    errors = [predict(model, r) - float(r["MONTHLY_RENT"]) for r in valid]
    print(f"Validation: {cutoff} to {months[-1]} ({len(valid):,} rows)")
    print(f"MAE: {sum(abs(e) for e in errors) / len(errors):.2f}")
    print(f"RMSE: {math.sqrt(sum(e * e for e in errors) / len(errors)):.2f}")
    if not args.raw:
        bounds = iqr_bounds(train)
        before = len(train)
        train = remove_outliers(train, bounds)
        print(f'Final-fit IQR: {bounds}; removed {before - len(train):,}; retained {len(train):,}')
        if not train:
            raise ValueError('No training rows after IQR filtering')
    model = fit(train)
    template = read("example-submission.csv")
    if len(template) != len(test):
        raise ValueError("Submission template and test row counts differ")
    filename = 'baseline-raw-submission.csv' if args.raw else 'baseline-submission.csv'
    output = ROOT / "outputs" / filename
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
        for sample, row in zip(template, test):
            writer.writerow([sample["Id"], f"{predict(model, row):.2f}"])
    print(f"Saved {len(test):,} predictions to {output}")


if __name__ == "__main__":
    main()
