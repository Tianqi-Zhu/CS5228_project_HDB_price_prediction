"""Reproduce the data audit, EDA figures, and temporal split for the progress report.

Run from the repository root with pandas and Pillow installed:
    python analysis/eda_train_test.py

The script reads the supplied CSV files without modifying them. All reported
group differences are descriptive; they do not estimate causal effects.
"""

from __future__ import annotations

import json

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from analysis.config import DATA, REPORTS, VALIDATION_START
from analysis.data import temporal_split

OUT = REPORTS / "train_test_eda"

WIDTH, HEIGHT = 1600, 900
BG = "#FFFFFF"
INK = "#193043"
MUTED = "#526675"
GRID = "#DFE6EA"
BLUE = "#147D91"
ORANGE = "#D78036"
LIGHT_ORANGE = "#FBEBD9"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"]
    names += ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE = font(45, True)
SUBTITLE = font(23)
LABEL = font(22)
SMALL = font(19)
TICK = font(19)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(im)
    draw.text((95, 54), title, fill=INK, font=TITLE)
    draw.text((98, 118), subtitle, fill=MUTED, font=SUBTITLE)
    draw.line((95, 166, 1505, 166), fill=GRID, width=2)
    return im, draw


def right_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, *,
               fill: str = MUTED, text_font: ImageFont.ImageFont = TICK) -> None:
    width = draw.textbbox((0, 0), value, font=text_font)[2]
    draw.text((xy[0] - width, xy[1]), value, fill=fill, font=text_font)


def save(im: Image.Image, name: str) -> None:
    im.save(OUT / name, optimize=True)


def rent_histogram(rent: pd.Series) -> None:
    im, draw = canvas("Monthly rent distribution", "Training set | SGD per month | n = 150,000")
    left, top, right, bottom = 165, 220, 1480, 750
    edges = list(range(0, 8250, 250))
    counts = pd.cut(rent, bins=edges, right=False, include_lowest=True).value_counts(sort=False)
    max_count = int(counts.max())
    for tick in range(0, max_count + 1, 5000):
        y = bottom - tick / max_count * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        right_text(draw, (left - 18, int(y) - 12), f"{tick:,}")
    n = len(counts)
    bar_width = (right - left) / n
    for i, count in enumerate(counts):
        x0 = left + i * bar_width + 2
        x1 = left + (i + 1) * bar_width - 2
        y0 = bottom - int(count) / max_count * (bottom - top)
        draw.rectangle((x0, y0, x1, bottom), fill=BLUE)
    for value in range(0, 8001, 1000):
        x = left + value / 8000 * (right - left)
        draw.text((x - 27, bottom + 18), f"{value:,}", fill=MUTED, font=TICK)
    median = float(rent.median())
    x = left + median / 8000 * (right - left)
    draw.line((x, top, x, bottom), fill=ORANGE, width=5)
    draw.text((x + 12, top + 8), f"Median: ${median:,.0f}", fill=ORANGE, font=LABEL)
    draw.text((left, 815), "Each bar spans SGD 250. Heights count rental approvals.", fill=MUTED, font=SMALL)
    save(im, "01_rent_distribution.png")


def monthly_trend(monthly: pd.DataFrame) -> None:
    im, draw = canvas("Median rent rose over the training period", "Raw monthly medians; changes in flat mix may also affect the line")
    left, top, right, bottom = 155, 235, 1475, 735
    values = monthly["median"].tolist()
    months = monthly.index.tolist()
    ymin, ymax = 1800, 3400
    x_for = lambda i: left + i / (len(values) - 1) * (right - left)
    y_for = lambda v: bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    val_index = months.index(VALIDATION_START)
    draw.rectangle((x_for(val_index) - 13, top, right, bottom), fill=LIGHT_ORANGE)
    for tick in range(ymin, ymax + 1, 400):
        y = y_for(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        right_text(draw, (left - 20, int(y) - 12), f"${tick:,}")
    points = [(x_for(i), y_for(v)) for i, v in enumerate(values)]
    draw.line(points, fill=BLUE, width=6, joint="curve")
    for x, y in points[::3] + [points[-1]]:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=BLUE)
    for i, month in enumerate(months):
        if (i % 6 == 0 and i <= len(months) - 5) or i == len(months) - 1:
            draw.text((x_for(i) - 33, bottom + 22), month, fill=MUTED, font=TICK)
    draw.text((x_for(val_index) + 15, top + 14), "Proposed validation window", fill=ORANGE, font=LABEL)
    draw.text((left, 816), f"{months[0]}: ${values[0]:,.0f}    {months[-1]}: ${values[-1]:,.0f}", fill=INK, font=LABEL)
    save(im, "02_monthly_median.png")


def flat_type_chart(groups: pd.DataFrame) -> None:
    im, draw = canvas("Rent by flat type", "Types with space and hyphen spellings combined | median and interquartile range")
    left, right = 385, 1460
    top, bottom = 250, 735
    xmin, xmax = 800, 4500
    x_for = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    for tick in range(1000, 4501, 500):
        x = x_for(tick)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x - 28, bottom + 17), f"{tick:,}", fill=MUTED, font=TICK)
    order = ["1-room", "2-room", "3-room", "4-room", "5-room", "executive"]
    gap = (bottom - top) / len(order)
    for i, label in enumerate(order):
        if label not in groups.index:
            continue
        row = groups.loc[label]
        y = top + gap * (i + 0.5)
        right_text(draw, (left - 38, int(y) - 14), f"{label} (n={int(row['count']):,})", text_font=LABEL)
        draw.line((x_for(row["q25"]), y, x_for(row["q75"]), y), fill=BLUE, width=12)
        x = x_for(row["median"])
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=ORANGE)
        draw.text((x + 18, y - 15), f"${row['median']:,.0f}", fill=INK, font=LABEL)
    draw.text((left, 817), "Raw group differences may also reflect location and approval date.", fill=MUTED, font=SMALL)
    save(im, "03_flat_type.png")


def town_chart(groups: pd.DataFrame) -> None:
    im, draw = canvas("Rent varies across towns", "All 26 towns | raw median monthly rent (SGD)")
    groups = groups.sort_values("median", ascending=False)
    left, right = 445, 1455
    top, bottom = 222, 767
    xmin, xmax = 2200, 3200
    x_for = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    for tick in range(2200, 3201, 200):
        x = x_for(tick)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x - 25, bottom + 14), f"{tick:,}", fill=MUTED, font=SMALL)
    gap = (bottom - top) / len(groups)
    for i, (town, row) in enumerate(groups.iterrows()):
        y = top + gap * (i + 0.5)
        right_text(draw, (left - 25, int(y) - 10), town, text_font=SMALL)
        x = x_for(row["median"])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BLUE)
        draw.text((x + 12, y - 10), f"{row['median']:,.0f}", fill=MUTED, font=SMALL)
    draw.text((left, 837), "Town medians are descriptive; compare models to assess predictive value.", fill=MUTED, font=SMALL)
    save(im, "04_town_median.png")


def area_chart(groups: pd.DataFrame) -> None:
    im, draw = canvas("Rent and floor area", "10 sqm bins with at least 100 approvals | raw median rent")
    left, top, right, bottom = 155, 230, 1460, 740
    xmin, xmax = 30, 220
    ymin, ymax = 1500, 4000
    x_for = lambda value: left + (value - xmin) / (xmax - xmin) * (right - left)
    y_for = lambda value: bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
    for tick in range(1500, 4001, 500):
        y = y_for(tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        right_text(draw, (left - 18, int(y) - 12), f"${tick:,}")
    for tick in range(40, 221, 20):
        x = x_for(tick)
        draw.text((x - 20, bottom + 17), str(tick), fill=MUTED, font=TICK)
    points = [(x_for(float(a) + 5), y_for(float(row["median"]))) for a, row in groups.iterrows()]
    draw.line(points, fill=BLUE, width=5)
    for x, y in points:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ORANGE)
    draw.text((650, 795), "Floor area (sqm)", fill=INK, font=LABEL)
    draw.text((left, 837), "The relationship is descriptive and may differ within each flat type.", fill=MUTED, font=SMALL)
    save(im, "05_floor_area.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    dates = pd.to_datetime(train["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    test_dates = pd.to_datetime(test["RENT_APPROVAL_DATE"], format="%Y-%m", errors="coerce")
    normalized_type = train["FLAT_TYPE"].str.lower().str.strip().str.replace(
        r"^(\d+)\s*[- ]\s*room$", r"\1-room", regex=True
    )
    monthly = train.groupby("RENT_APPROVAL_DATE")["MONTHLY_RENT"].agg(
        count="size", median="median", mean="mean"
    )
    flat_type = train.assign(FLAT_TYPE_CLEAN=normalized_type).groupby("FLAT_TYPE_CLEAN")[
        "MONTHLY_RENT"
    ].agg(count="size", median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
    town = train.groupby("TOWN")["MONTHLY_RENT"].agg(count="size", median="median")
    area_bin = (train["FLOOR_AREA_SQM"] // 10 * 10).astype(int)
    area = train.assign(AREA_BIN=area_bin).groupby("AREA_BIN")["MONTHLY_RENT"].agg(
        count="size", median="median"
    )
    area = area[area["count"] >= 100]
    train_part, valid_part = temporal_split(train)
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "FEE", "MONTHLY_RENT"]
    summary = {
        "source": {"train": "data/train.csv", "test": "data/test.csv"},
        "rows": {"train": len(train), "test": len(test)},
        "columns": {"train": train.columns.tolist(), "test": test.columns.tolist()},
        "date_range": {
            "train": [train["RENT_APPROVAL_DATE"].min(), train["RENT_APPROVAL_DATE"].max()],
            "test": [test["RENT_APPROVAL_DATE"].min(), test["RENT_APPROVAL_DATE"].max()],
            "invalid_train_dates": int(dates.isna().sum()),
            "invalid_test_dates": int(test_dates.isna().sum()),
        },
        "data_quality": {
            "train_missing_by_column": train.isna().sum().astype(int).to_dict(),
            "test_missing_by_column": test.isna().sum().astype(int).to_dict(),
            "exact_duplicate_train_rows": int(train.duplicated().sum()),
            "constant_train_columns": [c for c in train if train[c].nunique(dropna=False) == 1],
            "constant_test_columns": [c for c in test if test[c].nunique(dropna=False) == 1],
            "nonpositive_monthly_rent": int((train["MONTHLY_RENT"] <= 0).sum()),
            "nonpositive_floor_area": int((train["FLOOR_AREA_SQM"] <= 0).sum()),
            "lease_year_after_approval_year": int((train["LEASE_COMMENCE_DATE"] > dates.dt.year).sum()),
            "raw_flat_type_categories": train["FLAT_TYPE"].value_counts().to_dict(),
            "normalized_flat_type_categories": normalized_type.value_counts().to_dict(),
            "hyphenated_flat_type_by_approval_year": (
                train.assign(
                    YEAR=train["RENT_APPROVAL_DATE"].str[:4],
                    HYPHENATED=train["FLAT_TYPE"].str.contains("-room"),
                )
                .groupby("YEAR")["HYPHENATED"]
                .agg(count="size", hyphenated="sum")
                .reset_index()
                .to_dict("records")
            ),
            "numeric_ranges": {
                c: {"min": float(train[c].min()), "max": float(train[c].max())} for c in numeric
            },
        },
        "rent_quantiles_sgd": {
            str(q): float(train["MONTHLY_RENT"].quantile(q)) for q in [0, .01, .05, .25, .5, .75, .95, .99, 1]
        },
        "rent_mean_sgd": float(train["MONTHLY_RENT"].mean()),
        "area_rent_pearson_correlation": float(train[["FLOOR_AREA_SQM", "MONTHLY_RENT"]].corr().iloc[0, 1]),
        "monthly": monthly.reset_index().to_dict("records"),
        "flat_type": flat_type.reset_index().to_dict("records"),
        "town": town.reset_index().to_dict("records"),
        "area_10sqm_bins_min_100": area.reset_index().to_dict("records"),
        "proposed_validation": {
            "rule": f"train approval month < {VALIDATION_START}; validation month >= {VALIDATION_START}",
            "train_rows": len(train_part),
            "validation_rows": len(valid_part),
            "train_month_range": [train_part["RENT_APPROVAL_DATE"].min(), train_part["RENT_APPROVAL_DATE"].max()],
            "validation_month_range": [valid_part["RENT_APPROVAL_DATE"].min(), valid_part["RENT_APPROVAL_DATE"].max()],
            "test_month_range": [test["RENT_APPROVAL_DATE"].min(), test["RENT_APPROVAL_DATE"].max()],
            "test_months_beyond_train": int((test["RENT_APPROVAL_DATE"] > train["RENT_APPROVAL_DATE"].max()).sum()),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    rent_histogram(train["MONTHLY_RENT"])
    monthly_trend(monthly)
    flat_type_chart(flat_type)
    town_chart(town)
    area_chart(area)
    print(f"Wrote audit, validation design, and five figures to {OUT}")


if __name__ == "__main__":
    main()
