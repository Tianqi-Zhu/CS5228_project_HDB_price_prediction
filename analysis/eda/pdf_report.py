#!/usr/bin/env python3
"""Build the 16:9 PDF progress deck from the checked EDA outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.pipeline.data import check_output_path
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from analysis.pipeline.config import ROOT
DEFAULT_EDA = ROOT / "reports" / "eda"
EDA = DEFAULT_EDA
FIG = EDA / "figures"
TAB = EDA / "tables"
DEFAULT_OUT = DEFAULT_EDA / "progress_report.pdf"
OUT = DEFAULT_OUT
W, H = 960, 540
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#258C8C")
GOLD = colors.HexColor("#D89B33")
INK = colors.HexColor("#263746")
MUTED = colors.HexColor("#60717F")
PALE = colors.HexColor("#EAF2F3")


def money(x: float) -> str:
    return f"${x:,.0f}"


def load():
    summary = json.loads((EDA / "summary.json").read_text(encoding="utf-8"))
    tables = {p.stem: pd.read_csv(p) for p in TAB.glob("*.csv")}
    return summary, tables


def draw_paragraph(c, text: str, x: float, y: float, width: float, size=16, leading=23,
                   color=INK, font="Helvetica", max_height=360):
    """Small word wrapper for slide copy; returns the new baseline."""
    c.setFillColor(color)
    c.setFont(font, size)
    words = text.split()
    lines, line = [], ""
    for word in words:
        attempt = word if not line else line + " " + word
        if stringWidth(attempt, font, size) <= width:
            line = attempt
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    for item in lines:
        c.drawString(x, y, item)
        y -= leading
    return y - 10


def page_base(c, n: int, title: str, source: str):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    title_size = 29
    while title_size > 23 and stringWidth(title, "Helvetica-Bold", title_size) > W - 116:
        title_size -= 1
    c.setFont("Helvetica-Bold", title_size)
    c.drawString(58, H - 58, title)
    c.setStrokeColor(colors.HexColor("#D8E1E5"))
    c.setLineWidth(0.8)
    c.line(58, 48, W - 58, 48)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.5)
    max_source_width = 620
    safe_source = source
    while safe_source and stringWidth(safe_source, "Helvetica", 8.5) > max_source_width:
        safe_source = safe_source[:-2].rstrip(" ;,/.") + "..."
    c.drawString(58, 29, safe_source)
    c.drawRightString(W - 58, 29, f"CS5228 HDB rent project  |  {n:02d} / 15")


def image(c, name: str, x: float, y: float, width: float, height: float):
    path = FIG / name
    c.drawImage(ImageReader(str(path)), x, y, width, height, preserveAspectRatio=True,
                anchor="c", mask="auto")


def bullets(c, items: list[str], x=65, y=394, width=330, size=16, leading=22):
    for item in items:
        c.setFillColor(TEAL)
        c.circle(x + 4, y + 5, 3, fill=1, stroke=0)
        y = draw_paragraph(c, item, x + 18, y, width - 18, size, leading, INK)
    return y


def slide_cover(c, s):
    c.setFillColor(colors.white); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(0, 0, 18, H, fill=1, stroke=0)
    c.setFillColor(TEAL); c.rect(18, 0, 6, H, fill=1, stroke=0)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 38)
    c.drawString(74, 390, "HDB Monthly Rent Prediction")
    c.setFillColor(INK); c.setFont("Helvetica", 24)
    c.drawString(74, 346, "Progress Report")
    c.setFillColor(MUTED); c.setFont("Helvetica", 16)
    c.drawString(74, 286, "CS5228 Competition Project")
    c.drawString(74, 258, "Exploratory analysis and first validation evidence")
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 14)
    c.drawString(74, 104, f"{s['inputs']['train_rows']:,} labeled records  |  {s['inputs']['test_rows']:,} competition inputs")
    c.setFillColor(MUTED); c.setFont("Helvetica", 11)
    c.drawString(74, 77, f"Observed approval dates: {s['inputs']['train_date_min'][:7]} to {s['inputs']['test_date_max'][:7]}")
    c.setFillColor(MUTED); c.setFont("Helvetica", 9)
    c.drawRightString(W - 60, 30, "Local analysis of the supplied project data")
    c.showPage()


def slide_scope(c, s):
    page_base(c, 2, "Project question and analysis plan", "Source: data/train.csv, data/test.csv; EDA protocol in analysis/eda/comprehensive.py")
    draw_paragraph(c, "Question", 66, 405, 330, 19, 25, TEAL, "Helvetica-Bold")
    draw_paragraph(c, "How well can property, location and approval-time information predict monthly HDB rent for future records?", 66, 372, 330, 18, 26)
    draw_paragraph(c, "Current design", 66, 255, 330, 19, 25, TEAL, "Helvetica-Bold")
    bullets(c, ["Audit the supplied data and label distributions.", "Compare random and chronological holdouts.", "Test property and location features before adding date-aware economic series."], y=221, width=340, size=16, leading=22)
    image(c, "02_monthly_rent_trend.png", 428, 138, 475, 300)
    c.showPage()


def slide_data(c, s):
    page_base(c, 3, "Competition data and prediction boundary", "Source: data/train.csv and data/test.csv; row counts and dates: reports/eda/summary.json")
    data = [
        ("Labeled training", f"{s['inputs']['train_rows']:,} rows", f"{s['inputs']['train_date_min'][:7]} to {s['inputs']['train_date_max'][:7]}"),
        ("Competition test", f"{s['inputs']['test_rows']:,} rows", f"{s['inputs']['test_date_min'][:7]} to {s['inputs']['test_date_max'][:7]}"),
        ("Target", "MONTHLY_RENT", "SGD per month"),
        ("Inputs", "10 shared fields", "home, place, approval month"),
    ]
    y = 390
    for label, value, note in data:
        c.setFillColor(PALE); c.rect(66, y - 43, 828, 62, fill=1, stroke=0)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 16); c.drawString(82, y, label)
        c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 18); c.drawString(335, y, value)
        c.setFillColor(MUTED); c.setFont("Helvetica", 14); c.drawString(635, y, note)
        y -= 82
    draw_paragraph(c, "The test window extends beyond the labeled period, so future-period validation is central to estimating competition performance.", 70, 85, 820, 15, 20, INK)
    c.showPage()


def slide_target(c, s):
    t = s["target"]
    page_base(c, 4, "Target distribution", "Source: reports/eda/summary.json; reports/eda/figures/01_target_distribution.png")
    image(c, "01_target_distribution.png", 88, 114, 500, 310)
    bullets(c, [f"Mean {money(t['mean'])}; median {money(t['quantiles']['0.5'])} per month.",
                 f"Middle 50%: {money(t['quantiles']['0.25'])} to {money(t['quantiles']['0.75'])}.",
                 f"Skewness {t['skew']:.2f}; observed range {money(t['quantiles']['0.0'])} to {money(t['quantiles']['1.0'])}.",
                 "Mild right skew does not by itself justify a log target; compare validation errors before choosing a transform."], x=620, y=394, width=280, size=15, leading=21)
    c.showPage()


def slide_time(c, s, tab):
    annual = tab["annual_rent_trend"]
    first = annual.loc[annual.year == annual.year.min()].iloc[0]
    last = annual.loc[annual.year == annual.year.max()].iloc[0]
    page_base(c, 5, "Rent levels change across approval time", "Source: reports/eda/tables/annual_rent_trend.csv; monthly series in reports/eda/tables/monthly_rent_trend.csv")
    image(c, "02_monthly_rent_trend.png", 64, 108, 565, 324)
    bullets(c, [f"Annual median: {money(first['median'])} in {int(first['year'])} to {money(last['median'])} in {int(last['year'])}.",
                 f"The {int(last['year'])} figure covers {int(last['rows']):,} rows through March only.",
                 "Time is a first-order feature candidate and a reason to report chronological validation."], x=660, y=378, width=240, size=15, leading=22)
    c.showPage()


def slide_property(c, s):
    area = s["property"]["area_sqm"]
    page_base(c, 6, "Property and flat-type patterns", "Source: reports/eda/figures/05_rent_vs_floor_area.png and 09_monthly_rent_by_flat_type.png")
    image(c, "05_rent_vs_floor_area.png", 58, 150, 405, 272)
    image(c, "09_monthly_rent_by_flat_type.png", 495, 150, 405, 272)
    draw_paragraph(c, f"Area spans {area['min']:.0f}-{area['max']:.0f} sqm (median {area['median']:.0f}); rank association with rent is {area['spearman_approx']:.2f}. Large-area bins are sparse.", 68, 127, 385, 14, 19, INK)
    draw_paragraph(c, "Flat-type rent trends rise over time, with different levels by type. Normalize spelling before encoding categories.", 505, 127, 385, 14, 19, INK)
    c.showPage()


def slide_spatial(c, s, tab):
    b = s["spatial"]["block_join"]
    page_base(c, 7, "Block reference data provides full coordinate coverage", "Source: data/auxiliary/sg-hdb-block.csv; map and join diagnostics: reports/eda/summary.json")
    image(c, "11_block_rent_map.png", 370, 86, 520, 365)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 27); c.drawString(76, 365, "100%")
    c.setFillColor(INK); c.setFont("Helvetica", 16); c.drawString(76, 333, "train and test rows matched")
    c.drawString(76, 310, "a HDB block key")
    c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 21); c.drawString(76, 251, f"{b['unique_join_keys']:,}")
    c.setFillColor(INK); c.setFont("Helvetica", 15); c.drawString(76, 224, "unique block keys")
    draw_paragraph(c, "The map shows median rent at block centroids. Use location as a feature group and test its incremental value.", 76, 174, 255, 14, 19, MUTED)
    c.showPage()


def slide_distance(c, s):
    page_base(c, 8, "Raw distance associations are modest", "Source: reports/eda/figures/10_rent_by_open_mrt_distance.png; reports/eda/tables/location_feature_summary.csv")
    image(c, "10_rent_by_open_mrt_distance.png", 74, 115, 500, 322)
    loc = pd.read_csv(TAB / "rent_by_open_mrt_distance.csv")
    band = loc.iloc[:, 0].astype(str).tolist()
    # Keep only the predeclared two bands highlighted in the EDA output.
    low = next((r for _, r in loc.iterrows() if "0.3" in str(r.iloc[0])), None)
    mid = next((r for _, r in loc.iterrows() if "0.6" in str(r.iloc[0]) and "1.0" in str(r.iloc[0])), None)
    rho = s["spatial"]
    bullets(c, ["Nearest open MRT raw Spearman with rent: -0.086.",
                 f"Rent medians differ across distance bands ({money(float(low['median'])) if low is not None else 'see table'} near MRT; {money(float(mid['median'])) if mid is not None else 'see table'} at 0.6-1.0 km).",
                 "Town, flat type, area and time are mixed in these unadjusted comparisons; do not read them as an MRT premium."], x=620, y=392, width=280, size=15, leading=22)
    c.showPage()


def slide_quality(c, s, tab):
    quality = tab["data_quality"]
    cat = tab["categorical_normalization"].set_index("field")
    page_base(c, 9, "Data quality decisions preserve prediction-row integrity", "Source: reports/eda/tables/data_quality.csv and categorical_normalization.csv")
    entries = [
        ("No missing values", "in the supplied train and test columns"),
        ("Constant fields", "FURNISHED = yes; FEE = 0"),
        ("Label cleanup", f"FLAT_TYPE {int(cat.loc['FLAT_TYPE','raw_unique'])} to {int(cat.loc['FLAT_TYPE','normalized_unique'])} categories; STREET {int(cat.loc['STREET','raw_unique'])} to {int(cat.loc['STREET','normalized_unique'])}"),
        ("Repeated records", "425 exact duplicate train rows; 5,508 exact duplicate test rows. Of 13,434 repeated feature groups, 13,191 have different rents."),
        ("Policy", "Keep every test row in original order; do not deduplicate submission inputs"),
    ]
    y = 390
    for title, detail in entries:
        c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 17); c.drawString(84, y, title)
        y = draw_paragraph(c, detail, 330, y, 550, 16, 21, INK) - 15
    c.showPage()


def slide_drift(c, s, tab):
    dates = s["inputs"]
    drift = tab["train_test_drift"]
    page_base(c, 10, "The competition test set is a future-time test", "Source: reports/eda/summary.json; reports/eda/tables/train_test_drift.csv")
    image(c, "07_numeric_train_test_drift.png", 65, 140, 480, 285)
    bullets(c, [f"Train: {dates['train_date_min'][:7]} to {dates['train_date_max'][:7]}.",
                 f"Test: {dates['test_date_min'][:7]} to {dates['test_date_max'][:7]}.",
                 "Most test dates fall after the final labeled month; no target labels exist to measure 2026 error.",
                 "Sixteen test months are unseen in training (49,620 rows). Numeric KS distances are small for floor area and lease commencement year."], x=590, y=388, width=315, size=15, leading=22)
    c.showPage()


def slide_aux(c, s, tab):
    aux = tab["auxiliary_data_audit"].set_index("file")
    page_base(c, 11, "Auxiliary data creates useful features with timing constraints", "Source: data/auxiliary/*.csv; audit: reports/eda/tables/auxiliary_data_audit.csv")
    rows = [
        ("HDB blocks", int(aux.loc["sg-hdb-block.csv", "rows"]), "coordinates, completion year, planning area"),
        ("MRT stations", int(aux.loc["sg-mrt-stations.csv", "rows"]), "open and planned station points"),
        ("Malls / schools", f"{int(aux.loc['sg-shopping-malls.csv','rows'])} / {int(aux.loc['sg-schools.csv','rows'])}", "nearest-distance features"),
        ("COE / stocks", f"{int(aux.loc['sg-coe-prices.csv','rows'])} / {int(aux.loc['sg-stock-prices.csv','rows']):,}", "monthly or daily time series"),
    ]
    y = 383
    for label, count, desc in rows:
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 18); c.drawString(85, y, label)
        c.setFillColor(TEAL); c.setFont("Helvetica-Bold", 17); c.drawString(310, y, str(count))
        c.setFillColor(INK); c.setFont("Helvetica", 15); c.drawString(490, y, desc)
        y -= 66
    draw_paragraph(c, "Macro series are not merged in this EDA. Any later use must aggregate only information available by each approval date; same-month values can be published after the rental record. MRT status has no historical opening dates, so planned-station distances are current snapshots rather than time-valid features.", 85, 120, 790, 15, 21, MUTED)
    c.showPage()


def slide_baselines(c, s, tab):
    b = tab["baseline_validation"]
    temporal = b.query("split == 'temporal_2024plus' and baseline == 'train_only_town_flat_median'").iloc[0]
    temporal_recent = b.query("split == 'temporal_2024plus' and baseline == 'previous_12m_town_flat_median'").iloc[0]
    recent = b.query("split == 'recent_2025Q1' and baseline == 'previous_12m_town_flat_median'").iloc[0]
    random = b.query("split == 'random_80_20_seed5228' and baseline == 'train_only_town_flat_median'").iloc[0]
    page_base(c, 12, "Recent history improves the temporal baseline", "Source: reports/eda/tables/baseline_validation.csv; median predictor by TOWN x FLAT_TYPE")
    headers = ["Validation", "Fit window", "Rows", "MAE", "RMSE"]
    rows = [
        ("Random 20%", "all dates", f"{int(random.validation_rows):,}", money(random.mae), money(random.rmse)),
        ("Jan 24-Mar 25", "all history", f"{int(temporal.validation_rows):,}", money(temporal.mae), money(temporal.rmse)),
        ("Jan 24-Mar 25", "previous 12m", f"{int(temporal_recent.validation_rows):,}", money(temporal_recent.mae), money(temporal_recent.rmse)),
        ("Jan-Mar 25", "previous 12m", f"{int(recent.validation_rows):,}", money(recent.mae), money(recent.rmse)),
    ]
    xs = [76, 300, 485, 650, 760]
    y = 376
    c.setFillColor(NAVY); c.rect(65, y - 12, 830, 44, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 14)
    for x, h in zip(xs, headers): c.drawString(x, y + 3, h)
    y -= 55
    for i, row in enumerate(rows):
        c.setFillColor(PALE if i % 2 == 0 else colors.HexColor("#F7F9FA")); c.rect(65, y - 13, 830, 42, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont("Helvetica", 14)
        for x, v in zip(xs, row): c.drawString(x, y, str(v))
        y -= 48
    reduction = 1 - temporal_recent.rmse / temporal.rmse
    draw_paragraph(c, f"On the same future window, restricting group medians to the previous 12 months reduces RMSE by {reduction:.1%} versus using all history. The random holdout mixes years and is not directly comparable. These are local validation scores; the competition metric is RMSE.", 80, 115, 800, 15, 21, INK)
    c.showPage()


def slide_decisions(c, s):
    page_base(c, 13, "Current decisions and open challenges", "Evidence: reports/eda/summary.json and reports/eda/tables/*.csv")
    bullets(c, ["Use chronological holdouts as the primary model-selection evidence; keep random split as a diagnostic comparison.",
                 "Normalize text labels and preserve all test rows in source order.",
                 "Use property, time and town with LightGBM/CatBoost as supervised benchmarks; add spatial features as a separate ablation.",
                 "Treat nearest-place correlations as descriptive; compare adjusted model performance before attributing value.",
                 "Remaining challenge: the labeled data stop in March 2025, while most competition inputs are later."], x=84, y=400, width=790, size=17, leading=25)
    c.showPage()


def slide_next1(c):
    page_base(c, 14, "Next steps | test auxiliary features", "Ridge, Random Forest, CatBoost and LightGBM comparison is complete")
    bullets(c, ["Keep the current property + time + town models as fixed supervised benchmarks.",
                 "Add HDB block attributes first, then MRT, mall and school distance features in separate ablations.",
                 "Use date-aware joins for macro data and document assumptions where facility opening dates are unavailable.",
                 "Compare every feature group on the same temporal split using RMSE, MAE and mean bias."], x=86, y=392, width=790, size=18, leading=27)
    c.showPage()


def slide_next2(c):
    page_base(c, 15, "Next steps | robust validation and final report", "Planned work through project completion")
    bullets(c, ["Extend rolling-origin validation and compare full-history, recent-window and time-weighted training.",
                 "Tune the strongest boosting candidates only after selecting feature groups.",
                 "Check residuals by month, town and flat type, then report both improvements and failed additions.",
                 "Generate 50,000 predictions in the original row order and validate the Id,Predicted submission schema."], x=86, y=392, width=790, size=18, leading=27)
    c.showPage()


def main(eda_dir=None, output_path=None):
    global EDA, FIG, TAB, OUT
    EDA = Path(eda_dir) if eda_dir is not None else DEFAULT_EDA
    FIG, TAB = EDA / "figures", EDA / "tables"
    OUT = check_output_path(Path(output_path) if output_path is not None else DEFAULT_OUT)
    if OUT.suffix.lower() != ".pdf":
        raise ValueError("Report output must end in .pdf")
    s, tab = load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("CS5228 HDB Monthly Rent Prediction - Progress Report")
    c.setAuthor("CS5228 Project")
    slide_cover(c, s)
    slide_scope(c, s)
    slide_data(c, s)
    slide_target(c, s)
    slide_time(c, s, tab)
    slide_property(c, s)
    slide_spatial(c, s, tab)
    slide_distance(c, s)
    slide_quality(c, s, tab)
    slide_drift(c, s, tab)
    slide_aux(c, s, tab)
    slide_baselines(c, s, tab)
    slide_decisions(c, s)
    slide_next1(c)
    slide_next2(c)
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
