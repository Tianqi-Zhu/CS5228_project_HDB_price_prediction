"""Verify and report actual completed spatial feature experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import ROOT, DATA
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_local_economy_landscape import DEFAULT_CONFIG, paths, write_json, screen_scores
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.local_spatial_features import census_table, regional_features


def table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def cluster_bootstrap(actual, baseline, candidate, labels, n=5000):
    groups = pd.DataFrame({"cluster": np.asarray(labels), "a": (candidate - actual)**2, "b": (baseline - actual)**2,
                           "abs_a": np.abs(candidate - actual), "abs_b": np.abs(baseline - actual)})
    stats = groups.groupby("cluster").agg(a=("a", "sum"), b=("b", "sum"), abs_a=("abs_a", "sum"), abs_b=("abs_b", "sum"), n=("a", "size"))
    rng = np.random.default_rng(42)
    delta, mae = [], []
    values = stats.to_numpy(float)
    for _ in range(n):
        s = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        delta.append(np.sqrt(s[0] / s[4]) - np.sqrt(s[1] / s[4]))
        mae.append((s[2] - s[3]) / s[4])
    return {"clusters": len(stats), "resamples": n, "seed": 42,
            "rmse_delta_interval_95": np.quantile(delta, [.025, .975]).tolist(),
            "mae_delta_interval_95": np.quantile(mae, [.025, .975]).tolist(),
            "interpretation": "paired descriptive development-period uncertainty; does not adjust for feature selection or repeated outer-holdout reuse"}


def summarize(config=DEFAULT_CONFIG):
    cfg, out, cache = paths(config)
    settings = json.loads((out / "settings.json").read_text())
    assert sha256(CHECKPOINT) == settings["checkpoint_sha256"]
    assert sha256(ROOT / cfg["baseline"]) == cfg["baseline_sha256"]
    assert sha256(ROOT / cfg["baseline_predictions"]) == settings["baseline_predictions_sha256"]
    prepared_info = json.loads((out / "prepared_manifest.json").read_text())
    assert sha256(cache / "prepared.joblib") == prepared_info["sha256"]
    for path, expected in settings["input_sha256"].items():
        assert sha256(ROOT / path) == expected
    source_manifest = json.loads((out / "source_manifest.json").read_text())
    for name, entry in source_manifest.items():
        if entry["status"] == "acquired":
            assert sha256(ROOT / entry["path"]) == entry["sha256"]
    dictionary_path = out / "feature_dictionary.json"
    dictionary = json.loads(dictionary_path.read_text())
    income_note = out / "income_extension.json"
    income_manifest = None
    if income_note.exists():
        extension = json.loads(income_note.read_text())
        income_out = ROOT / extension["directory"]
        income_manifest = json.loads((income_out / "source_manifest.json").read_text())["income2022"]
        assert sha256(ROOT / income_manifest["path"]) == extension["source_sha256"]
        assert sha256(ROOT / ".cache" / income_out.name / "features.joblib") == extension["features_sha256"]
        for name in ("CENSUS_INCOME_NO_EMPLOYED_SHARE", "CENSUS_INCOME_BELOW3000_SHARE", "CENSUS_INCOME_GE10000_SHARE", "CENSUS_INCOME_UNAVAILABLE"):
            dictionary[name] = {"target_labels_used": False, "generation_class": "A", "units": "household proportion or unavailable indicator",
                                "policy": "Archived 2022-labelled Census 2020 table; conservative availability August 2022; planning-area support only"}
    for name, item in dictionary.items():
        if name.startswith("LOCAL_"):
            item.update(observation_timestamp="latest allowed month = row forecast_origin minus 1 month",
                        availability_timestamp="assumed 1-month publication lag; row origins in prepared cache",
                        geographic_support="same-flat-type nearby buildings; declared radius or kernel")
        elif name.startswith("CENSUS_INCOME_"):
            item.update(observation_timestamp="Census 2020", availability_timestamp="2022-08-01, next-month cutoff after July 28 2022 metadata date",
                        geographic_support="planning area; no household/building income inference")
        elif name.startswith("CENSUS2020_") and "RETRO" not in name:
            item.update(observation_timestamp="Census 2020", availability_timestamp="2021-07-01 conservative operational cutoff; published 2021-06-18",
                        geographic_support="planning area; no interpolation")
        else:
            item.update(observation_timestamp="current source snapshot; see source_manifest.json",
                        availability_timestamp="historical publication unknown; retrospective evaluation only",
                        geographic_support="building-centred metric buffer, Gaussian kernel, nearest geometry, or regional density")
    write_json(dictionary_path, dictionary)
    prepared = joblib.load(cache / "prepared.joblib")
    b = prepared["outer"]["bundle"]
    original_bundle = joblib.load(ROOT / cfg["baseline"])
    for key in ("X_train", "X_query"):
        pd.testing.assert_frame_equal(b[key], original_bundle[key])
    for key in ("y_train", "y_query", "train_baseline", "query_baseline"):
        np.testing.assert_array_equal(b[key], original_bundle[key])
    reference = pd.read_csv(ROOT / cfg["baseline_predictions"])
    np.testing.assert_array_equal(reference.row_id, b["X_query"].index)
    np.testing.assert_array_equal(reference.actual, b["y_query"])
    predictions = {"F0_baseline": reference}
    timings = []
    for path in sorted(out.glob("predictions_outer_*.csv")):
        if "_smoke" in path.stem:
            continue
        name = path.stem.removeprefix("predictions_outer_")
        f = pd.read_csv(path)
        np.testing.assert_array_equal(f.row_id, reference.row_id)
        np.testing.assert_array_equal(f.actual, reference.actual)
        np.testing.assert_array_equal(f.month, reference.month)
        assert np.isfinite(f.predicted).all() and len(f) == 28995 and f.row_id.is_unique
        predictions[name] = f
        timing = json.loads((out / f"timing_outer_{name}.json").read_text())
        assert timing["context_rows"] == 121005 and not timing["subsampling"] and timing["parameter_versions_unchanged"]
        timings.append({"model": name, **timing})
    for path in out.glob("alias_outer_*.json"):
        alias = path.stem.removeprefix("alias_outer_")
        target = json.loads(path.read_text())["same_features_as"]
        if target in predictions:
            predictions[alias] = predictions[target]
    raw = pd.read_csv(DATA / "train.csv")
    links = pd.read_csv(out / "transaction_building_links.csv", index_col="row_id")
    query_meta = raw.loc[reference.row_id].set_axis(reference.index)
    query_meta["building_id"] = links.loc[reference.row_id, "building_id"].to_numpy()
    static = pd.read_csv(cache / "building_static_features.csv", index_col="building_id")
    train_buildings = links.loc[b["X_train"].index, "building_id"]
    thresholds = {}
    for feature in ("PARK_FRACTION_300M", "FOOD_KDE_500M_KM2"):
        if feature in static:
            # Predetermined median split, fitted on context locations only.
            thresholds[feature] = float(static.loc[train_buildings.unique(), feature].median())
            values = static.loc[query_meta.building_id, feature].to_numpy()
            query_meta[feature] = np.where(values > thresholds[feature], "high", "low")
    rent_cuts = np.quantile(b["y_train"], [.25, .5, .75])
    query_meta["rent_range"] = pd.cut(reference.actual, [-np.inf, *np.unique(rent_cuts), np.inf], duplicates="drop").astype(str)
    write_json(out / "diagnostic_bins.json", {"training_location_medians": thresholds, "training_target_rent_quartiles": rent_cuts.tolist(), "uses_outer_labels_to_choose_bins": False})
    score_rows, monthly_rows, grouped, intervals = [], [], [], {}
    actual, baseline = reference.actual.to_numpy(), reference.predicted.to_numpy()
    base_rmse = metrics(actual, baseline)["rmse_sgd"]
    for name, f in predictions.items():
        p = f.predicted.to_numpy()
        m = metrics(actual, p)
        score_rows.append({"model": name, "n": len(f), **m, "r2": 1 - np.sum((p - actual)**2) / np.sum((actual - actual.mean())**2),
                           "delta_rmse_sgd": m["rmse_sgd"] - base_rmse, "relative_rmse_improvement_pct": 100 * (base_rmse - m["rmse_sgd"]) / base_rmse})
        for month, indices in f.groupby("month").groups.items():
            monthly_rows.append({"model": name, "month": month, "n": len(indices), **metrics(actual[indices], p[indices])})
        for category in ("TOWN", "FLAT_TYPE", "rent_range", *thresholds):
            for value, indices in query_meta.groupby(category, observed=True).groups.items():
                grouped.append({"model": name, "category": category, "group": value, "n": len(indices), **metrics(actual[indices], p[indices])})
        if name != "F0_baseline":
            intervals[name] = {"building": cluster_bootstrap(actual, baseline, p, query_meta.building_id),
                               "month": cluster_bootstrap(actual, baseline, p, reference.month)}
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / "scores.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "monthly_scores.csv", index=False)
    pd.DataFrame(grouped).to_csv(out / "grouped_scores.csv", index=False)
    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    write_json(out / "cluster_bootstrap.json", intervals)
    joined = reference.rename(columns={"predicted": "F0_baseline"})
    for name, f in predictions.items():
        if name != "F0_baseline":
            joined[name] = f.predicted.to_numpy()
    joined.to_csv(out / "predictions_comparison.csv", index=False, float_format="%.10f")
    screen = screen_scores(out, cfg["inner_origins"])
    if income_manifest:
        files = [income_out / f"predictions_inner_{origin}.csv" for origin in cfg["inner_origins"]]
        if all(p.exists() for p in files):
            parts = [pd.read_csv(p) for p in files]
            w = np.concatenate([f.weight.to_numpy() for f in parts])
            error = np.concatenate([(f.predicted - f.actual).to_numpy() for f in parts])
            screen = pd.concat([screen, pd.DataFrame([{"model": "regional_income", "rmse_sgd": float(np.sqrt(np.average(error**2, weights=w))),
                                                       "mae_sgd": float(np.average(np.abs(error), weights=w)), "query_rows": len(error)}])]).sort_values("rmse_sgd")
            screen.to_csv(out / "inner_scores_with_income.csv", index=False)
    selection_path = out / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    headline = []
    for label, name in (("Selected economy", "F1_economy"), ("Selected landscape", "F2_landscape"), ("Combined", "F3_combined"), ("Lagged rental fields", "strict_incremental"), ("Published census including income", "regional_income")):
        row = scores.loc[scores.model == name]
        if len(row):
            row = row.iloc[0]
            headline.append(f"{label}: RMSE S${row.rmse_sgd:.3f}, change {row.delta_rmse_sgd:+.3f} against frozen S${base_rmse:.3f}.")
    synergy = None
    if {"F0_baseline", "F1_economy", "F2_landscape", "F3_combined"} <= set(predictions):
        index = scores.set_index("model")
        # Higher is better score = negative MAE; positive interaction helps.
        synergy = float(-index.loc["F3_combined", "mae_sgd"] + index.loc["F1_economy", "mae_sgd"] + index.loc["F2_landscape", "mae_sgd"] - index.loc["F0_baseline", "mae_sgd"])
        write_json(out / "factorial_interaction.json", {"score": "negative MAE", "interaction_sgd": synergy, "positive_means": "complementarity on negative MAE; descriptive, not a causal/significance claim"})
    coverage = pd.read_csv(out / "feature_coverage.csv")
    town_coverage = []
    for split in ("train", "query"):
        extras = prepared["outer"]["added"][split]
        towns = raw.loc[extras.index, "TOWN"]
        for town, ids in towns.groupby(towns).groups.items():
            for feature in ("CENSUS2020_POPULATION", "PARK_FRACTION_300M", "FOOD_KDE_500M_KM2", "LOCAL_KERNEL_SHRUNK"):
                town_coverage.append({"split": split, "town": town, "feature": feature, "rows": len(ids), "missing_fraction": float(extras.loc[ids, feature].isna().mean())})
    pd.DataFrame(town_coverage).to_csv(out / "coverage_by_town.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    display_scores = scores.loc[~scores.model.isin(["commercial_kde_selected", "market_selected"])].copy()
    for position, row in enumerate(display_scores.itertuples(index=False)):
        lo, hi = intervals[row.model]["building"]["rmse_delta_interval_95"] if row.model in intervals else (0., 0.)
        axes[0].plot([lo, hi], [position, position], color="#0d9488", linewidth=2)
        axes[0].scatter([row.delta_rmse_sgd], [position], color="#64748b" if row.model == "F0_baseline" else "#0d9488", s=25)
    axes[0].set_yticks(range(len(display_scores)), display_scores.model)
    axes[0].axvline(0, color="#b91c1c", linestyle="--", linewidth=1)
    axes[0].set_xlabel("RMSE change vs baseline (SGD; negative improves)")
    axes[0].set_title("Paired building-cluster bootstrap 95% intervals", fontsize=10)
    axes[0].invert_yaxis()
    monthly = pd.DataFrame(monthly_rows)
    for name in ("F0_baseline", "F1_economy", "F2_landscape", "F3_combined", "strict_incremental"):
        part = monthly.loc[monthly.model == name]
        if len(part):
            axes[1].plot(part.month, part.rmse_sgd, marker="o", markersize=3, label=name)
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set_ylabel("Monthly RMSE (SGD)")
    axes[1].legend(fontsize=7)
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    buildings = pd.read_csv(out / "canonical_buildings.csv").set_index("building_id")
    source_coverage = []
    for split, feature_split in (("train", "train"), ("query", "query")):
        extras = prepared["outer"]["added"][feature_split]
        for group, columns in (("commercial_snapshot", ["FOOD_KDE_500M_KM2"]), ("park_snapshot", ["PARK_FRACTION_300M"]),
                               ("published_census", ["CENSUS2020_POPULATION", "CENSUS2020_AGE65_SHARE"]),
                               ("lagged_market", ["LOCAL_KERNEL_SHRUNK", "LOCAL_KERNEL_PPSM"])):
            available = extras[columns].notna().all(axis=1)
            source_coverage.append({"split": split, "group": group, "rows": len(extras), "available_rows": int(available.sum()), "missing_fraction": float(1 - available.mean())})
    test_links = links.loc[links["split"] == "test"]
    test_geography = buildings.loc[test_links.building_id].set_axis(test_links.index)
    if "census" in source_manifest:
        census = census_table(ROOT / source_manifest["census"]["path"])
        census_test = regional_features(test_geography, pd.Series(pd.Period("2025-03", "M"), index=test_geography.index), census)
        available = census_test.CENSUS2020_POPULATION.notna()
        source_coverage.append({"split": "unlabelled_test", "group": "published_census_at_last_labelled_month", "rows": len(test_links), "available_rows": int(available.sum()), "missing_fraction": float(1 - available.mean())})
    for group, feature in (("commercial_snapshot", "FOOD_KDE_500M_KM2"), ("park_snapshot", "PARK_FRACTION_300M")):
        available = static.loc[test_links.building_id, feature].notna()
        source_coverage.append({"split": "unlabelled_test", "group": group, "rows": len(test_links), "available_rows": int(available.sum()), "missing_fraction": float(1 - available.mean())})
    pd.DataFrame(source_coverage).to_csv(out / "source_coverage_by_split.csv", index=False)
    for ax, feature, title in zip(axes, ("PARK_FRACTION_300M", "FOOD_KDE_500M_KM2"), ("Managed park fraction within 300 m (snapshot)", "Food-centre Gaussian intensity, sigma 500 m")):
        artist = ax.scatter(buildings.LONGITUDE, buildings.LATITUDE, c=static.loc[buildings.index, feature], s=3, cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        fig.colorbar(artist, ax=ax, shrink=.7)
    fig.savefig(out / "geographic_features.png", dpi=170)
    plt.close(fig)
    recommendation = ""
    if income_manifest and {"F1_economy", "F2_landscape", "F3_combined", "regional_income", "commercial_fixed", "strict_incremental"} <= set(predictions):
        score_index = scores.set_index("model")
        lo, hi = intervals["F3_combined"]["building"]["rmse_delta_interval_95"]
        recommendation = (f"**Retain the original 77-feature baseline.** The selected combination changes RMSE by only S${score_index.loc['F3_combined', 'delta_rmse_sgd']:+.3f}; "
                          f"its paired building-cluster 95% interval is [{lo:+.3f}, {hi:+.3f}], crossing zero. "
                          "The strongest standalone economic result is the published census/income block, but it also fails to improve baseline RMSE. "
                          "The selected park representation worsens RMSE. Fixed-radius commercial counts outperform KDE on the full holdout, reversing their earlier screening order; neither beats X0. "
                          "There is no demonstrated robust improvement from these tested additions. Existing production/submission artifacts remain unchanged.")
    report = ["# Frozen TabPFN local economy and landscape experiments", "", recommendation, "", " ".join(headline), "",
              "The existing 77 features, rent residuals, pretrained checkpoint, full context, categorical encodings and inference settings are preserved. No TabPFN parameter is optimized. This is a repeatedly examined development holdout, not a new untouched test.", "",
              "## Data and information policy", "",
              "Public government sources acquired: Census 2020 (planning-area population and age composition), NEA market/food-centre points, NParks managed-park polygons, and URA MP2019 subzone polygons. Their source files total approximately 6.1 MiB; manifests preserve source URLs, observation/publication information and checksums. Existing mall coordinates are reused. Every one of 9,464 canonical buildings has one polygon match, agreeing with supplied planning area and subzone. No rows are removed.", "",
              ("A fifth 3.6 KB source provides Census 2020 household-income bands by planning area in a dataset last updated 28 July 2022. The separate regional_income ablation adds population/elderly share plus shares of households with no employed person, employed income below S$3,000, and income at least S$10,000, with availability flags (seven dimensions). Income from work includes employer CPF contributions, per the source. These are actual published band proportions, not estimated building incomes. Income features are unavailable before August 2022 forecast origins; the first earlier fold has no training-origin income coverage and that limitation is preserved. This predeclared regional ablation does not alter the already locked factorial selection. Its own source/settings/coverage/cache manifests are in ../tabpfn_local_economy_landscape_80_20_income/. The separate source revised in 2026 is excluded." if income_manifest else "The additional archived planning-area income ablation is pending."), "",
              "Census values enter only at origins from July 2021 after the 18 June 2021 release. Suppression/unmatched regions remain missing. Added rental fields use same flat type and labels observed by origin minus one month, with no self/current/future labels. Training features share X0's pseudo-origins; query origins remain fixed. The original 77 histories keep their existing availability policy.", "",
              "POIs and park geometries are current mirrors without verified opening/closing or archived boundary versions. All experiments using these additions are **retrospective geography experiments**, including the factorial; they do not establish historical point-in-time gains. `strict_incremental` is the earlier-selected lagged rental-field addition; `regional_income` is the predeclared published-census ablation. Their availability claim concerns additions, since inherited X0 already includes static amenities.", "",
              "Food features count food-service centres (not individual stalls/restaurants); malls are not a complete business register. KDE uses the normalized sum of Gaussian kernels in centres/km². EPSG:3414 is used for all distances and areas. Park fractions measure designated/managed park area, not satellite tree, grass, water or built-up probabilities. Polygon unions prevent overlap double counting. Landscape structure uses park-patch support only.", "",
              "Commercial radius counts use 300/500/1,000 m; selected non-redundant inputs add food-centre counts at all three radii and mall counts at 300 m, since X0 already has mall counts at 500/1,000/2,000 m. Gaussian KDE uses sigma 200/500/1,000 m: sum(exp(-d²/(2 sigma²)))/(2 pi sigma²), converted to centres/km², without boundary correction. All category points contribute. Count-derived densities and nearest distances are cached but not concatenated into every candidate.", "",
              "Park fractions use union area intersected with 100/300/500 m buffers, divided by actual buffer area. The Gaussian park feature approximates a sigma-200 m radial integral with piecewise-constant weights on these annuli. Structure adds largest connected park-patch fraction at 500 m and binary park/nonpark entropy. These are deterministic representations, not fitted encoders.", "",
              "Rental fields cover 6/12-month windows at 500/1,000/2,000 m. Raw comparisons use means/counts (neighborhood medians were not computed). Selected shrinkage uses 20 pseudo-observations toward the same-flat-type national history. The continuous field uses sigma 500 m truncated at 2 km and a 6-month temporal half-life; its weighted support mass is not an effective sample size. A historical rent-per-square-metre field supplies a limited composition adjustment, not a full hedonic adjustment. Raw means remain missing with zero local observations; shrunk estimates may back off to national history. No temporal model uses query-period labels.", "",
              "## Earlier representation comparisons", "", table(screen) if len(screen) else "Screening not complete.", "",
              "Two earlier ten-month folds keep all 60,079 / 87,937 context rows. Each fold has a fixed nested sample of 60 queries/month, chosen without labels and weighted to monthly population. All feature representations use the same sampled IDs. These are screening estimates, not complete-fold scores. Regions/KDE/market combinations are built only after individual screening; no outer labels choose the representation.", "",
              (f"Frozen earlier selections: economy `{selection['economy']}`, landscape `{selection['landscape']}`, market `{selection['market']}`. Complete column lists and selection audit: selection.json." if selection else "Earlier selection is pending."), "",
              "## Full chronological 80/20 results", "", table(display_scores), "",
              "Context: 121,005 January 2021–May 2024 rows; evaluation: all 28,995 June 2024–March 2025 rows. Every prediction file has identical unique IDs, months and actual rents. F0 is loaded after a full-context 400-query replay agreeing within S$0.00001. Shared selected columns reuse predictions through explicit aliases.", "",
              "## Factorial and uncertainty", "", f"Negative-MAE interaction `Score(F3)-Score(F1)-Score(F2)+Score(F0)` = {synergy} SGD. Positive values indicate descriptive complementarity. This interaction alone does not prove statistical significance.", "",
              "`cluster_bootstrap.json` contains 5,000 paired resamples clustered separately by building and month. Grouped results cover town, flat type, rent ranges defined by training-target quartiles, and high/low park/commercial bins defined by training-location medians. They do not adjust for repeated feature selection or development-holdout inspection. Entire-neighborhood spatial holdout was not run; the claim is chronological generalization only.", "",
              "## Coverage and compute", "", table(pd.DataFrame(timings)[["model", "added_dimensions", "fit_seconds", "predict_seconds", "peak_vram_gib", "context_rows"]]) if timings else "Outer inference pending.", "",
              "VRAM figures are peak PyTorch-allocated memory, not total driver reservation. One nvidia-smi observation during the economic run reached 15,874 MiB of device usage; driver memory was not sampled continuously. All complete runs fit the RTX 4080 without a context reduction or memory retry.", "",
              "Full feature/town missingness is recorded in feature_coverage.csv / coverage_by_town.csv. Source coverage for context, holdout and unlabelled test locations is in source_coverage_by_split.csv; test census availability is assessed at the last labelled month, not with future observations. New test rental fields/predictions were not generated. Unknown imagery/census is not zero-filled. No training-context reduction or prediction-row omission is permitted. Each inference job is isolated in a fresh process and saves resumable 200-row batches. The unlabelled competition test and existing submission are unchanged.", "",
              "## Unavailable sources and next steps", "",
              "Dynamic World probabilities and AlphaEarth embeddings were not extracted: Earth Engine credentials/project and imported exports are unavailable. Export scripts, dated import validation, train-only PCA at configurable 8/16/32 components, missingness handling and setup instructions are implemented; no satellite performance result is fabricated. The live AlphaEarth catalog lists observation years 2017–2024, but the product was publicly released on 30 July 2025, after this entire holdout. Strict historical AlphaEarth features would therefore be unavailable even with credentials. Retrospective observation-lagged embedding experiments must be labelled separately. Dynamic World likewise requires its actual 2022 product release as well as image dates.", "",
              "Income/employment at building resolution, complete commercial establishments, historical park snapshots, optional supervised economic encoders/GPs, extra landscape fragmentation indices, permutation importance and a spatially held-out study were not run. The targeted deterministic/KDE comparisons precede those optional costs. Fixed-radius and KDE candidates add four versus two columns, respectively; this compares the declared representations rather than proving a dimension-matched intrinsic advantage of continuous fields. Additional simple/continuous-economy × simple-landscape combinations were not swept; the complete selected 2×2 factorial is prioritized.", "",
              "Primary-source references: [Census table/publication](https://data.gov.sg/datasets/d_d95ae740c0f8961a0b10435836660ce0/view), [archived household-income table](https://data.gov.sg/datasets/d_2d6793de474551149c438ba349a108fd/view), [NEA](https://data.gov.sg/datasets/d_a57a245b3cf3ec76ad36d55393a16e97/view), [NParks](https://data.gov.sg/datasets/d_77d7ec97be83d44f61b85454f844382f/view), [URA](https://data.gov.sg/datasets/d_8594ae9ff96d0c708bc2af633048edfb/view), [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1), [AlphaEarth catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL), [AlphaEarth release](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/).", "",
              "Reproduce main study: `.venv/Scripts/python.exe analysis/acquire_local_spatial_sources.py`; `.venv/Scripts/python.exe analysis/compare_local_economy_landscape.py --stage all`. Regional income extension: `.venv/Scripts/python.exe analysis/acquire_local_spatial_sources.py --config analysis/configs/local_income_extension.json`; `.venv/Scripts/python.exe analysis/compare_local_income_extension.py --stage all`. Final verification/report: `.venv/Scripts/python.exe analysis/summarize_local_economy_landscape.py`. Use `--stage dry-run` on the main runner for access/configuration inventory. See inventory.md, feature_dictionary.json, both source manifests and SATELLITE_SETUP.md."]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    replay = json.loads((out / "baseline_replay.json").read_text()) if (out / "baseline_replay.json").exists() else None
    if replay:
        assert replay["verified"] and replay["max_abs_difference"] <= 1e-5
    submission = json.loads((out / "submission_preservation.json").read_text())
    assert sha256(ROOT / submission["path"]) == submission["sha256"]
    verification = {"baseline_checkpoint_hash_verified": True, "source_hashes_verified": True, "prepared_cache_hash_verified": True,
                    "all_77_baseline_columns_labels_and_residual_baselines_verified": True,
                    "baseline_replay_verified": bool(replay), "existing_submission_hash_verified": True,
                    "identical_evaluation_rows_verified": True, "finite_predictions_verified": True, "no_parameter_updates_verified": True,
                    "complete_outer_models": list(predictions), "prediction_hashes": {p.name: sha256(p) for p in out.glob("predictions_outer_*.csv")},
                    "code_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT / "analysis/compare_local_economy_landscape.py", Path(__file__), *list((ROOT / "analysis/experiments").glob("local_*fields.py")), ROOT / "analysis/experiments/local_spatial_features.py", ROOT / "analysis/experiments/satellite_feature_import.py"]}}
    write_json(out / "verification.json", verification)
    if selection and set(selection["outer_jobs"]) <= set(predictions) and income_manifest and replay:
        write_json(out / "status.json", {"status": "available_experiments_complete", "completed_models": list(predictions),
                                        "blocked": ["Dynamic World extraction", "AlphaEarth extraction; historical publication additionally unavailable"],
                                        "full_evaluation_rows": len(reference), "report": "report.md"})
    print(scores.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summarize(args.config)
