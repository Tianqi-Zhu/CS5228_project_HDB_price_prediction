"""Frozen full-context TabPFN: local economic and landscape feature ablations.

All additions preserve the 77-column residual baseline. Earlier chronological
folds select representations; complete 80/20 development queries confirm them.
Satellite runs are gated by real, dated exports, never synthetic substitutes.
"""
from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT, DATA, AUX
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.experiments.calendar_encoding import encoded_features, stratified_queries
from analysis.experiments.local_spatial_features import (
    canonical_buildings, commercial_fields, load_geometry, park_fields,
    polygon_assignment, census_table, regional_features, metric_xy,
)
from analysis.experiments.local_market_fields import historical_market_features
from analysis.experiments.spatiotemporal_hybrid import spatial_frame

DEFAULT_CONFIG = ROOT / "analysis/configs/local_economy_landscape.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Inputs/configuration changed; preserve existing output: {path}")
    if not path.exists():
        write_json(path, value)


def paths(config):
    cfg = json.loads(Path(config).read_text())
    out, cache = ROOT / "reports" / cfg["experiment"], ROOT / ".cache" / cfg["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return cfg, out, cache


def load_base(cfg):
    path = ROOT / cfg["baseline"]
    if sha256(path) != cfg["baseline_sha256"]:
        raise ValueError("Baseline feature artifact changed")
    b = joblib.load(path)
    assert b["X_train"].shape == (121005, 77) and b["X_query"].shape == (28995, 77)
    return b


def group_columns(columns):
    columns = list(columns)
    groups = {"baseline": [], "regional": [c for c in columns if c.startswith("CENSUS2020_") and "RETRO" not in c],
              "commercial_fixed": [f"FOOD_COUNT_{r}M" for r in (300, 500, 1000)] + ["MALL_COUNT_300M"],
              "park_basic": ["PARK_NEAREST_M", "PARK_FRACTION_300M"],
              "park_multiscale": ["PARK_NEAREST_M", "PARK_FRACTION_100M", "PARK_FRACTION_300M", "PARK_FRACTION_500M", "PARK_GAUSSIAN_200M_FRACTION"],
              "park_structure": ["PARK_NEAREST_M", "PARK_FRACTION_100M", "PARK_FRACTION_300M", "PARK_FRACTION_500M", "PARK_GAUSSIAN_200M_FRACTION", "PARK_LARGEST_PATCH_FRACTION_500M", "PARK_BINARY_ENTROPY"]}
    for sigma in (200, 500, 1000):
        groups[f"commercial_kde{sigma}"] = [f"FOOD_KDE_{sigma}M_KM2", f"MALL_KDE_{sigma}M_KM2"]
    groups["market_raw"] = [c for c in columns if c.startswith("LOCAL_") and (c.endswith("_MEAN") or c.endswith("_COUNT")) and "KERNEL" not in c] + ["LOCAL_HISTORY_UNAVAILABLE"]
    groups["market_smoothed"] = [f"LOCAL_{r}M_{w}M_{s}" for r in (500, 1000) for w in (6, 12) for s in ("SHRUNK", "COUNT")] + ["LOCAL_KERNEL_SHRUNK", "LOCAL_KERNEL_MASS", "LOCAL_KERNEL_PPSM", "LOCAL_HISTORY_UNAVAILABLE"]
    return {name: names for name, names in groups.items() if set(names) <= set(columns)}


def prepare(config):
    cfg, out, cache = paths(config)
    manifest = json.loads((out / "source_manifest.json").read_text())
    sources = {}
    for name, entry in manifest.items():
        if entry["status"] == "acquired":
            path = ROOT / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Source changed: {name}")
            sources[name] = path
    base = load_base(cfg)
    settings = {"config": cfg, "source_sha256": {name: sha256(p) for name, p in sources.items()},
                "baseline_predictions_sha256": sha256(ROOT / cfg["baseline_predictions"]),
                "checkpoint_sha256": sha256(CHECKPOINT),
                "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (DATA / "train.csv", DATA / "test.csv", AUX / "sg-hdb-block.csv", AUX / "sg-shopping-malls.csv")},
                "inference": {"frozen": True, "version": "v3.5-fast", "n_estimators": 1, "random_state": 42,
                              "batch_size": 200, "fit_mode": "fit_with_cache", "subsampling": False, "context_rows": 121005,
                              "target": "y_train - train_baseline; query_baseline added once"},
                "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn", "pyproj", "shapely")},
                "selection": "Earlier pooled weighted RMSE; fewer added dimensions within 0.1% of best; baseline competes for retained recommendation. Forced nonempty groups for factorial are reported even if screening is worse.",
                "strict_scope": "New feature availability only; inherited 77 features include static amenity snapshots and are preserved for paired comparison."}
    immutable(out / "settings.json", settings)
    if (cache / "prepared.joblib").exists():
        info = json.loads((out / "prepared_manifest.json").read_text())
        if sha256(cache / "prepared.joblib") != info["sha256"]:
            raise ValueError("Prepared feature cache changed")
        print("Prepared features already verified", flush=True)
        return
    raw_train, raw_test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    combined = pd.concat([raw_train.assign(_split="train"), raw_test.assign(_split="test")], ignore_index=True)
    buildings, links = canonical_buildings(pd.read_csv(AUX / "sg-hdb-block.csv"), combined)
    buildings.to_csv(out / "canonical_buildings.csv", index=False)
    links.assign(split=combined._split).to_csv(out / "transaction_building_links.csv")
    xy = buildings[["x_m", "y_m"]].to_numpy()
    static = pd.DataFrame(index=buildings.index)
    if "food" in sources:
        geo, props = load_geometry(sources["food"])
        # Count actual food-service centres; standalone wet markets are a
        # different commercial category and are not called restaurants/stalls.
        points = [g for g, p in zip(geo, props) if float(p.get("CF_STALLS", 0) or 0) > 0]
        food = commercial_fields(xy, np.array([[g.x, g.y] for g in points]), cfg["commercial_radii_m"], cfg["commercial_bandwidths_m"], "FOOD")
        food.index = static.index
        static = static.join(food)
    malls = pd.read_csv(AUX / "sg-shopping-malls.csv")
    mall = commercial_fields(xy, metric_xy(malls.LONGITUDE, malls.LATITUDE), cfg["commercial_radii_m"], cfg["commercial_bandwidths_m"], "MALL")
    mall.index = static.index
    static = static.join(mall)
    if "parks" in sources:
        parks, _ = load_geometry(sources["parks"])
        features = park_fields(xy, parks, cfg["landscape_radii_m"])
        features.index = static.index
        static = static.join(features)
    if "subzones" in sources:
        polygons, properties = load_geometry(sources["subzones"])
        assignment = polygon_assignment(buildings, polygons, properties)
        assignment["planning_area_agrees"] = assignment.polygon_planning_area.eq(buildings.PLANNING_AREA.str.lower())
        assignment["subzone_agrees"] = assignment.polygon_subzone.eq(buildings.SUBZONE.str.lower())
        assignment.to_csv(out / "polygon_join_audit.csv")
        # Density is explicitly retrospective: the downloaded geometry mirror
        # is not a verified archived historical release.
        if "census" in sources:
            census = census_table(sources["census"])
            area = pd.DataFrame({"name": [str(p["PLN_AREA_N"]).lower() for p in properties], "m2": [g.area for g in polygons]}).groupby("name").m2.sum()
            pop = census.loc[census.is_planning_area].set_index("region_label").CENSUS2020_POPULATION
            density = pop / (area / 1e6)
            static["CENSUS2020_DENSITY_RETRO_KM2"] = buildings.PLANNING_AREA.str.lower().map(density)
    static.to_csv(cache / "building_static_features.csv")
    frame = spatial_frame(raw_train)
    frame["building_id"] = links.building_id.reindex(frame.index)
    bundles = {"outer": base}
    for origin in cfg["inner_origins"]:
        source = ROOT / f".cache/tabpfn_v3_features/calendar_inner_{origin}.joblib"
        source_info = ROOT / f"reports/tabpfn_v3_5_fast_calendar_80_20/inner_features_{origin}.json"
        if sha256(source) != json.loads(source_info.read_text())["sha256"]:
            raise ValueError("Earlier-fold cache changed")
        b = joblib.load(source)
        # Categorical month matches our explicitly chosen 77-feature baseline.
        for split in ("train", "query"):
            b[f"X_{split}"] = encoded_features(b[f"X_{split}"], b[f"{split}_month"], "categorical")
        # Source queries were sampled 200/month independently of labels. Take
        # a nested 60/month sample, preserving monthly population weights.
        selected, _ = stratified_queries(b["query_month"], b["X_query"].index, cfg["inner_query_per_month"], cfg["seed"])
        before = pd.PeriodIndex(b["query_month"], freq="M")
        weights = np.asarray(b["query_weights"])[selected] * np.array([(before == before[p]).sum() / (before[selected] == before[p]).sum() for p in selected])
        b["X_query"] = b["X_query"].iloc[selected].copy()
        for name in ("query_month", "y_query", "query_baseline"):
            b[name] = np.asarray(b[name])[selected]
        b["query_weights"] = weights
        bundles[f"inner_{origin}"] = b
    census = census_table(sources["census"]) if "census" in sources else None
    availability = []
    prepared = {}
    for stage, b in bundles.items():
        added = {}
        for split in ("train", "query"):
            matrix = b[f"X_{split}"]
            ids = matrix.index
            if split == "train":
                origins = base["train_temporal_origins"].reindex(ids).forecast_origin
            else:
                origin = pd.PeriodIndex(b["query_month"], freq="M").min() - 1
                origins = pd.Series(origin, index=ids)
            transaction = frame.loc[ids]
            extra = static.reindex(transaction.building_id).set_axis(ids)
            # The newly lagged local market fields only receive context labels.
            local = historical_market_features(frame.loc[b["X_train"].index], transaction.drop(columns="MONTHLY_RENT"), origins, buildings, cfg)
            extra = extra.join(local)
            if census is not None:
                extra = extra.join(regional_features(transaction, origins, census, cfg["datasets"]["census"]["available_at"]))
            added[split] = extra
            for column in extra:
                availability.append({"stage": stage, "split": split, "feature": column, "rows": len(extra),
                                     "missing": int(extra[column].isna().sum()), "missing_fraction": float(extra[column].isna().mean()),
                                     "locations": int(transaction.building_id.nunique())})
            extra.to_csv(cache / f"features_{stage}_{split}.csv")
        prepared[stage] = {"bundle": b, "added": added}
        print(f"Prepared {stage}: {len(added['train']):,} context, {len(added['query']):,} queries", flush=True)
    groups = group_columns(prepared["outer"]["added"]["train"].columns)
    if "regional" in groups and "CENSUS2020_DENSITY_RETRO_KM2" in static:
        # Keep the region-safe and current-geometry density comparisons distinct.
        groups["regional_density_retro"] = groups["regional"] + ["CENSUS2020_DENSITY_RETRO_KM2"]
    pd.DataFrame(availability).to_csv(out / "feature_coverage.csv", index=False)
    write_json(out / "feature_groups.json", groups)
    dictionary = {}
    for c in prepared["outer"]["added"]["train"]:
        target_used = c.startswith("LOCAL_")
        policy = "lagged prequential, origin-aligned; context labels only" if target_used else ("published 2020 census, unavailable before July 2021" if c.startswith("CENSUS2020_") and "RETRO" not in c else "retrospective current geographic snapshot; historical availability unverified")
        dictionary[c] = {"target_labels_used": target_used, "policy": policy,
                         "generation_class": "C" if target_used else "A", "units": "fraction" if "FRACTION" in c or "SHARE" in c else ("per km2" if "KM2" in c else ("metres" if c.endswith("_M") else "see feature name"))}
    write_json(out / "feature_dictionary.json", dictionary)
    joblib.dump(prepared, cache / "prepared.joblib", compress=3)
    write_json(out / "prepared_manifest.json", {"sha256": sha256(cache / "prepared.joblib"), "stages": list(prepared), "features_added_available": len(dictionary)})
    write_json(out / "satellite_status.json", {
        "dynamic_world": {"status": "blocked", "reason": "Earth Engine package/credentials/project absent; no user export supplied", "catalog": "https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1"},
        "alphaearth": {"status": "blocked", "reason": "Earth Engine authorization absent; strict historical availability also fails: public release July 30 2025 is after entire outer holdout", "catalog_years": [2017, 2024], "catalog_checked_on": "2026-10-30", "release_url": "https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/"}})
    write_inventory(out, base, manifest, groups)


def write_inventory(out, base, manifest, groups):
    text = ["# Initial implementation inventory", "", "Predictand: monthly rent in SGD, with an additive temporal residual target; no logarithm.", "",
            "X0: frozen TabPFN-3.5 Fast, TabPFN 9.2.0, checkpoint tabpfn-v3.5-fast-20260909.safetensors; 77 features, one estimator, seed 42, 200-row query batches, full cached context with no subsampling. No DiagScale/LoRA/adapters.", "",
            "Chronological 80/20: 121,005 January 2021–May 2024 context rows and 28,995 June 2024–March 2025 development rows. Unlabelled 50,000-row test remains untouched. The supplied test extends to July 2026; this comparison does not validate that longer horizon.", "",
            "Frozen categorical baseline: RMSE S$455.81357012; existing adapted models are outside this frozen-model experiment. X0 has property, spatial, amenity, 37 historical and 9 temporal features. It does not contain national economic features; earlier economic experiments are separate.", "",
            "Preserved baseline inputs:", "", ", ".join(base["X_train"].columns), "",
            "Available additions: published 2020 Census planning-area totals/elderly share; food-service-centre and mall radius/KDE intensities; publication-lagged flat-type-specific rental fields; current NParks managed-park polygon statistics.", "",
            "Unavailable: building/subzone income, complete retail/restaurant establishments, historical establishment openings/closures, archived park boundaries, authenticated Dynamic World/AlphaEarth extracts. Mall/food-centre counts are not total commercial establishments or employment measures. Park area is not tree/grass cover.", "",
            "Census publication: 18 June 2021; new features conservatively enter July 2021 forecast origins onward. Newly added rental fields use origin minus one month. Existing X0 histories are preserved at their original information policy; incremental availability does not certify the inherited amenity snapshots as historically available.", "",
            "Original source metadata, retrieval timestamps, hashes, resolution and policies are in source_manifest.json. Distances/areas use EPSG:3414. Canonical location table uses exact normalized address and supplied coordinates; no postal-only merging.", "",
            "Predeclared screening: two earlier ten-month folds, full context, nested label-free 60 query rows/month with population weights. Choose representations solely on earlier pooled RMSE; prefer fewer added columns within 0.1%. The baseline competes for the final retained recommendation; forced nonempty economy/landscape groups permit a meaningful factorial even if they lose screening.", "",
            "Strict-incremental and retrospective results must be reported separately. AlphaEarth's July 2025 public release means even 2017–2023 image-year embeddings could not have been available during this 2024–25 holdout."]
    (out / "inventory.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def matrix_pair(entry, names):
    b = entry["bundle"]
    result = []
    for split in ("train", "query"):
        original = b[f"X_{split}"]
        if set(names) & set(original):
            raise ValueError("Attempt to replace baseline inputs")
        matrix = original.join(entry["added"][split][names])
        pd.testing.assert_frame_equal(matrix[original.columns], original)
        result.append(matrix)
    return result


def frozen_predict(x, y, query, progress, batch=200):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda",
                n_estimators=1, random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
                inference_config={"SUBSAMPLE_SAMPLES": None})
    start = time.perf_counter()
    try:
        model.fit(x, y)
        torch.cuda.synchronize()
        fitted = time.perf_counter()
        assert model.n_train_samples_ == len(x) and model.inference_config_.SUBSAMPLE_SAMPLES is None
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None:
            assert all(len(v) == len(x) and len(np.unique(v)) == len(x) for v in indices)
        # Inference-only signature catches unintended parameter mutation.
        signature = [(p._version, p.requires_grad) for m in model.models_ for p in m.parameters()]
        values = np.full(len(query), np.nan)
        if progress.exists():
            saved = np.load(progress)
            assert saved.shape == values.shape
            values = saved
        for offset in range(0, len(query), batch):
            stop = min(offset + batch, len(query))
            if np.isfinite(values[offset:stop]).all():
                continue
            values[offset:stop] = model.predict(query.iloc[offset:stop], output_type="mean")
            np.save(progress, values)
            if offset % 4000 == 0 or stop == len(query):
                print(f"  Predicted {stop:,}/{len(query):,}", flush=True)
        torch.cuda.synchronize()
        assert signature == [(p._version, p.requires_grad) for m in model.models_ for p in m.parameters()]
        assert np.isfinite(values).all()
        return values, {"fit_seconds": fitted - start, "predict_seconds": time.perf_counter() - fitted,
                        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "context_rows": len(x), "query_rows": len(query), "features": len(x.columns),
                        "batch_size": batch, "subsampling": False, "parameter_versions_unchanged": True}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run_one(config, stage, name, names=None, smoke=False):
    cfg, out, cache = paths(config)
    prepared = joblib.load(cache / "prepared.joblib")
    groups = json.loads((out / "feature_groups.json").read_text())
    if names is None:
        names = groups[name]
    entry, tag = prepared[stage], f"{stage}_{name}"
    if smoke:
        tag += "_smoke"
    path = out / f"predictions_{tag}.csv"
    x, q = matrix_pair(entry, names)
    b = entry["bundle"]
    limit = 400 if smoke else len(q)
    q = q.iloc[:limit]
    target = np.asarray(b["y_train"], float) - b["train_baseline"]
    job_settings = {"stage": stage, "name": name, "added_columns": names, "context_ids_sha256": hashlib_array(x.index),
                    "query_ids_sha256": hashlib_array(q.index), "prepared_sha256": sha256(cache / "prepared.joblib"),
                    "code_sha256": sha256(Path(__file__)), "checkpoint_sha256": sha256(CHECKPOINT), "smoke": smoke}
    immutable(out / f"job_{tag}.json", job_settings)
    if path.exists() and (out / f"timing_{tag}.json").exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id, q.index)
        np.testing.assert_array_equal(saved.actual, np.asarray(b["y_query"])[:limit])
        print(f"Reused {tag}", flush=True)
        return
    print(f"{tag}: {len(x):,} full context; {len(q):,} queries; +{len(names)} features", flush=True)
    write_json(out / "status.json", {"stage": stage, "name": name, "status": "inference"})
    residual, timing = frozen_predict(x, target, q, cache / f"partial_{tag}.npy", cfg["outer_batch_size"])
    predicted = residual + np.asarray(b["query_baseline"])[:limit]
    saved = pd.DataFrame({"row_id": q.index, "month": np.asarray(b["query_month"])[:limit],
                          "actual": np.asarray(b["y_query"])[:limit], "predicted": predicted})
    saved["error"] = saved.predicted - saved.actual
    saved["absolute_error"] = saved.error.abs()
    saved["weight"] = np.asarray(b.get("query_weights", np.ones(len(b["X_query"]))))[:limit]
    saved.to_csv(path, index=False, float_format="%.10f")
    timing.update(added_dimensions=len(names), **metrics(saved.actual, saved.predicted))
    write_json(out / f"timing_{tag}.json", timing)
    print(json.dumps(timing), flush=True)
    if smoke and not names:
        reference = pd.read_csv(ROOT / cfg["baseline_predictions"]).set_index("row_id").loc[q.index]
        np.testing.assert_allclose(predicted, reference.predicted, atol=1e-5, rtol=0)
        write_json(out / "baseline_replay.json", {"rows": len(q), "max_abs_difference": float(np.max(np.abs(predicted - reference.predicted))), "verified": True})


def hashlib_array(values):
    import hashlib
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def screen_scores(out, origins):
    records = []
    for path in sorted(out.glob("predictions_inner_*.csv")):
        if "_smoke" in path.stem:
            continue
        name = path.stem.split("_", 3)[3]
        stage = path.stem.split("_", 3)[2]
        f = pd.read_csv(path)
        d = f.predicted - f.actual
        records.append({"model": name, "origin": stage, "n": len(f), "squared_sum": float(np.sum(f.weight * d**2)), "absolute_sum": float(np.sum(f.weight * d.abs())), "weight": float(f.weight.sum())})
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    scores = []
    for name, rows in frame.groupby("model"):
        if len(rows) == len(origins):
            scores.append({"model": name, "rmse_sgd": float(np.sqrt(rows.squared_sum.sum() / rows.weight.sum())), "mae_sgd": float(rows.absolute_sum.sum() / rows.weight.sum()), "query_rows": int(rows.n.sum())})
    result = pd.DataFrame(scores).sort_values("rmse_sgd")
    result.to_csv(out / "inner_scores.csv", index=False)
    frame.to_csv(out / "inner_fold_sufficient_statistics.csv", index=False)
    return result


def select(config):
    cfg, out, cache = paths(config)
    groups = json.loads((out / "feature_groups.json").read_text())
    scores = screen_scores(out, cfg["inner_origins"]).set_index("model")
    def best(names):
        names = [n for n in names if n in groups]
        if not set(names) <= set(scores.index):
            raise ValueError(f"Incomplete earlier screening: {set(names) - set(scores.index)}")
        threshold = scores.loc[names, "rmse_sgd"].min() * (1 + cfg["selection_relative_tolerance"])
        eligible = [n for n in names if scores.loc[n, "rmse_sgd"] <= threshold]
        return min(eligible, key=lambda n: (len(groups[n]), names.index(n)))
    economy = best(["regional", "commercial_fixed", "commercial_kde200", "commercial_kde500", "commercial_kde1000", "regional_kde", "economy_all_selected"])
    landscape = best(["park_basic", "park_multiscale", "park_structure"])
    market = best(["market_raw", "market_smoothed"])
    kde = best(["commercial_kde200", "commercial_kde500", "commercial_kde1000"])
    strict = best(["regional", "market_raw", "market_smoothed"])
    jobs = {"F1_economy": groups[economy], "F2_landscape": groups[landscape],
            "F3_combined": list(dict.fromkeys(groups[economy] + groups[landscape])),
            "strict_incremental": groups[strict], "commercial_fixed": groups["commercial_fixed"],
            "commercial_kde_selected": groups[kde], "market_selected": groups[market]}
    selection = {"economy": economy, "landscape": landscape, "market": market, "kde": kde, "strict_incremental": strict,
                 "uses_outer_targets": False, "outer_jobs": jobs,
                 "factorial_policy": "Retrospective if selected economy uses static POIs; park boundaries are always retrospective. F0 inherited unchanged.",
                 "chosen_before_outer": True}
    immutable(out / "selection.json", selection)
    return selection


def child(config, stage, name, names=None, smoke=False):
    command = [sys.executable, str(Path(__file__)), "--config", str(config), "--stage", "one", "--fold", stage, "--variant", name]
    if names is not None:
        command += ["--columns-json", json.dumps(names)]
    if smoke:
        command += ["--smoke"]
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["prepare", "dry-run", "smoke", "screen", "select", "outer", "one", "all"], default="all")
    parser.add_argument("--fold", default="outer")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--columns-json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg, out, cache = paths(args.config)
    if args.stage == "dry-run":
        print(json.dumps({"configuration": cfg, "sources": json.loads((out / "source_manifest.json").read_text()) if (out / "source_manifest.json").exists() else "Run acquisition", "baseline_shape": [121005, 28995, 77], "fine_tuning": False}, indent=2))
        return
    if args.stage in ("prepare", "all"):
        prepare(args.config)
    if args.stage in ("smoke", "all"):
        child(args.config, "outer", "baseline", smoke=True)
        child(args.config, "inner_2023-07", "park_multiscale", smoke=True)
    if args.stage in ("screen", "all"):
        groups = json.loads((out / "feature_groups.json").read_text())
        for name in groups:
            for origin in cfg["inner_origins"]:
                child(args.config, f"inner_{origin}", name)
            screen_scores(out, cfg["inner_origins"])
        scores = screen_scores(out, cfg["inner_origins"]).set_index("model")
        kde = min(("commercial_kde200", "commercial_kde500", "commercial_kde1000"), key=lambda n: scores.loc[n, "rmse_sgd"])
        market = min(("market_raw", "market_smoothed"), key=lambda n: scores.loc[n, "rmse_sgd"])
        # Two targeted combinations, selected entirely from earlier data.
        combinations = {"regional_kde": groups.get("regional", []) + groups.get(kde, []),
                        "economy_all_selected": groups.get("regional", []) + groups.get(kde, []) + groups.get(market, [])}
        for name, names in combinations.items():
            names = list(dict.fromkeys(names))
            if name in groups and groups[name] != names:
                raise ValueError("Earlier selected combination changed")
            groups[name] = names
            write_json(out / "feature_groups.json", groups)
            for origin in cfg["inner_origins"]:
                child(args.config, f"inner_{origin}", name, names)
            screen_scores(out, cfg["inner_origins"])
    if args.stage in ("select", "outer", "all"):
        selection = select(args.config)
        print(json.dumps(selection, indent=2), flush=True)
    if args.stage in ("outer", "all"):
        completed = {}
        for name, names in selection["outer_jobs"].items():
            key = tuple(names)
            if key in completed:
                write_json(out / f"alias_outer_{name}.json", {"same_features_as": completed[key], "columns": names})
            else:
                child(args.config, "outer", name, names)
                completed[key] = name
    if args.stage == "one":
        run_one(args.config, args.fold, args.variant, json.loads(args.columns_json) if args.columns_json else None, args.smoke)


if __name__ == "__main__":
    main()
