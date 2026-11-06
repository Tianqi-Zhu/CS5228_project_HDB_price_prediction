"""Flat-model statistics and matched category ablations of the best ensemble.

Run: .venv/Scripts/python.exe analysis/compare_flat_model_grouping.py --stage all
Only the direct model feature changes; historical keys/values remain fixed.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from analysis.config import ROOT, DATA, REPORTS
from analysis.data import sha256, prepare_main
from analysis.evaluation import metrics
from analysis.flat_model_groups import MERGES, VARIANTS, normalized_model, model_group, apply_model_group
from analysis.compare_tabpfn_anniversary_history_80_20 import source_bundle, matrices, CHECKPOINT

OUT = REPORTS / "flat_model_grouping_ensemble"
CACHE = ROOT / ".cache/flat_model_grouping_ensemble"
CONTROL = REPORTS / "zch_anniversary_tabpfn_blend/predictions_june.csv"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def category_stats(raw, key, test):
    grouped = raw.groupby(key, dropna=False)
    result = grouped.agg(rows=("MONTHLY_RENT", "size"), minimum_rent=("MONTHLY_RENT", "min"),
        mean_rent=("MONTHLY_RENT", "mean"), median_rent=("MONTHLY_RENT", "median"),
        std_rent=("MONTHLY_RENT", "std"), maximum_rent=("MONTHLY_RENT", "max"),
        median_floor_area_sqm=("FLOOR_AREA_SQM", "median"), median_lease_commence_year=("LEASE_COMMENCE_DATE", "median"),
        first_month=("RENT_APPROVAL_DATE", "min"), last_month=("RENT_APPROVAL_DATE", "max"))
    for quantile in [.10, .25, .75, .90]:
        result[f"rent_p{int(100*quantile)}"] = grouped.MONTHLY_RENT.quantile(quantile)
    result["share_pct"] = 100*result.rows/len(raw)
    for name, mask in [("training_rows", raw.RENT_APPROVAL_DATE.lt("2024-06")), ("validation_rows", raw.RENT_APPROVAL_DATE.ge("2024-06"))]:
        result[name] = raw.loc[mask].groupby(key).size().reindex(result.index, fill_value=0)
    result["test_rows"] = test.groupby(key).size().reindex(result.index, fill_value=0)
    return result.reset_index().sort_values("rows", ascending=False)


def stats():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    raw, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    for frame in [raw, test]:
        frame["MODEL"] = normalized_model(frame.FLAT_MODEL).fillna("__missing__")
        frame["MODEL_GROUP"] = model_group(frame.FLAT_MODEL).fillna("__missing__")
        frame["YEAR"] = frame.RENT_APPROVAL_DATE.str[:4].astype(int)
    mapping = raw[["MODEL", "MODEL_GROUP"]].drop_duplicates().sort_values("MODEL")
    assert len(mapping) == 20 and mapping.MODEL_GROUP.nunique() == 16
    mapping.to_csv(OUT / "model_mapping.csv", index=False)
    for key, filename in [("MODEL", "flat_model_statistics"), ("MODEL_GROUP", "grouped_model_statistics")]:
        result = category_stats(raw, key, test)
        assert int(result.rows.sum()) == len(raw)
        result.to_csv(OUT / f"{filename}.csv", index=False, float_format="%.6f")
    raw.groupby(["YEAR", "MODEL"]).MONTHLY_RENT.agg(rows="size", minimum_rent="min", mean_rent="mean", median_rent="median", maximum_rent="max").reset_index().to_csv(OUT / "model_statistics_by_year.csv", index=False)
    clean = prepare_main(raw)
    pd.crosstab(clean.MODEL, clean.FLAT_TYPE_CLEAN).to_csv(OUT / "model_by_flat_type.csv")
    # Observable consistency check, not verification of the unknown enrichment join.
    for col in ["TOWN", "STREET", "BLOCK"]:
        clean[col] = clean[col].astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    coarse = ["TOWN", "STREET", "BLOCK", "FLAT_TYPE_CLEAN"]
    diagnostics = {}
    for name, keys in [("block_type", coarse), ("block_type_area_lease", coarse+["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"])]:
        groups = clean.groupby(keys, dropna=False).agg(rows=("MODEL", "size"), distinct_models=("MODEL", "nunique"), models=("MODEL", lambda v:" | ".join(sorted(v.unique())))).reset_index()
        ambiguous = groups.loc[groups.distinct_models.gt(1)]
        ambiguous.to_csv(OUT / f"multiple_models_{name}.csv", index=False)
        diagnostics[name] = dict(total_keys=len(groups), keys_with_multiple_models=len(ambiguous), rental_rows_in_these_keys=int(ambiguous.rows.sum()))
    counts = raw.MODEL.value_counts()
    summary = dict(source_sha256=sha256(DATA / "train.csv"), test_sha256=sha256(DATA / "test.csv"),
        training_rows=len(raw), categories=len(mapping), grouped_categories=16,
        top_four_share_pct=float(100*counts.head(4).sum()/len(raw)),
        categories_with_fewer_than_500_rows=int(counts.lt(500).sum()),
        rows_in_categories_with_fewer_than_500=int(counts.loc[counts.lt(500)].sum()),
        rows_whose_label_changes=int(raw.MODEL.ne(raw.MODEL_GROUP).sum()),
        training_rows_whose_label_changes=int((raw.MODEL.ne(raw.MODEL_GROUP)&raw.RENT_APPROVAL_DATE.lt("2024-06")).sum()),
        validation_rows_whose_label_changes=int((raw.MODEL.ne(raw.MODEL_GROUP)&raw.RENT_APPROVAL_DATE.ge("2024-06")).sum()),
        observed_model_consistency=diagnostics,
        enrichment_provenance="FLAT_MODEL is supplied in train.csv; original resale enrichment table/join code not found in available data files. Multiple models per observed key do not prove incorrect assignments.")
    write_json(OUT / "statistics_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def tabpfn():
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from analysis.experiments.gpu_capacity import wait_for_gpu

    base = source_bundle()
    original_x, original_q = matrices("anniversary87", base)
    raw_models = pd.read_csv(DATA / "train.csv").FLAT_MODEL
    for variant in VARIANTS:
        output = OUT / f"predictions_tabpfn_{variant}.csv"
        if output.exists():
            record = json.loads((OUT / f"timing_tabpfn_{variant}.json").read_text())
            assert sha256(output) == record["prediction_sha256"]
            print(f"TABPFN reusing verified {variant}", flush=True)
            continue
        x, q = apply_model_group(original_x, original_q, variant,
                                raw_models.loc[original_x.index], raw_models.loc[original_q.index])
        for col in original_x:
            if col != "FLAT_MODEL_CLEAN" or variant == "original_plus_group":
                pd.testing.assert_series_equal(x[col], original_x[col])
                pd.testing.assert_series_equal(q[col], original_q[col])
        assert len(x)==121005 and len(q)==28995
        assert len(x.columns)==(87 if variant=="grouped" else 88)
        wait_for_gpu(min_free_gib=10, max_utilization=35, stable_samples=2, poll_seconds=10, status_path=OUT / "gpu_gate.json")
        torch.set_num_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        categories = [i for i,c in enumerate(x) if isinstance(x[c].dtype,pd.CategoricalDtype)]
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT),
            device="cuda", n_estimators=2, random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
            categorical_features_indices=categories, inference_config={"SUBSAMPLE_SAMPLES":None}, n_preprocessing_jobs=1)
        partial = CACHE / f"partial_tabpfn_rawlabels_{variant}.npy"
        values = np.load(partial) if partial.exists() else np.full(len(q),np.nan)
        assert values.shape==(len(q),)
        started=time.perf_counter()
        try:
            model.fit(x,np.asarray(base["y_train"])-np.asarray(base["train_baseline"]))
            assert model.n_train_samples_==len(x) and model.n_estimators_==2 and model.inference_config_.SUBSAMPLE_SAMPLES is None
            fitted=time.perf_counter()
            print(f"TABPFN fitted {variant}: {len(x.columns)} features",flush=True)
            for start in range(0,len(q),200):
                stop=min(start+200,len(q))
                if np.isfinite(values[start:stop]).all(): continue
                torch.cuda.empty_cache()
                values[start:stop]=np.asarray(model.predict(q.iloc[start:stop],output_type="mean"),float)+base["query_baseline"][start:stop]
                np.save(partial,values)
                if start%4000==0 or stop==len(q):print(f"TABPFN {variant}: {stop:,}/{len(q):,}",flush=True)
            assert np.isfinite(values).all()
            pd.DataFrame(dict(row_id=q.index,month=base["query_month"],actual=base["y_query"],predicted=values)).to_csv(output,index=False,float_format="%.17g")
            write_json(OUT / f"timing_tabpfn_{variant}.json",dict(features=list(x),context_rows=len(x),query_rows=len(q),
                fit_seconds=fitted-started,predict_seconds=time.perf_counter()-fitted,peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                checkpoint_sha256=sha256(CHECKPOINT),prediction_sha256=sha256(output),unchanged_histories=True))
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def summarize():
    frame=pd.read_csv(CONTROL).sort_values("row_id").reset_index(drop=True)
    frame=frame.rename(columns={"zch_original":"zch_original", "tabpfn87":"tabpfn_original"})
    frame=frame.drop(columns="locked_blend")
    frame["blend_original"]=(frame.zch_original+frame.tabpfn_original)/2
    for variant in VARIANTS:
        for component in ["zch","tabpfn"]:
            part=pd.read_csv(OUT / f"predictions_{component}_{variant}.csv").sort_values("row_id").reset_index(drop=True)
            np.testing.assert_array_equal(part.row_id,frame.row_id)
            np.testing.assert_array_equal(part.actual,frame.actual)
            np.testing.assert_array_equal(part.month,frame.month)
            frame[f"{component}_{variant}"]=part.predicted
        frame[f"blend_{variant}"]=(frame[f"zch_{variant}"]+frame[f"tabpfn_{variant}"])/2
    raw=pd.read_csv(DATA / "train.csv").loc[frame.row_id]
    frame["flat_model"]=normalized_model(raw.FLAT_MODEL).to_numpy()
    frame["town"]=raw.TOWN.to_numpy()
    models=["zch_original","tabpfn_original","blend_original"]+[f"{c}_{v}" for v in VARIANTS for c in ["zch","tabpfn","blend"]]
    scores=pd.DataFrame([dict(model=c,rows=len(frame),**metrics(frame.actual,frame[c])) for c in models])
    scores.to_csv(OUT / "scores.csv",index=False,float_format="%.10f")
    for key,filename in [("month","monthly_scores"),("flat_model","per_model_scores"),("town","town_scores")]:
        records=[dict(category=value,model=c,rows=len(part),**metrics(part.actual,part[c])) for value,part in frame.groupby(key) for c in models]
        pd.DataFrame(records).to_csv(OUT / f"{filename}.csv",index=False,float_format="%.10f")
    frame.to_csv(OUT / "comparison_predictions.csv",index=False,float_format="%.17g")
    base_error=(frame.blend_original-frame.actual)**2
    monthly=pd.read_csv(OUT / "monthly_scores.csv").pivot(index="category",columns="model",values="rmse_sgd")
    rng=np.random.default_rng(42)
    draws=rng.integers(10,size=(10000,10))
    comparisons=[]
    for variant in VARIANTS:
        name=f"blend_{variant}"
        deltas=(frame[name]-frame.actual)**2-base_error
        block=pd.DataFrame(dict(month=frame.month,delta=deltas)).groupby("month").agg(sse=("delta","sum"),n=("delta","size"))
        boot=block.sse.to_numpy()[draws].sum(axis=1)/block.n.to_numpy()[draws].sum(axis=1)
        affected=frame.flat_model.isin(MERGES)
        comparisons.append(dict(variant=variant,delta_rmse_sgd=float(scores.set_index("model").loc[name,"rmse_sgd"]-scores.set_index("model").loc["blend_original","rmse_sgd"]),
            months_improved=int((monthly[name]<monthly.blend_original).sum()),
            delta_mse_month_bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),
            affected_rows=int(affected.sum()),affected_original_rmse=metrics(frame.actual[affected],frame.blend_original[affected])["rmse_sgd"],
            affected_candidate_rmse=metrics(frame.actual[affected],frame.loc[affected,name])["rmse_sgd"]))
    write_json(OUT / "comparison_summary.json",dict(comparisons=comparisons,source_sha256=sha256(DATA / "train.csv"),baseline_predictions_sha256=sha256(CONTROL)))
    stat=pd.read_csv(OUT / "flat_model_statistics.csv")
    stat_summary=json.loads((OUT / "statistics_summary.json").read_text())
    lines=["# Flat-model categories and the proposed 20-to-16 grouping", "",
        "All 150,000 source training rows are retained. Fit on 121,005 January 2021-May 2024 rows; evaluate all 28,995 June 2024-March 2025 rows. No low-rent filtering is used.", "",
        "A: original model categories (saved baseline). B: replace the direct `FLAT_MODEL_CLEAN` feature with the proposed group. C: retain the original and add categorical `FLAT_MODEL_GROUP`. B has 87 TabPFN columns; C has 88. Category levels are learned from fitting rows only. No other input, target, row, temporal baseline or historical grouping key is changed. Thus B tests the direct category feature while retaining the model-specific signature histories used by the existing ensemble.", "",
        "Merges: type s1 / type s2 -> type s; model a maisonette / premium maisonette / improved maisonette -> maisonette. All other labels stay separate, including Model A2, DBSS, Apartment and Premium Apartment. This is the user's proposed grouping, not an official HDB classification.", "",
        "The ensemble keeps 50% zch and 50% anniversary TabPFN. All seven zch weights, earlier-selected rounds, TabPFN checkpoint, two estimator views, seed 42 and full context remain fixed. There is no validation early stopping or reweighting.", "",
        "## Per-category source statistics", "",
        "Raw rents combine different years, towns and flat types; differences are descriptive, not estimates of a causal model premium. Full quantiles, standard deviations, areas, lease years, fitting/validation/test counts are in `flat_model_statistics.csv`.", "",
        "| Model | Rows | Median rent | Mean rent | Min | Max | Fit rows | Validation rows |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in stat.itertuples():lines.append(f"| {r.MODEL} | {r.rows:,} | {r.median_rent:,.0f} | {r.mean_rent:,.2f} | {r.minimum_rent:,} | {r.maximum_rent:,} | {r.training_rows:,} | {r.validation_rows:,} |")
    lines += ["",f"Top four categories cover {stat_summary['top_four_share_pct']:.2f}%; {stat_summary['rows_in_categories_with_fewer_than_500']:,} rows occur in categories with fewer than 500 rows. Only {stat_summary['rows_whose_label_changes']:,} source labels change under the proposed mapping.","",
        "## Validation performance", "", "| Model | RMSE | MAE | Bias |", "|---|---:|---:|---:|"]
    for r in scores.itertuples():lines.append(f"| {r.model} | {r.rmse_sgd:.4f} | {r.mae_sgd:.4f} | {r.bias_sgd:+.4f} |")
    for r in comparisons:
        lines += ["",f"{r['variant']}: ensemble RMSE change {r['delta_rmse_sgd']:+.4f} SGD; improved {r['months_improved']}/10 months. Month-cluster bootstrap delta-MSE descriptive 95% interval {r['delta_mse_month_bootstrap_95pct']}. Among {r['affected_rows']} validation rows whose label changes, RMSE {r['affected_original_rmse']:.3f} -> {r['affected_candidate_rmse']:.3f}."]
    lines += ["", "`per_model_scores.csv`, `town_scores.csv` and `monthly_scores.csv` contain all component and ensemble errors. The same validation set has been used repeatedly for development. Any observed gain needs confirmation on additional chronological folds; these results do not establish Kaggle accuracy.", "",
        "## Model-label reliability", "",
        "The official [rental dataset](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view) exposes six columns and no flat model; the official [resale dataset](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) includes flat model. Both schemas were checked for this audit. This establishes that the public rental table alone cannot supply that field, but does not establish how the supplied competition training table was built.", "",
        stat_summary['enrichment_provenance'], ""]
    for key,info in stat_summary['observed_model_consistency'].items():lines.append(f"Observed {key}: {info['keys_with_multiple_models']:,}/{info['total_keys']:,} keys have multiple model labels, covering {info['rental_rows_in_these_keys']:,} training rows. See `multiple_models_{key}.csv`.")
    lines += ["", "Different units can share these keys; this audit cannot identify incorrect assignments without the original property-level enrichment source. These diagnostics do not relabel or exclude any row.", "",
        "Original source files and existing submissions are preserved. This evaluation does not generate a replacement full-data submission.", "",
        "Reproduce: `.venv/Scripts/python.exe analysis/compare_flat_model_grouping.py --stage all`.", ""]
    (OUT / "report.md").write_text("\n".join(lines),encoding="utf-8")
    print(scores.to_string(index=False),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["stats","tabpfn","trees","summarize","all"],default="all")
    stage=p.parse_args().stage
    os.environ.setdefault("TABPFN_NO_BROWSER","1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY","1")
    os.environ.pop("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS",None)
    if stage in ["stats","all"]:stats()
    if stage in ["tabpfn","all"]:tabpfn()
    if stage in ["trees","all"]:subprocess.run([sys.executable,str(ROOT / "analysis/train_flat_model_grouping_zch_worker.py")],check=True,cwd=ROOT)
    if stage in ["summarize","all"]:summarize()


if __name__=="__main__":main()
