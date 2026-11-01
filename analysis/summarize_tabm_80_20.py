"""Verify the direct TabM/TabPFN comparison and write its report and figures."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.compare_tabm_80_20 import OUT, CACHE, SEEDS, CANDIDATES, REFERENCE, summarize, write_json
from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics


def table(frame, columns):
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False,name=None):
        result.append("| "+" | ".join(f"{v:.3f}" if isinstance(v,float) else str(v) for v in row)+" |")
    return "\n".join(result)


def main():
    summarize()
    settings = json.loads((OUT / "settings.json").read_text())
    for path, expected in {**settings["source_hashes"], **settings["code_hashes"]}.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Source changed: {path}")
    for path, expected in settings["library_hashes"].items():
        if sha256(ROOT / ".venv/Lib/site-packages" / path) != expected:
            raise ValueError(f"Library changed: {path}")
    scores = pd.read_csv(OUT / "scores.csv")
    month = pd.read_csv(OUT / "monthly_scores.csv")
    groups = pd.read_csv(OUT / "size_group_scores.csv")
    selection = json.loads((OUT / "selection.json").read_text())
    calibration = [json.loads((OUT / "calibration" / name / "training.json").read_text()) for name in CANDIDATES]
    calibration_frame = pd.DataFrame([{k:r[k] for k in ("name","calibration_rmse_sgd","best_epoch","epochs_run",
                                                       "seconds","parameters","peak_vram_gib")} for r in calibration])
    calibration_frame.to_csv(OUT / "calibration_scores.csv",index=False,float_format="%.8f")
    verify = {"source_hashes_match": True,"library_hashes_match": True,"outer_rows":28995,
              "outer_labels_used_for_selection":False,"models":{}}
    for label in [f"calibration/{n}" for n in CANDIDATES]+[f"final_seed{s}" for s in SEEDS]:
        training = json.loads((OUT / label / "training.json").read_text())
        assert sha256(CACHE / label / "model.pt") == training["model_sha256"]
        assert sha256(CACHE / label / "preprocessor.joblib") == training["preprocessor_sha256"]
        verify["models"][label] = {"model_sha256":training["model_sha256"],
                                   "preprocessor_sha256":training["preprocessor_sha256"]}
        if label.startswith("calibration/"):
            assert training["checkpoint_replay_max_difference"] == 0.
        else:
            frame = pd.read_csv(OUT / label / "predictions.csv")
            record = json.loads((OUT / label / "score.json").read_text())
            assert sha256(OUT / label / "predictions.csv") == record["predictions_sha256"]
            assert frame.row_id.is_unique and len(frame)==28995 and np.isfinite(frame.predicted).all()
            assert abs(metrics(frame.actual,frame.predicted)["rmse_sgd"]-record["rmse_sgd"]) < 1e-7
    write_json(OUT / "verification.json",verify)

    # Paired resampling of entire months; transactions are not treated as IID.
    primary = pd.read_csv(OUT / "final_seed42/predictions.csv")
    baseline = pd.read_csv(REFERENCE)
    group = pd.DataFrame({"month":primary.month,"tabm":(primary.predicted-primary.actual)**2,
                          "tabpfn":(baseline.predicted-baseline.actual)**2}).groupby("month").agg(
                              n=("tabm","size"),tabm=("tabm","sum"),tabpfn=("tabpfn","sum"))
    draw = np.random.default_rng(42).integers(0,len(group),(20000,len(group)))
    count = group.n.to_numpy()[draw].sum(axis=1)
    difference = (np.sqrt(group.tabm.to_numpy()[draw].sum(axis=1)/count)
                  -np.sqrt(group.tabpfn.to_numpy()[draw].sum(axis=1)/count))
    bootstrap = {"primary": "TabM seed 42 minus frozen TabPFN", "draws":20000,"seed":42,
                 "months_improved":int((group.tabm<group.tabpfn).sum()),
                 "median_delta_sgd":float(np.median(difference)),
                 "interval_95pct_sgd":np.quantile(difference,[.025,.975]).tolist()}
    write_json(OUT / "month_bootstrap.json",bootstrap)
    tabm_scores = scores[scores.model.str.startswith("tabm_")]
    seed_mean,seed_std = float(tabm_scores.rmse_sgd.mean()),float(tabm_scores.rmse_sgd.std(ddof=1))
    primary_rmse = float(scores.set_index("model").loc["tabm_seed42","rmse_sgd"])
    delta = primary_rmse-float(scores.set_index("model").loc["tabpfn_frozen","rmse_sgd"])

    fig,axes = plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    short = scores.model.str.replace("tabpfn_","PFN ",regex=False).str.replace("tabm_","TabM ",regex=False)
    axes[0].barh(short.iloc[::-1],scores.rmse_sgd.iloc[::-1],color=["#287f83" if "TabM" in x else "#66718d" for x in short.iloc[::-1]])
    axes[0].set_xlabel("RMSE (SGD; lower is better)")
    axes[0].set_title("Same 77 source features and full 80/20 holdout")
    for model in ("tabpfn_frozen","tabpfn_original_diagscale","tabm_seed42"):
        part = month[month.model.eq(model)]
        axes[1].plot(part.month,part.rmse_sgd,marker="o",label=model)
    axes[1].tick_params(axis="x",rotation=45)
    axes[1].set_ylabel("Monthly RMSE (SGD)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Complete monthly errors")
    fig.savefig(OUT / "comparison.png",dpi=160)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4.5),layout="constrained")
    for name in CANDIDATES:
        curve = pd.DataFrame(json.loads((OUT / "calibration" / name / "learning_curve.json").read_text()))
        ax.plot(curve.epoch,curve.calibration_rmse_sgd,label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Earlier full-query RMSE (SGD)")
    ax.legend()
    fig.savefig(OUT / "learning_curves.png",dpi=160)
    plt.close(fig)

    recommendation = ("TabM improves this development holdout; confirm at additional forecast origins before replacing a submission."
                      if delta < 0 else "Retain TabPFN for this setup: the primary TabM run does not improve the same development holdout.")
    report = f"""# Direct TabM versus TabPFN comparison on the chronological 80/20 split

**Primary TabM seed-42 RMSE: S${primary_rmse:.3f}; difference from frozen TabPFN: {delta:+.3f} SGD.**

{recommendation} Across the three fixed seeds, RMSE mean is **S${seed_mean:.3f}**, with sample standard deviation **S${seed_std:.3f}**. Seeds are separate refits; no cross-seed prediction blend or seed selection is used.

## Matched data and target

All models are evaluated on the exact same 28,995 June 2024-March 2025 rows. TabM is trained from random initialization on all 121,005 January 2021-May 2024 rows. The 77-column categorical-month TabPFN source artifact is reused unchanged, including its pseudo-origin aligned training histories and fixed-May-2024 query histories. It is SHA-256 checked against the saved TabPFN feature manifest.

The target is the same residual rent: `y_train - train_baseline`; add `query_baseline` exactly once at prediction time. Source rent labels and all source rows are retained. No TabPFN embeddings, predictions or pretrained weights are inputs to TabM. This is a standalone model comparison, not the separate proposed area-adjustment/history-feature experiment. Hidden test data and the existing submission were not modified or scored.

## Selection before outer evaluation

All 102,829 rows before December 2023 fit each calibration candidate. The complete 18,176 December 2023-May 2024 rows select settings and stopping epoch. All 37 history inputs and nine temporal inputs for those queries are rebuilt using only labels through November 2023. This uses a full earlier calibration set, whereas some prior TabPFN adapter experiments used 600 sampled queries; their earlier calibration scores are not directly comparable.

Preprocessing fits only on each fitting partition: numerical medians, missingness indicators, means/scales and categorical vocabularies. Constant numerical representation columns are removed. The original nine categorical inputs use official TabM one-hot encoding with a reserved unknown/missing bucket. PLE bins are unsupervised fitting-data quantiles, 48 bins, dimension 16, version B. No outer inputs fit preprocessing, and no outer label chooses a hyperparameter, epoch, seed or checkpoint.

The official `tabm==0.0.3` implementation uses three 256-unit blocks, dropout 0.1 and 32 parameter-sharing internal members. Each member is trained with its own MSE before predictions are averaged. AdamW weight decay is 0.0003; tested learning rates are shown below. Targets are standardized on fitting rows only. Training uses BF16 autocast with FP32 loss/averaging, batch size 256, gradient clipping 1, maximum 100 epochs and patience 12. No broad hyperparameter search was performed.

{table(calibration_frame,["name","calibration_rmse_sgd","best_epoch","epochs_run","parameters","seconds","peak_vram_gib"])}

Earlier-selected configuration: **`{selection['candidate']}`**. All final seeds refit on the entire 80% portion with a fixed epoch count of **{selection['fixed_refit_epochs']}**. The remaining 20% is never an early-stopping set. Primary seed 42 and additional seeds 43/44 were declared before scoring.

## Complete outer results

{table(scores,["model","n","rmse_sgd","mae_sgd","bias_sgd","delta_vs_frozen_sgd"])}

Frozen TabPFN is the direct source-pipeline control. Original DiagScale and the numerically best MSE-only adapter are contextual development references; the latter's previously measured gain was tiny and not reliable across months. All saved reference losses are recomputed from aligned row predictions.

Primary TabM improves **{bootstrap['months_improved']} of 10 months** against frozen TabPFN. Paired month-bootstrap median RMSE difference is **{bootstrap['median_delta_sgd']:+.3f} SGD**, with a descriptive 95% interval **[{bootstrap['interval_95pct_sgd'][0]:+.3f}, {bootstrap['interval_95pct_sgd'][1]:+.3f}]** (20,000 resamples, seed 42). The seed standard deviation and the month-bootstrap interval measure different sources of variation.

## Size-related diagnostic groups

{table(groups,["model","group","n","rmse_sgd","mae_sgd","bias_sgd"])}

These groups are diagnostic only and did not choose the model. Higher absolute errors for larger properties can partly reflect higher rent levels; they do not by themselves establish a size-adjustment defect.

## Verification, artifacts and limits

Three targeted tests pass: fitting-only preprocessing and unknown categories, independent per-member loss, and official-model checkpoint replay with residual reconstruction. Full-schema GPU smoke tests passed for plain and PLE models. Every calibration checkpoint replays exactly after reload. All 28,995 outer IDs, months and actual labels align with the TabPFN references; predictions are finite and all losses are recomputed. `verification.json` records matching source, implementation, library, checkpoint and preprocessor hashes. Model weights and preprocessors are under `.cache/tabm_80_20/`; `scores.csv`, `monthly_scores.csv`, `size_group_scores.csv`, `calibration_scores.csv`, per-run learning curves, timing/memory and prediction files are under this report directory.

This repeatedly inspected ten-month development period is not an untouched test. Three final seeds do not replace additional chronological forecast origins. The supplied test extends to longer horizons. This is a bounded comparison of three declared configurations, not evidence about all possible TabM tuning choices. PyTorch peak allocation is not a measurement of resident physical VRAM.

## Reproduction

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-tabm.txt
.venv/Scripts/python.exe -m unittest tests.test_tabm_regression
.venv/Scripts/python.exe analysis/compare_tabm_80_20.py --smoke
.venv/Scripts/python.exe analysis/compare_tabm_80_20.py --stage all
.venv/Scripts/python.exe analysis/summarize_tabm_80_20.py
```

The runner supports `--stage prepare|calibration|final|summarize` and `--candidate`. Completed runs are reused with hash checks; unfinished training resumes optimizer, checkpoint, shuffle and dropout RNG states from the latest epoch. Main log: `reports/tabm_run.log`.

Method reference: [official TabM implementation and example](https://github.com/yandex-research/tabm).
"""
    (OUT / "report.md").write_text(report,encoding="utf-8")
    print(f"Verified report: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
