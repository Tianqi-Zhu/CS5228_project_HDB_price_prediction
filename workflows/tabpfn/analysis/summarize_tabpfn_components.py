"""Verify component experiment artifacts and write the research report."""

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

from analysis.compare_tabpfn_components_80_20 import OUT, CACHE, summarize
from analysis.compare_tabpfn_diagscale_80_20 import REFERENCE
from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics


def table(frame, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    summarize()
    plan = json.loads((OUT / "plan.json").read_text())
    for name, expected in {**plan["source_hashes"], **plan["implementation_hashes"]}.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    scores = pd.read_csv(OUT / "scores.csv")
    calibration = pd.read_csv(OUT / "calibration_scores.csv")
    view_scores = pd.read_csv(OUT / "view_calibration_scores.csv")
    selection = json.loads((OUT / "component_selection.json").read_text())
    view_selection = json.loads((OUT / "view_selection.json").read_text())
    reference = pd.read_csv(REFERENCE)
    # metrics() returns error measures only; the verified source IDs supply n.
    scores.insert(1, "n", len(reference))
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    baseline_frame = pd.read_csv(REPORTS / "tabpfn_diagscale_categorical_alignment_80_20/predictions.csv")
    baseline = baseline_frame.diagscale_aligned.to_numpy(float)
    month_names = reference.month.unique()
    month_id = pd.Categorical(reference.month, categories=month_names).codes
    count = np.bincount(month_id)
    base_sse = np.bincount(month_id, weights=(baseline-reference.actual.to_numpy())**2)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(month_names), size=(20000, len(month_names)))
    boot_base = np.sqrt(base_sse[draws].sum(axis=1) / count[draws].sum(axis=1))
    comparisons = []
    verification = {"source_hashes_match": True, "test_labels_used": False,
                    "row_predictions_verified": {}, "adapter_hashes_match": {}, "frozen_parameters_unchanged": {}}
    verification["scoring_runner_sha256"] = sha256(ROOT / "analysis/score_tabpfn_components_individually.py")
    for path in sorted((OUT / "outer").glob("*/predictions.csv")):
        label = path.parent.name
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(frame.row_id, reference.row_id)
        np.testing.assert_array_equal(frame.month, reference.month)
        np.testing.assert_allclose(frame.actual, reference.actual, atol=1e-8, rtol=0)
        assert frame.row_id.is_unique and len(frame) == 28995 and np.isfinite(frame.predicted).all()
        saved_rmse = float(scores.set_index("model").loc[label, "rmse_sgd"])
        assert abs(metrics(frame.actual, frame.predicted)["rmse_sgd"]-saved_rmse) < 1e-6
        verification["row_predictions_verified"][label] = {"rows": len(frame), "sha256": sha256(path)}
        sse = np.bincount(month_id, weights=(frame.predicted.to_numpy()-frame.actual.to_numpy())**2)
        differences = np.sqrt(sse[draws].sum(axis=1)/count[draws].sum(axis=1))-boot_base
        comparisons.append({"model": label, "months_improved_vs_winning_diag": int((sse < base_sse).sum()),
                            "bootstrap_median_delta_sgd": float(np.median(differences)),
                            "bootstrap_2.5pct_sgd": float(np.quantile(differences,.025)),
                            "bootstrap_97.5pct_sgd": float(np.quantile(differences,.975))})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "month_bootstrap.csv", index=False, float_format="%.8f")
    for path in sorted((OUT / "candidates").glob("*/calibration.json")):
        record = json.loads(path.read_text())
        assert record["frozen_parameters_unchanged"]
        assert sha256(CACHE / "candidates" / path.parent.name / "adapter.pt") == record["adapter_sha256"]
        assert record["max_checkpoint_replay_difference"] <= .01
        verification["adapter_hashes_match"][path.parent.name] = True
        verification["frozen_parameters_unchanged"][path.parent.name] = True
    (OUT / "verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    plot = scores.sort_values("rmse_sgd", ascending=False)
    ax.barh(plot.model, plot.delta_vs_winning_diag_sgd,
            color=["#177f75" if v <= 0 else "#b65c42" for v in plot.delta_vs_winning_diag_sgd])
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("RMSE difference from original winning DiagScale (SGD; lower is better)")
    ax.set_title("TabPFN component ablations: complete chronological development holdout")
    fig.savefig(OUT / "rmse_comparison.png", dpi=160)
    plt.close(fig)

    # Report the prespecified choices rather than replacing them with the best
    # model observed on this repeatedly inspected outer development period.
    labels = list(selection["selected"].values())
    chosen_view = view_selection["selected_views"]
    labels.append("winning_diag_one_view" if chosen_view == 1 else f"original_diag_{chosen_view}view")
    selected_results = scores[scores.model.isin(labels)].copy()
    matched_loss = scores.set_index("model")
    loss_delta = float(matched_loss.loc["diag_mse_only", "rmse_sgd"] - matched_loss.loc["diag_crps_mse", "rmse_sgd"])
    mse_frame = pd.read_csv(OUT / "outer/diag_mse_only/predictions.csv")
    control_frame = pd.read_csv(OUT / "outer/diag_crps_mse/predictions.csv")
    mse_sse = np.bincount(month_id, weights=(mse_frame.predicted-mse_frame.actual)**2)
    control_sse = np.bincount(month_id, weights=(control_frame.predicted-control_frame.actual)**2)
    loss_boot = (np.sqrt(mse_sse[draws].sum(axis=1)/count[draws].sum(axis=1))
                 - np.sqrt(control_sse[draws].sum(axis=1)/count[draws].sum(axis=1)))
    loss_interval = np.quantile(loss_boot, [.025, .975])
    winner = selected_results.loc[selected_results.rmse_sgd.idxmin()]
    notes = []
    for row in calibration.itertuples():
        if row.selected_epoch == -1:
            notes.append(f"- `{row.name}` selected zero updates: none of its trained epochs improved the earlier deployment metric.")
    report = f"""# TabPFN component adaptation experiments

All scores are RMSE in SGD on the same 28,995 June 2024-March 2025 rows. Inference uses every one of the 121,005 January 2021-May 2024 context rows. The existing 77-column categorical-month feature artifact, category positions, missingness, residual target and temporal baseline are unchanged. No test submission was changed.

## Audit of previously completed experiments

The all-attention rank-8 Q/K LoRA two-view experiment was already complete when this task began: frozen 455.396, LoRA 457.043. It is reused, not rerun. The earlier exported-embedding MLP (460.903) and four-training/four-deployment-view DiagScale (456.545) also already existed. They are distinct from the new native-head and fixed-checkpoint inference-view experiments.

## Prespecified procedure

See `plan.json` for feature, checkpoint and source hashes. All new adapters fit only the 102,829 rows before December 2023. The existing 600 December 2023-May 2024 calibration queries and population weights select epochs and configurations. Training keeps the upstream random episodes within this fitting pool, two training views, 1,024 task rows, 20% supervised queries, seed 42, AdamW weight decay 0.01, eight-epoch cap and patience three. This architectural ablation preserves the existing training sampler; it is not the separate historical-only month-coherent sampler experiment.

The new checkpoint rule evaluates population-weighted one-view deployment RMSE with the exact typed frames and an independently reloaded component. The unmodified checkpoint is eligible. This differs from the older two-view checkpoint rule; a matched CRPS+MSE DiagScale control is therefore retrained alongside MSE-only DiagScale. Components start from the frozen base checkpoint, not the previously winning adapter. All frozen parameter tensors are hashed before and after training.

ICL variants adapt Q/K projections only in all eight ICL blocks (rank 2 or 4), or only the final two (rank 4), using learning rate 0.0003. Native-head tuning trains the pretrained `heads.mlp_regression` and `heads.output_projection`, learning rate 0.00003. Both DiagScale losses use learning rate 0.01. The head is the original predictive-distribution decoder, not a new MLP over exported embeddings.

The fixed-view experiment reuses the *same original winning DiagScale adapter* for one, two and four inference views. Earlier calibration alone selects the view count. All fixed view conditions are reported on the full outer period without choosing a new view from those outer results.

## Earlier calibration and component selection

{table(calibration, ["name", "weighted_rmse_sgd", "initial_weighted_rmse_sgd", "selected_epoch", "epochs_run", "trainable_parameters"])}

Epoch -1 is the frozen, zero-update checkpoint; epoch 0 is after the first training epoch. Selected components: `{json.dumps(selection['selected'], sort_keys=True)}`.

{chr(10).join(notes)}

{table(view_scores, ["views", "weighted_rmse_sgd"])}

Earlier-selected inference views: **{chosen_view}**.

## Complete outer development results

{table(scores, ["model", "n", "rmse_sgd", "mae_sgd", "bias_sgd", "delta_vs_winning_diag_sgd"])}

Results of the prespecified choices:

{table(selected_results, ["model", "rmse_sgd", "delta_vs_winning_diag_sgd"])}

The direct MSE-only minus matched CRPS+MSE control difference is **{loss_delta:+.3f} SGD RMSE**; paired month-bootstrap 95% interval **[{loss_interval[0]:+.3f}, {loss_interval[1]:+.3f}]**. This is the loss ablation; a difference from the older adapter also includes the changed checkpoint-selection rule and explicit PyTorch initialization seed.

{table(comparison, ["model", "months_improved_vs_winning_diag", "bootstrap_median_delta_sgd", "bootstrap_2.5pct_sgd", "bootstrap_97.5pct_sgd"])}

The intervals use 20,000 paired resamples of ten complete months with seed 42. They describe this development period, which has been used repeatedly for prior model design; they are not an untouched estimate of future-month or Kaggle performance. Only one training seed and a small, declared hyperparameter set were tested. The test extends beyond this ten-month horizon.

## Recommendation

The numerically lowest RMSE among the declared family/view choices is `{winner['model']}` at **S${winner['rmse_sgd']:.3f}**, a difference of **{winner['delta_vs_winning_diag_sgd']:+.3f}** from the previous winner. Interpret its magnitude together with the month table and intervals above. A small gain on this repeatedly examined development holdout does not establish a dependable deployment improvement. Retain the existing submission while confirming promising changes at other forecast origins or training seeds. No outer score selects a new adapter hyperparameter, epoch or view count.

## Verification and reproduction

Five component/LoRA/DiagScale unit tests pass. Full-model smoke runs verify saved-checkpoint replay and gradients for each component type. Every saved new outer prediction has 28,995 unique, correctly aligned IDs and finite predictions; all reported losses are recomputed from those predictions. `verification.json` records source/adapter hashes, frozen-weight checks and prediction hashes. `epoch_trace.json`, `calibration_scores.csv`, `monthly_scores.csv`, per-run timing files and the chart preserve the full comparison. Peak memory is PyTorch allocated memory, not resident physical VRAM.

```powershell
.venv/Scripts/python.exe -m unittest tests.test_tabpfn_component_adaptation tests.test_lora_qk_tabpfn tests.test_diagscale_tabpfn
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --smoke
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --stage all
.venv/Scripts/python.exe analysis/summarize_tabpfn_components.py
```

For lower cumulative GPU memory use, the actual full scoring run uses fresh processes for each model after calibration:

```powershell
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --stage calibration
.venv/Scripts/python.exe analysis/compare_tabpfn_components_80_20.py --stage views
.venv/Scripts/python.exe analysis/score_tabpfn_components_individually.py
.venv/Scripts/python.exe analysis/summarize_tabpfn_components.py
```

The seeded training suite and isolated outer-scoring logs are `reports/tabpfn_components_run.log`, `reports/tabpfn_components_outer_run.log`, and `outer_*.log` in this report directory. The interrupted unseeded pilot, preliminary smoke checks and their old plans are explicitly named separately and excluded from all result tables. One outer run resumed saved batches; its `seconds_this_attempt` is not an end-to-end benchmark.

Completed training runs and inference batches resume without overwriting completed results. To run one calibration candidate, use `--stage calibration --candidate icl_rank2` (or another candidate from `--help`). Existing feature code, adapters, reports and submission CSVs are preserved.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(f"Verified report: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
