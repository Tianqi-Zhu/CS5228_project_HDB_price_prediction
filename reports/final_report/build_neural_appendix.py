"""Build supplementary neural comparisons from saved, aligned development predictions."""
from pathlib import Path

import numpy as np
import pandas as pd


def build_neural_appendix(root: Path, here: Path, train, table):
    sources = set()
    records = []
    expected_ids = train.index[train.RENT_APPROVAL_DATE >= "2024-06"].to_numpy()

    def read(path):
        sources.add(path)
        return pd.read_csv(root / path)

    def score(path, column="predicted", origin=None):
        frame = read(path)
        if origin is not None:
            frame = frame[frame.origin == origin]
        assert frame.row_id.is_unique and len(frame) == 28995, path
        frame = frame.set_index("row_id").sort_index()
        np.testing.assert_array_equal(frame.index, expected_ids, err_msg=path)
        np.testing.assert_array_equal(frame.actual, train.loc[frame.index, "MONTHLY_RENT"], err_msg=path)
        np.testing.assert_array_equal(frame.month, train.loc[frame.index, "RENT_APPROVAL_DATE"], err_msg=path)
        assert np.isfinite(frame[column]).all(), path
        error = (frame[column] - frame.actual).to_numpy()
        return {"rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.abs(error).mean()), "bias": float(error.mean())}

    def checked(path, column, expected, origin=None):
        result = score(path, column, origin)
        np.testing.assert_allclose(result["rmse"], expected, rtol=0, atol=1e-6, err_msg=path)
        return result

    component_path = "reports/tabpfn_components_80_20/scores.csv"
    components = read(component_path).set_index("model")
    aligned = "reports/tabpfn_diagscale_categorical_alignment_80_20/predictions.csv"
    frozen = checked(aligned, "frozen", components.loc["frozen_one_view", "rmse_sgd"])
    fine_rows = []

    def fine(label, views, parameters, result, reference):
        fine_rows.append([label, views, parameters, f"{result['rmse']:.3f}",
                          f"{result['mae']:.3f}", f"{result['rmse']-reference:+.3f}"])
        records.append({"study": "fine_tuning", "model": label, "views": views,
                        "trainable_parameters": parameters, **result,
                        "delta_vs_matched_frozen": result["rmse"]-reference})

    fine("Frozen control", 1, "0", frozen, frozen["rmse"])
    for rate, suffix in (("0.003", "_lr0003"), ("0.005", "_lr0005"), ("0.01", ""), ("0.05", "_lr005")):
        directory = f"reports/tabpfn_diagscale_categorical_alignment{suffix}_80_20"
        saved = read(f"{directory}/scores.csv").set_index("model")
        result = checked(f"{directory}/predictions.csv", "diagscale_aligned", saved.loc["diagscale_aligned", "rmse_sgd"])
        fine(f"DiagScale, LR {rate}", 1, "8,576", result, frozen["rmse"])
    for name, label, parameters in (("diag_crps_mse", "DiagScale, matched CRPS+MSE", "8,576"),
                                    ("diag_mse_only", "DiagScale, matched MSE only", "8,576"),
                                    ("icl_rank2", "ICL Q/K LoRA, rank 2", "65,536"),
                                    ("regression_head", "Native regression decoder", "9,320,328")):
        result = checked(f"reports/tabpfn_components_80_20/outer/{name}/predictions.csv", "predicted", components.loc[name, "rmse_sgd"])
        fine(label, 1, parameters, result, frozen["rmse"])
    lora = "reports/tabpfn_lora_qk_lr00003_80_20/predictions_2views.csv"
    frozen_two = checked(lora, "frozen", components.loc["existing_frozen_two_views", "rmse_sgd"])
    fine("Frozen control", 2, "0", frozen_two, frozen_two["rmse"])
    result = checked(lora, "lora_qk", components.loc["existing_all_qk_lora_two_views", "rmse_sgd"])
    fine("All-attention Q/K LoRA, rank 8", 2, "299,008", result, frozen_two["rmse"])
    table("appendix_fine_tuning", ["Model / setting", "Views", "Updated weights", "RMSE", "MAE", r"$\Delta$RMSE"], fine_rows, "lrrrrr")

    sampling = read("reports/tabpfn_context_sampling/combined_outer_scores.csv").set_index("run_id")
    support_rows = []
    for strategy, label in (("S1", "Uniform"), ("S2", "Recency"), ("S3", "Local/property"), ("S4", "Mixture")):
        run = f"phase_b_{strategy}_seed42_episodes100_epochs8" + ("_patience8" if strategy in ("S1", "S2") else "")
        results = []
        for inference in ("S0", "S4"):
            saved = sampling.loc[f"{run}_infer{inference}"]
            assert saved.actual_training_epochs == 8
            result = checked(f"reports/tabpfn_context_sampling/{run}/outer_predictions_full_{inference}.csv", "predicted", saved.RMSE)
            results.append(f"{result['rmse']:.3f}")
            records.append({"study": "month_coherent_support", "model": strategy, "inference": inference, **result})
        support_rows.append([label] + results)
    table("appendix_fine_tuning_support", ["Training support", "Full 121,005-row inference", "Mixed 10,000-row inference"], support_rows, "lrr")

    scratch = read("reports/spatial_temporal_transformer/scores.csv").set_index("run")
    scratch_rows = []
    means = {}
    for variant, label in (("T0", "T0: property/history Transformer"), ("T1", "T1: T0 + rent sequence"),
                           ("T2", "T2: T1 + COE sequence"), ("T3", "T3: T2 + stock-return proxy")):
        results = []
        for seed in (42, 43, 44):
            saved = scratch.loc[f"outer/{variant}/seed{seed}"]
            assert not saved.outer_labels_used_for_selection
            result = checked(f"reports/spatial_temporal_transformer/outer/{variant}/seed{seed}/predictions.csv", "predicted", saved.rmse_sgd)
            results.append(result)
            records.append({"study": "scratch_transformer", "model": variant, "seed": seed, **result})
        means[variant] = float(np.mean([r["rmse"] for r in results]))
        scratch_rows.append([label] + [f"{r['rmse']:.3f}" for r in results] + [f"{means[variant]:.3f}"])
    decay_scores = read("reports/transformer_ranked_experiments/23_forced_decay/scores.csv").set_index("seed")
    decay_results = []
    for seed in (42, 43, 44):
        result = checked(f"reports/transformer_ranked_experiments/23_forced_decay/outer_seed{seed}.csv", "decayed_prediction", decay_scores.loc[seed, "decayed_rmse_sgd"])
        decay_results.append(result)
        records.append({"study": "scratch_transformer", "model": "decayed T0/T1", "seed": seed, **result})
    means["decayed"] = float(np.mean([r["rmse"] for r in decay_results]))
    scratch_rows.append(["Earlier-selected decayed T0/T1 blend"] + [f"{r['rmse']:.3f}" for r in decay_results] + [f"{means['decayed']:.3f}"])
    table("appendix_scratch_transformer", ["Model", "Seed 42", "Seed 43", "Seed 44", "Mean RMSE"], scratch_rows, "lrrrr")

    abcde_scores = read("reports/tabpfn_abcde_ensemble/scores.csv")
    expected = abcde_scores[(abcde_scores.origin == "2024-06") & (abcde_scores.model == "ABCDE_equal")].iloc[0].rmse_sgd
    abcde = checked("reports/tabpfn_abcde_ensemble/predictions_all_origins.csv", "ABCDE_equal", expected, origin="2024-06")
    expected_cde = abcde_scores[(abcde_scores.origin == "2024-06") & (abcde_scores.model == "selected")].iloc[0].rmse_sgd
    selected_cde = checked("reports/tabpfn_abcde_ensemble/predictions_all_origins.csv", "selected", expected_cde, origin="2024-06")
    reference_rows = []
    for label, views, result in (("Frozen TabPFN, original 77 inputs", "2", frozen_two),
                                  ("TabPFN CDE, earlier-selected family", "2", selected_cde),
                                  ("TabPFN ABCDE equal ensemble", "varied", abcde)):
        reference_rows.append([label, views, f"{result['rmse']:.3f}", f"{result['mae']:.3f}"])
        records.append({"study": "tabpfn_reference", "model": label, **result})
    table("appendix_transformer_tabpfn", ["Reference pipeline", "Views", "RMSE", "MAE"], reference_rows, "llrr")
    pd.DataFrame(records).to_csv(here / "supplementary_neural_scores.csv", index=False)
    for path in ("reports/tabpfn_components_80_20/plan.json", "reports/tabpfn_components_80_20/report.md",
                 "reports/tabpfn_diagscale_learning_rates_80_20/report.md", "reports/tabpfn_context_sampling/report.md",
                 "reports/spatial_temporal_transformer/settings.json", "reports/transformer_ranked_experiments/report.md"):
        sources.add(path)
    return {"all_prediction_ids_months_labels_checked": True, "all_reported_rmse_recomputed": True,
            "validation_rows_per_prediction": 28995, "scratch_seed_means": means,
            "abcde_rmse": abcde["rmse"], "comparisons": records}, sorted(sources)
