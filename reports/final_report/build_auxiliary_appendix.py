"""Reconcile auxiliary temporal-data results without training any model."""
import numpy as np
import pandas as pd


def build_auxiliary_appendix(root, here, train, table):
    sources = set()
    records = []

    def read(path):
        sources.add(path)
        return pd.read_csv(root / path)

    def close(actual, expected, tolerance=1e-6):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)

    economic = read("reports/economic_feature_ablation/forward_scores.csv")
    monthly = read("reports/economic_feature_ablation/monthly_errors.csv")
    feature_rows = []
    for start, part in economic.groupby("validation_start", sort=True):
        part = part.set_index("variant")
        for variant, saved in part.iterrows():
            months = monthly[(monthly.validation_start == start) & (monthly.variant == variant)]
            assert months.n.sum() == saved.n_validation
            assert months.approval_month.is_unique
            pooled = np.sqrt(np.average(months.rmse_sgd**2, weights=months.n))
            close(pooled, saved.rmse_sgd, tolerance=1e-4)  # CSV monthly scores have four decimals.
            records.append({"study": "direct_economic_features", "window": start,
                            "model": variant, "rmse_sgd": float(saved.rmse_sgd),
                            "verification": "pooled_saved_monthly_errors"})
        feature_rows.append([start, int(part.iloc[0].n_validation)] +
                            [f"{part.loc[key, 'rmse_sgd']:.2f}" for key in
                             ("E_amenities", "E_plus_coe", "E_plus_sg_stocks", "E_plus_both")])
    table("appendix_auxiliary_direct", ["Validation start", "Rows", "29-input control", "+ COE (35)", "+ Stocks (32)", "+ Both (38)"], feature_rows, "lrrrrr")

    forecast = read("reports/temporal_sequence_ablation/forecast_predictions.csv")
    cells = read("reports/temporal_sequence_ablation/input_processing_rmse_sgd.csv")
    selected = read("reports/temporal_sequence_ablation/selected_comparisons.csv")
    aggregate_rows = []
    for group, label in (("persistence", "Rent persistence"), ("rent", "LSTM: rent only"),
                          ("rent_coe", "LSTM: rent + COE"), ("rent_stock", "LSTM: rent + stocks"),
                          ("rent_stock_coe", "LSTM: rent + stocks + COE")):
        values = []
        for scenario in ("fixed_origin", "rolling"):
            if group == "persistence":
                saved = selected[(selected.target == "adjusted_mean_rent") &
                                 (selected.scenario == scenario) & (selected.selection == "overall")].iloc[0]
                assert saved.model.startswith("persistence__")
            else:
                saved = cells[(cells.target == "adjusted_mean_rent") & (cells.scenario == scenario) &
                              (cells.method == "lstm") & (cells.group == group)].iloc[0]
            cases = forecast[(forecast.target == "adjusted_mean_rent") & (forecast.phase == "outer") &
                             (forecast.scenario == scenario) & (forecast.model == saved.model)].copy()
            assert not cases.duplicated(["origin", "month", "horizon"]).any()
            assert cases.month.nunique() == 10
            assert (cases.latest_input_month <= cases.origin).all()
            assert (cases.month > cases.origin).all()
            mature = cases.dropna(subset=["last_training_target"])
            assert (mature.last_training_target <= mature.origin).all()
            cases["squared_error"] = (cases.predicted-cases.actual)**2
            rmse = float(np.sqrt(cases.groupby("month").squared_error.mean().mean()))
            close(rmse, saved.outer_rmse_sgd)
            values.append(f"{rmse:.3f}")
            records.append({"study": "adjusted_monthly_mean", "scenario": scenario,
                            "model": saved.model, "rmse_sgd": rmse,
                            "verification": "saved_forecasts_equal_target_month_weight"})
        aggregate_rows.append([label] + values)
    table("appendix_auxiliary_sequences", ["Input / processor", "Fixed-origin RMSE", "Rolling RMSE"], aggregate_rows, "lrr")

    transfer = read("reports/temporal_sequence_ablation/row_transfer_extended_scores.csv")
    transfer_months = read("reports/temporal_sequence_ablation/row_transfer_extended_monthly.csv")
    transfer_rows = []
    for candidate, label in (("persistence", "Rent persistence"), ("rent_lstm", "Rent-only LSTM (follow-up)"),
                              ("joint", "Rent + stock + COE Ridge"), ("joint_lstm", "Rent + stock + COE LSTM"),
                              ("rent_coe_lstm", "Rent + COE LSTM (follow-up)")):
        values = []
        for scenario in ("fixed_origin", "rolling"):
            saved = transfer[(transfer.phase == "outer") & (transfer.scenario == scenario) &
                             (transfer.candidate == candidate)].iloc[0]
            months = transfer_months[(transfer_months.phase == "outer") & (transfer_months.scenario == scenario) &
                                     (transfer_months.candidate == candidate)]
            assert months.month.is_unique and months.n.sum() == 28995
            close(np.sqrt(np.average(months.baseline_mse, weights=months.n)), saved.baseline_rmse_sgd)
            shift = months.gamma * months.correction
            mse = months.baseline_mse-2*shift*months.mean_residual+shift**2
            rmse = float(np.sqrt(np.average(mse, weights=months.n)))
            close(rmse, saved.corrected_rmse_sgd)
            values += [f"{rmse:.3f}", f"{saved.delta_rmse_sgd:+.3f}"]
            records.append({"study": "tree_correction", "scenario": scenario,
                            "model": candidate, "rmse_sgd": rmse,
                            "verification": "pooled_saved_monthly_residual_moments"})
        transfer_rows.append([label] + values)
    table("appendix_auxiliary_transfer", ["Correction to 471.631 tree control", "Fixed RMSE", r"$\Delta$", "Rolling RMSE", r"$\Delta$"], transfer_rows, "lrrrr")

    macro = read("reports/tabpfn_reliability_macro_80_20/scores.csv")
    macro_rows = []
    for member, label in (("baseline", "Original control"), ("reliability", "+ Reliability flags"),
                          ("rent_forecast", "+ Rent-only LSTM forecast"), ("coe_increment", "+ Incremental COE forecast")):
        saved = macro[(macro.origin == "2024-06") & (macro.member == member)].iloc[0]
        path = ("reports/tabpfn_anniversary_history_80_20/predictions_baseline77.csv" if member == "baseline" else
                f"reports/tabpfn_reliability_macro_80_20/predictions_2024-06_{member}.csv")
        frame = read(path).set_index("row_id").sort_index()
        assert frame.index.is_unique and len(frame) == 28995
        np.testing.assert_array_equal(frame.index, train.index[train.RENT_APPROVAL_DATE >= "2024-06"])
        np.testing.assert_array_equal(frame.actual, train.loc[frame.index, "MONTHLY_RENT"])
        np.testing.assert_array_equal(frame.month, train.loc[frame.index, "RENT_APPROVAL_DATE"])
        assert np.isfinite(frame.predicted).all()
        error = frame.predicted-frame.actual
        rmse = float(np.sqrt(np.mean(error**2)))
        close(rmse, saved.rmse_sgd)
        close(float(error.abs().mean()), saved.mae_sgd)
        macro_rows.append([label, int(saved.features), f"{rmse:.3f}", f"{saved.mae_sgd:.3f}", f"{saved.delta_vs_baseline_sgd:+.3f}"])
        records.append({"study": "tabpfn_macro_inputs", "model": member, "rmse_sgd": rmse,
                        "verification": "original_labels_aligned_row_predictions"})
    table("appendix_auxiliary_tabpfn", ["Nested inputs", "Fields", "RMSE", "MAE", r"$\Delta$RMSE"], macro_rows, "lrrrr")

    # Retain the source tables for secondary results described in the prose.
    for path in ("reports/stock_sector_signals/row_selected_comparison.csv", "reports/flat_type_coe_forecast/scores.csv",
                 "reports/economic_signals/forward_scores.csv",
                 "reports/temporal_sequence_ablation/followup_rent_coe_sensitivity.csv"):
        read(path)
    for path in ("reports/economic_feature_ablation/report.md", "reports/stock_sector_signals/report.md",
                 "reports/economic_signals/report.md",
                 "reports/flat_type_coe_forecast/report.md", "reports/temporal_sequence_ablation/report.md",
                 "reports/temporal_sequence_ablation/findings.md", "reports/tabpfn_reliability_macro_80_20/report.md",
                 "reports/tabpfn_reliability_macro_80_20/settings.json", "reports/final_report/build_auxiliary_appendix.py"):
        sources.add(path)
    pd.DataFrame(records).to_csv(here / "supplementary_auxiliary_scores.csv", index=False, lineterminator="\n")
    return {"no_models_refitted": True, "direct_features_reconciled_with_rounded_monthly_scores": True,
            "aggregate_forecast_rmse_recomputed": True, "tree_corrections_reconstructed_from_residual_moments": True,
            "tabpfn_prediction_ids_dates_labels_checked": True, "comparisons": records}, sorted(sources)
