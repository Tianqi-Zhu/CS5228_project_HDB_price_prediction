"""Earlier-calibrated ensembles adding the saved LSTM-COE TabPFN variant."""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from analysis import compare_tabpfn_abcde as previous
from analysis import compare_tabpfn_reliability_macro as macro
from analysis.compare_local_economy_landscape import immutable, write_json
from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.feature_diversified_ensemble import combine_predictions
from analysis.experiments.temporal_feature_views import simplex_weights

OUT = REPORTS / "tabpfn_abcde_lstm_coe_ensemble"
MEMBERS = list("ABCDEF")
ORIGINS = previous.ORIGINS
CANDIDATES = ("ABDEF_refit", "ABCDEF_refit", "ABDE_plus_F", "ABCDE_equal_plus_F")
CONTROLS = ("A_single", "ABDE", "ABCDE", "ABCDE_equal", "BDE_existing")


def fit_simplex(actual, predictions, sample_weight):
    """Extend the existing exact face enumeration to six members; verify KKT."""
    p, y, v = (np.asarray(a, dtype=float) for a in (predictions, actual, sample_weight))
    if (p.ndim != 2 or not 1 <= p.shape[1] <= 6 or y.shape != (len(p),)
            or v.shape != y.shape or not all(np.isfinite(a).all() for a in (p, y, v))
            or (v <= 0).any()):
        raise ValueError("Finite aligned predictions/targets and positive weights required")
    v = v / v.sum()
    error = p - y[:, None]
    scale = max(float(np.sqrt(np.sum(v[:, None] * error**2) / p.shape[1])), 1.)
    q = (error / scale).T @ (v[:, None] * (error / scale))
    candidates = []
    for size in range(1, p.shape[1] + 1):
        for subset in combinations(range(p.shape[1]), size):
            face = q[np.ix_(subset, subset)]
            system = np.block([[face, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(system, np.r_[np.zeros(size), 1.], rcond=None)[0][:size]
            if (solution < -1e-9).any() or not np.isclose(solution.sum(), 1., atol=1e-8):
                continue
            w = np.zeros(p.shape[1])
            w[list(subset)] = np.maximum(solution, 0)
            w /= w.sum()
            candidates.append((float(np.sum(v * (error @ w)**2)), w))
    best = min(loss for loss, _ in candidates)
    ties = [(loss, w) for loss, w in candidates if loss <= best + max(1e-9, best * 1e-13)]
    _, w = min(ties, key=lambda item: (-item[1][0], np.count_nonzero(item[1]), tuple(-item[1])))
    # Necessary and sufficient first-order conditions for this convex problem.
    gradient = 2 * q @ w
    level = float(gradient @ w)
    assert np.max(abs(gradient[w > 1e-8] - level)) < 1e-7
    assert np.min(gradient - level) > -1e-7
    if p.shape[1] <= 5:
        old, _ = simplex_weights(y, p, sample_weight)
        np.testing.assert_allclose(w, old, atol=1e-10, rtol=0)
    return w


def metrics(frame, predictions):
    return previous.score(frame, predictions)


def fit_members(frame, names):
    w = fit_simplex(frame.actual, frame[list(names)], frame.weight)
    return np.array([w[list(names).index(m)] if m in names else 0. for m in MEMBERS])


def fit_candidates(calibration):
    """Receives calibration only; no later frame/labels are accessible here."""
    p = calibration[MEMBERS].to_numpy()
    abde = fit_members(calibration, "ABDE")
    abcde = fit_members(calibration, "ABCDE")
    equal = np.r_[np.ones(5) / 5, 0.]
    result = {"ABDE": abde, "ABCDE": abcde,
              "ABDEF_refit": fit_members(calibration, "ABDEF"),
              "ABCDEF_refit": fit_members(calibration, "ABCDEF")}
    for name, base in (("ABDE_plus_F", abde), ("ABCDE_equal_plus_F", equal)):
        blend = fit_simplex(calibration.actual, np.column_stack([p @ base, calibration.F]), calibration.weight)
        result[name] = blend[0] * base + blend[1] * np.array([0., 0., 0., 0., 0., 1.])
    for candidate, reference in (("ABDEF_refit", "ABDE"), ("ABCDEF_refit", "ABCDE")):
        assert metrics(calibration, p @ result[candidate])["rmse_sgd"] <= metrics(calibration, p @ result[reference])["rmse_sgd"] + 1e-8
    return result


def load_frames():
    previous.check_sources()
    macro.verify_sources()
    audit = json.loads((previous.OUT / "independent_audit.json").read_text())
    assert audit["verified"] and audit["weights_earlier_only"]
    causal = json.loads((macro.OUT / "causality_audit.json").read_text())
    assert causal["verified"]
    saved = pd.read_csv(previous.OUT / "predictions_all_origins.csv", float_precision="round_trip")
    frames = {}
    for origin in ORIGINS:
        base = previous.load_predictions(origin)
        f = saved.loc[saved.origin.eq(origin)].reset_index(drop=True).copy()
        for column in ["row_id", "origin", "month", "actual", "weight", *previous.MEMBERS]:
            np.testing.assert_array_equal(base[column], f[column])
        coe, meta = macro.read_prediction(origin, "coe_increment")
        assert meta["features"] == 85 and meta["gradients_prohibited"]
        for column in ("row_id", "origin", "month", "actual", "weight"):
            np.testing.assert_array_equal(f[column], coe[column])
        f["F"] = coe.predicted.to_numpy()
        assert np.isfinite(f[MEMBERS].to_numpy()).all()
        frames[origin] = f
    early_ids = np.concatenate([frames[o].row_id.to_numpy() for o in ORIGINS[:-1]])
    assert len(np.unique(early_ids)) == 4000
    assert set(early_ids).isdisjoint(frames[ORIGINS[-1]].row_id)
    assert max(frames[ORIGINS[0]].month) < min(frames[ORIGINS[1]].month)
    assert max(frames[ORIGINS[1]].month) < min(frames[ORIGINS[-1]].month)
    return frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [previous.OUT / name for name in ("settings.json", "selection.json", "independent_audit.json", "predictions_all_origins.csv", "scores.csv")]
    sources += [macro.OUT / name for name in ("settings.json", "causality_audit.json", "national_forecasts.csv")]
    for origin in ORIGINS:
        sources += [previous.prediction_path(origin, m) for m in previous.MEMBERS]
        sources += [macro.OUT / f"{kind}_{origin}_coe_increment.{ext}" for kind, ext in (("predictions", "csv"), ("timing", "json"))]
    protected = [REPORTS / "tabpfn_top2_final_submission" / name for name in ("submission_tabpfn_ABCDE_equal_150k.csv", "submission_tabpfn_ABDE_weighted_150k.csv")]
    protected += [REPORTS / "tabpfn_api_77/submission_v3.5_plus_77_150k/submission_tabpfn_api.csv"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources + protected}
    immutable(OUT / "settings.json", dict(
        members={**json.loads((previous.OUT / "settings.json").read_text())["members"],
                 "F": "85-feature frozen Fast coe_increment: reliability, rent-only LSTM growth, rent+COE minus rent-only LSTM growth"},
        candidates=CANDIDATES, controls=CONTROLS, origins=ORIGINS,
        method="Nonnegative sum-one exact face optimization, no intercept; main weights fit only 4000 earlier rows through April2024",
        nested="September2022-June2023 fits weights; July2023-April2024 evaluates transfer; earlier-selected candidate chosen before later evaluation",
        add_on="ABDE_plus_F preserves fitted ABDE ratios; ABCDE_equal_plus_F preserves equal A-E ratios; only the F contribution is fitted",
        limitations="Reused development labels; F also includes reliability flags and rent-only LSTM forecasts, so this does not isolate COE causally. No new model training or submission.",
        source_sha256=hashes, code_sha256=sha256(Path(__file__)),
    ))
    frames = load_frames()
    early, transfer = [frames[o] for o in ORIGINS[:-1]]
    nested = fit_candidates(early)
    nested_scores = {name: metrics(transfer, transfer[MEMBERS].to_numpy() @ nested[name])["rmse_sgd"] for name in CANDIDATES}
    selected = min(CANDIDATES, key=lambda name: (nested_scores[name], np.count_nonzero(nested[name]), CANDIDATES.index(name)))
    pool = pd.concat([early, transfer], ignore_index=True)
    weights = fit_candidates(pool)
    choice = dict(selected=selected, nested_rmse_sgd=nested_scores,
                  main_weights={name: w.tolist() for name, w in weights.items()},
                  nested_weights={name: w.tolist() for name, w in nested.items()},
                  weight_fit_max_month=pool.month.max(), later_labels_used_for_weights_or_selection=False)
    immutable(OUT / "selection.json", choice)
    print("LOCKED earlier-only weights", json.dumps(choice), flush=True)
    score_rows, weight_rows, predictions, monthly = [], [], [], []
    for origin, fitted in ((ORIGINS[1], nested), (ORIGINS[2], weights)):
        f = frames[origin].copy()
        p = f[MEMBERS].to_numpy()
        for name, w in fitted.items():
            if name in ("ABDE", "ABCDE"):
                np.testing.assert_allclose(p @ w, f[name], atol=1e-8, rtol=0)
            else:
                f[name] = combine_predictions(p, w)
            weight_rows.append(dict(origin=origin, model=name, **dict(zip(MEMBERS, w))))
        f["ABCDEF_equal"] = p.mean(axis=1)
        f["selected"] = f[selected]
        for name in [*CONTROLS, "F", *CANDIDATES, "ABCDEF_equal", "selected"]:
            m = metrics(f, f[name])
            score_rows.append(dict(origin=origin, model=name, rows=len(f), **m,
                                   delta_vs_ABDE=m["rmse_sgd"] - metrics(f, f.ABDE)["rmse_sgd"],
                                   delta_vs_ABCDE_equal=m["rmse_sgd"] - metrics(f, f.ABCDE_equal)["rmse_sgd"]))
            for month, part in f.groupby("month", sort=True):
                monthly.append(dict(origin=origin, month=month, model=name, rows=len(part), **metrics(part, part[name])))
        predictions.append(f)
    later = predictions[-1]
    comparisons = []
    for candidate, reference in (("ABDEF_refit", "ABDE"), ("ABCDEF_refit", "ABCDE"),
                                 ("ABDE_plus_F", "ABDE"), ("ABCDE_equal_plus_F", "ABCDE_equal"),
                                 ("ABCDEF_equal", "ABCDE_equal"), ("selected", "ABDE"), ("selected", "ABCDE_equal")):
        comparisons.append(dict(candidate=candidate, reference=reference,
                                delta_rmse=metrics(later, later[candidate])["rmse_sgd"] - metrics(later, later[reference])["rmse_sgd"],
                                **previous.interval(later, candidate, reference)))
    scores = pd.DataFrame(score_rows)
    main_scores = scores.loc[scores.origin.eq(ORIGINS[-1])]
    for name, score in main_scores.set_index("model").iterrows():
        assert abs(metrics(later, later[name])["rmse_sgd"] - score.rmse_sgd) < 1e-10
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "predictions.csv", index=False, float_format="%.17g")
    scores.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(OUT / "weights.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_scores.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    later[MEMBERS].sub(later.actual, axis=0).corr().to_csv(OUT / "error_correlations.csv")
    # Independent serialized-output replay, including IDs and convex combinations.
    replay = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
    for origin, fitted in ((ORIGINS[1], nested), (ORIGINS[2], weights)):
        f = replay.loc[replay.origin.eq(origin)]
        np.testing.assert_array_equal(f.row_id, frames[origin].row_id)
        for name, w in fitted.items():
            np.testing.assert_allclose(f[name], combine_predictions(f[MEMBERS], w), atol=1e-8, rtol=0)
    for name, digest in hashes.items():
        assert sha256(ROOT / name) == digest, name
    write_json(OUT / "verification.json", dict(verified=True, earlier_rows=len(pool), later_rows=len(later),
        exact_ID_month_target_population_weight_alignment=True, main_labels_used_for_weights=False,
        original_ABDE_ABCDE_weights_and_predictions_reproduced=True, source_hashes_unchanged=True,
        submissions_unchanged=True, simplex_KKT_verified=True, existing_up_to_five_member_solver_reproduced=True,
        serialized_predictions_replayed=True, output_sha256={p.name: sha256(p) for p in OUT.glob("*.csv")}))
    write_report(main_scores, weights, comparisons, choice)
    print(main_scores[["model", "rmse_sgd", "delta_vs_ABDE", "delta_vs_ABCDE_equal"]].to_string(index=False), flush=True)


def write_report(scores, weights, comparisons, choice):
    lines = ["# TabPFN ensemble with LSTM-derived COE features", "",
             "F is the existing85-feature frozen TabPFN coe_increment variant. It appends reliability flags, rent-only national LSTM forecast growth and the difference between matched rent+COE and rent-only LSTM forecasts. The LSTMs forecast rental growth, not COE prices. Both use12-month windows,8 hidden units,100 fixed epochs,seed42 and10 direct horizons. Macro data stop one month before each original row's feature origin.", "",
             "All member predictions are reused with matching row IDs, months, actual rents and population weights. No models are retrained. This is an extension of the local Fast/DiagScale ensemble, not a blend with the hosted Plus API submission, which has no matching eight-estimator development predictions.", "",
             "## Earlier-only weight fitting", "",
             "September2022-June2023 (2000 sampled queries) calibrates weights for July2023-April2024 (2000 queries). The latter period chooses a candidate family. Both earlier windows then calibrate main weights through April2024, before the June2024-March2025 development evaluation. No later labels fit weights, select families, or tune an F-weight sweep. All fitted weights are nonnegative and sum to one; no intercept is added.", "",
             f"Earlier-transfer selected candidate: **{choice['selected']}**. Nested-transfer RMSEs: " + ", ".join(f"{k} S${v:.6f}" for k, v in choice["nested_rmse_sgd"].items()) + ".", "",
             "ABDEF_refit and ABCDEF_refit reoptimize all allowed members. ABDE_plus_F preserves the ABDE weight ratios and fits a single contribution for F. ABCDE_equal_plus_F preserves equal A-E ratios and fits a single F contribution. ABCDEF_equal is a fixed equal-weight sensitivity, not a selected winner.", "",
             "## Main reused development results", "",
             "| Model | RMSE (SGD) | Change vs weighted ABDE | Change vs ABCDE equal |", "| --- | ---: | ---: | ---: |"]
    for row in scores.itertuples():
        lines.append(f"| {row.model} | {row.rmse_sgd:.6f} | {row.delta_vs_ABDE:+.6f} | {row.delta_vs_ABCDE_equal:+.6f} |")
    lines += ["", "## Main weights", "", "| Model | A | B | C | D | E | F (LSTM-COE) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, w in weights.items():
        lines.append("| " + name + " | " + " | ".join(f"{v:.6f}" for v in w) + " |")
    lines += ["", "## Paired development comparisons", "", "Negative change favors the candidate. Intervals jointly resample contiguous three-month blocks of the ten development months,5000 repeats,seed42.", "", "| Candidate | Reference | RMSE change | 95% block interval |", "| --- | --- | ---: | --- |"]
    for row in comparisons:
        lines.append(f"| {row['candidate']} | {row['reference']} | {row['delta_rmse']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |")
    lines += ["", "## Interpretation and limits", "",
              "All later labels have been repeatedly examined in prior project work. These results are development evidence, not an untouched test or a Kaggle score. The intervals condition on fitted models/weights and omit model-selection and project-wide holdout-reuse uncertainty. A calibration optimum can improve while later RMSE worsens. F includes other appended inputs besides the COE increment, so any difference cannot be attributed exclusively to COE. Existing submissions remain unchanged.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_abcde_lstm_coe.py`. Settings retain source hashes; selection.json locks earlier-only weights; scores.csv,weights.csv,monthly_scores.csv,paired_comparisons.csv,predictions.csv and verification.json retain the comparison and replay checks.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
