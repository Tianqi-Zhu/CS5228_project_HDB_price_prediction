"""Summarize completed worksite robustness and cached-model experiments."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.experiments.company_worksite_followup import table


def main():
    out = REPORTS / "company_worksite_followup"
    scores = pd.read_csv(out / "frozen_panel_scores.csv")
    placebo = pd.read_csv(out / "matched_placebo_summary.csv")
    row = pd.read_csv(out / "row_incremental_comparisons.csv")
    correction_path = out / "tabpfn_local_correction/scenario_comparisons.csv"
    correction = pd.read_csv(correction_path)
    focal = scores[(scores.site_id == "kulicke_soffa_serangoon") & (scores.band == "frozen")]
    bands = scores[(scores.site_id == "kulicke_soffa_serangoon") & (scores.band != "frozen") & (scores.model == "sector_fx_controls")]
    row_primary = row[(row.candidate == "klic_local") & row.baseline.isin(["klic_sector_local", "history77"])]
    head = correction[(correction.subset == "klic_within_1km") & (correction.candidate == "company_local")]
    paths = [out / "frozen_panel_scores.csv", out / "matched_placebo_summary.csv", out / "row_incremental_comparisons.csv", correction_path]
    metadata = {"source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
                "code_sha256": sha256(Path(__file__)), "source_period": "June 2024-March 2025 already examined", "submission_changed": False}
    (out / "findings_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    text = ["# Follow-up findings: company stocks and rents near engineering sites", "",
            "Recommendation: do not add a company-location stock feature to the current submission on this evidence. The tentative K&S relationship weakens under stronger market controls, is not unusual among matched distant neighborhoods, and does not deliver a stable incremental gain on saved TabPFN forecasts. This is a robustness study on already examined rental periods, not new causal identification or untouched confirmation.", "",
            "The prior K&S hypothesis remains fixed: 23A Serangoon North Avenue 5, 1 km proximity, six-month-lag company return. ST Engineering, Venture, Flex and Seatrium retain their previously selected site/radius/lag combinations. No later-period radius/lag re-selection occurs.", "",
            "## Aggregate local-rent check", "",
            table(focal, ["model", "development_ratio", "baseline_rmse", "stock_rmse", "delta_rmse_pct", "delta_ci_low", "delta_ci_high"]), "",
            "Adding US broad-market, semiconductor and SGD/USD controls reduces the incremental K&S gain relative to its matched control. The new control's absolute error is also larger: compare both the stock model and the control, rather than crediting a gain against a weaker comparator as proof of information. These are local-growth log-point errors, not SGD individual-rental errors.", "",
            table(placebo, ["model", "valid_placebos", "true_delta_pct", "placebo_median_delta_pct", "placebos_improving_more", "descriptive_rank_fraction"]), "",
            "Thirty-two placebo centers were matched on 2021-only housing characteristics and spread geographically; thirty-one have adequate later coverage. A descriptive rank fraction is not a randomized p-value. Six matched distant neighborhoods improve at least as much as the actual site after sector/currency controls.", "",
            table(bands, ["band", "delta_rmse_pct", "delta_ci_low", "delta_ci_high"]), "",
            "The strengthened-control results do not show a reliable monotonic distance-decay pattern. The wrong-company diagnostic favors KLIC over the five substitute stocks at this particular site, but that alone does not establish workplace causation or geographic uniqueness.", "",
            "## Individual-rental feature ablation with rent histories", "",
            "CatBoost keeps the original 77 aligned rent-history/property inputs and temporal baseline fixed, then adds distance/global signals, local sector/currency signals, and finally local company-return interactions. Earlier origins have monthly-stratified queries with population weights; the reused later origin evaluates all 28,995 rentals, including 418 within 1 km of K&S. Both stock-availability scenarios are tested. Rental histories remain frozen at each fold origin.", "",
            table(row_primary, ["scenario", "subset", "candidate", "baseline", "earlier_delta_rmse", "later_delta_rmse", "ci_low", "ci_high"]), "",
            "These deltas are individual-rental SGD RMSE. Sector-local controls distinguish company-specific interactions from a local industry-cycle effect. The 77-feature reference prevents a gain against an unnecessarily weak nuisance model from being mistaken for improvement over rent histories. Adding a feature can change CatBoost tree topology globally; a gain outside spatial support is not evidence that nearby employees caused rents to change. This is a single-seed controlled screen, not a matched TabPFN-backbone feature fit.", "",
            "## Exactly local correction on saved TabPFN ensemble backtests", "",
            "This second check preserves the saved TabPFN predictions. Correction inputs are multiplied by a 1 km Gaussian exposure truncated at 3 km. Weighted RMS scaling, no intercept and no centering make every correction exactly zero beyond 3 km. It cannot improve unrelated rows through a changed model topology. A first historical origin fits the correction; a second chooses among no correction, sector-only and company-local models; both earlier origins refit before the later reused period.", "",
            table(head, ["scenario", "candidate", "baseline", "control_rmse", "candidate_rmse", "delta_rmse", "ci_low", "ci_high"]), "",
            "The company-specific addition is negligible/inconclusive against sector/currency controls, and the full company-local correction does not beat the unchanged saved TabPFN near-site predictions in either availability scenario. Earlier local bias correction does not transfer reliably to the later regime. These are saved feature-ensemble backtests, not the all-150k submission model; no TabPFN backbone was retrained.", "",
            "## Limits and next evidence", "",
            "The original 63-scan search history still applies. Conditional bootstrap intervals omit model-fitting/search uncertainty, the later ten months have been reused, and workplace locations do not reveal employee headcounts, residences or commuting. Current geocodes and later-vintage stock/market histories are imperfect proxies for historical exposure. We need new rental months and evidence about actual local employment changes before claiming a causal company effect or promoting this feature.", "",
            "Existing submissions and original experiment outputs are preserved. [Full follow-up methods](report.md), [strictly local TabPFN details](tabpfn_local_correction/report.md), [distance/placebo chart](01_distance_and_matched_placebos.png), and [row feature-ablation chart](02_individual_rental_increment.png).", "",
            "Reproduce with `.venv/Scripts/python.exe analysis/acquire_worksite_market_controls.py`, `.venv/Scripts/python.exe analysis/compare_company_worksite_followup.py`, `.venv/Scripts/python.exe analysis/compare_worksite_tabpfn_correction.py`, then `.venv/Scripts/python.exe analysis/summarize_company_worksite_followup.py`.", ""]
    (out / "findings.md").write_text("\n".join(text), encoding="utf-8")
    print(out / "findings.md")


if __name__ == "__main__":
    main()
