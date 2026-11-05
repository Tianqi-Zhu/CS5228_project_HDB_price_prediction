# Strictly local stock correction of saved TabPFN ensemble backtests

This deterministic residual correction can change predictions only within 3 km of the frozen K&S site. It compares a local sector/FX/bias adjustment with the same model plus the company's six-month-lag return. All field scaling uses training-only weighted RMS, without centering or intercept, so zero spatial exposure guarantees exactly zero correction. No TabPFN backbone is retrained.

The first historical origin fits the correction, the second chooses among no correction, sector-only and company-local correction, and both earlier origins refit before the already examined later period. Fixed-origin stocks stop at the original fold cutoff; monthly-updated stocks use t-6 observations for each approval month. Both preserve the original frozen rental histories. The later period is reused development evidence, not new confirmation.

| scenario | candidate | baseline | earlier_selected | control_rmse | candidate_rmse | delta_rmse | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_origin | company_local | sector_local | False | 428.332 | 428.248 | -0.085 | -0.282 | 0.065 |
| fixed_origin | company_local | uncorrected | False | 397.688 | 428.248 | 30.559 | 6.002 | 49.202 |
| fixed_origin | sector_local | uncorrected | True | 397.688 | 428.332 | 30.644 | 6.086 | 49.160 |
| monthly_updated | company_local | sector_local | False | 405.552 | 405.844 | 0.292 | -0.579 | 0.976 |
| monthly_updated | company_local | uncorrected | False | 397.688 | 405.844 | 8.156 | -2.610 | 18.751 |
| monthly_updated | sector_local | uncorrected | True | 397.688 | 405.552 | 7.863 | -2.539 | 17.604 |

See [fixed-origin details](fixed_origin/report.md) and [monthly-updated details](monthly_updated/report.md) for calibration scores, full-sample errors and unchanged predictions beyond 3 km. Conditional bootstrap intervals omit model/selection uncertainty. This uses an earlier project's saved optimized feature-ensemble backtests, not the all-150k final submission model. Existing submissions are unchanged.
