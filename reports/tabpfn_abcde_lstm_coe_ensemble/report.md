# TabPFN ensemble with LSTM-derived COE features

F is the existing85-feature frozen TabPFN coe_increment variant. It appends reliability flags, rent-only national LSTM forecast growth and the difference between matched rent+COE and rent-only LSTM forecasts. The LSTMs forecast rental growth, not COE prices. Both use12-month windows,8 hidden units,100 fixed epochs,seed42 and10 direct horizons. Macro data stop one month before each original row's feature origin.

All member predictions are reused with matching row IDs, months, actual rents and population weights. No models are retrained. This is an extension of the local Fast/DiagScale ensemble, not a blend with the hosted Plus API submission, which has no matching eight-estimator development predictions.

## Earlier-only weight fitting

September2022-June2023 (2000 sampled queries) calibrates weights for July2023-April2024 (2000 queries). The latter period chooses a candidate family. Both earlier windows then calibrate main weights through April2024, before the June2024-March2025 development evaluation. No later labels fit weights, select families, or tune an F-weight sweep. All fitted weights are nonnegative and sum to one; no intercept is added.

Earlier-transfer selected candidate: **ABCDEF_refit**. Nested-transfer RMSEs: ABDEF_refit S$570.422354, ABCDEF_refit S$566.834514, ABDE_plus_F S$570.422354, ABCDE_equal_plus_F S$567.802277.

ABDEF_refit and ABCDEF_refit reoptimize all allowed members. ABDE_plus_F preserves the ABDE weight ratios and fits a single contribution for F. ABCDE_equal_plus_F preserves equal A-E ratios and fits a single F contribution. ABCDEF_equal is a fixed equal-weight sensitivity, not a selected winner.

## Main reused development results

| Model | RMSE (SGD) | Change vs weighted ABDE | Change vs ABCDE equal |
| --- | ---: | ---: | ---: |
| A_single | 455.071028 | +0.283616 | +0.367041 |
| ABDE | 454.787412 | +0.000000 | +0.083425 |
| ABCDE | 455.139341 | +0.351929 | +0.435353 |
| ABCDE_equal | 454.703987 | -0.083425 | +0.000000 |
| BDE_existing | 455.171873 | +0.384461 | +0.467886 |
| F | 456.905108 | +2.117696 | +2.201121 |
| ABDEF_refit | 454.787412 | +0.000000 | +0.083425 |
| ABCDEF_refit | 455.139341 | +0.351929 | +0.435353 |
| ABDE_plus_F | 454.787412 | +0.000000 | +0.083425 |
| ABCDE_equal_plus_F | 454.703987 | -0.083425 | +0.000000 |
| ABCDEF_equal | 454.942234 | +0.154822 | +0.238247 |
| selected | 455.139341 | +0.351929 | +0.435353 |

## Main weights

| Model | A | B | C | D | E | F (LSTM-COE) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ABDE | 0.410716 | 0.139229 | 0.000000 | 0.199630 | 0.250425 | 0.000000 |
| ABCDE | 0.072886 | 0.000000 | 0.927114 | 0.000000 | 0.000000 | 0.000000 |
| ABDEF_refit | 0.410716 | 0.139229 | 0.000000 | 0.199630 | 0.250425 | 0.000000 |
| ABCDEF_refit | 0.072886 | 0.000000 | 0.927114 | 0.000000 | 0.000000 | 0.000000 |
| ABDE_plus_F | 0.410716 | 0.139229 | 0.000000 | 0.199630 | 0.250425 | 0.000000 |
| ABCDE_equal_plus_F | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.000000 |

## Paired development comparisons

Negative change favors the candidate. Intervals jointly resample contiguous three-month blocks of the ten development months,5000 repeats,seed42.

| Candidate | Reference | RMSE change | 95% block interval |
| --- | --- | ---: | --- |
| ABDEF_refit | ABDE | +0.000000 | [-0.000000, +0.000000] |
| ABCDEF_refit | ABCDE | -0.000000 | [-0.000000, +0.000000] |
| ABDE_plus_F | ABDE | +0.000000 | [+0.000000, +0.000000] |
| ABCDE_equal_plus_F | ABCDE_equal | +0.000000 | [-0.000000, +0.000000] |
| ABCDEF_equal | ABCDE_equal | +0.238247 | [+0.183650, +0.305484] |
| selected | ABDE | +0.351929 | [-0.034030, +1.091831] |
| selected | ABCDE_equal | +0.435353 | [+0.150625, +1.004163] |

## Interpretation and limits

All later labels have been repeatedly examined in prior project work. These results are development evidence, not an untouched test or a Kaggle score. The intervals condition on fitted models/weights and omit model-selection and project-wide holdout-reuse uncertainty. A calibration optimum can improve while later RMSE worsens. F includes other appended inputs besides the COE increment, so any difference cannot be attributed exclusively to COE. Existing submissions remain unchanged.

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_abcde_lstm_coe.py`. Settings retain source hashes; selection.json locks earlier-only weights; scores.csv,weights.csv,monthly_scores.csv,paired_comparisons.csv,predictions.csv and verification.json retain the comparison and replay checks.
