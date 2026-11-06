[Chinese version](02_anniversary_history.zh-CN.md)

# Second rental-prediction exploration round

Screen **88 configurations**:31 error corrections, 14 TabICL, 24 market normalizations, 11 anniversary features, eight compact/interactions. Select models/weights on earlier predictions.

**Same-block/type anniversary rents add useful information; target 400–430 remains unmet.**

| Validation window | Rows | Previous blend RMSE | Current main blend RMSE | Current MAE |
| --- | ---: | ---: | ---: | ---: |
| 2024-04 through 2025-03 | 35,158 | 463.7589 | **452.9386** | 328.4365 |
| 2024-06 through 2025-03 | 28,995 | 456.8220 | **443.8630** | 322.9187 |
| 2023-12 through 2025-03, sixteen months | 47,171 | 487.5770(original XGBoost) | **479.6351** | 349.2609 |

Prior TabPFN minimum on the same June origin/28,995 rows was455.0710. Current443.8630 improves about **11.2080 SGD**. TabPFN is an archived reference, not rerun here.

## Why this round helps

Old histories pool past contracts within block/type. New features query12/24/36 months earlier, exact/±one month, providing counts, means, ratios to contemporaneous type markets, ratio dispersion and low-rent shares.
This may retain recurring-rental levels, but without tenant/unit IDs it neither identifies renewals nor supports causal claims.
Validation queries match only months observed before fitting origin; unavailable long-horizon records stay missing/count0 without future-rent filling.

| Control | Original47-input LightGBM | 62-input LightGBM with anniversary statistics | Difference |
| --- | ---: | ---: | ---: |
| outer | 466.6302 | 457.8450 | -8.7852 |
| june | 458.3339 | 447.1701 | -11.1639 |

Match fitting/validation rows, base-tree settings and earlier stopping rules; each model selects duration earlier. The delta concerns this feature bundle/pipeline, not one field or causal renewals.

## Main blend this round

| Member | Weight |
| --- | ---: |
| normalized_market/compact_ratio15_market_last | 13.094% |
| anniversary/broad_lgb15 | 9.659% |
| anniversary/broad_lgb63 | 10.352% |
| anniversary/exact_lgb15 | 17.311% |
| anniversary/exact_lgb63 | 0.658% |
| anniversary/window_cat6 | 41.380% |
| anniversary/window_lgb63 | 7.546% |

Nonnegative weights sum1, cap0.6, and fit only 17,977 earlier2023-10–2024-03 predictions. Annual/June models refit before their origins with earlier durations/weights fixed.

## Other routes and negative results

- Second-level bias correction: 31 configurations, best annual 463.6825; no June improvement.
- Market/property-relative decomposition: 24 configurations; does not beat cohorts alone, but one lightweight model complements the blend.
- TabICLv2:14 context/target/feature configurations; early sampling200 rows/month. First full relative-history residual model scores annual 488.1767; earlier-weighted blend with old models about 463.7127. Not this round's winner.
- Anniversary TabICL scores 534.9542 on 1,200 matched samples versus CatBoost530.1784/main blend526.8643. Do not rank sampled against full-window scores.
- Eight more36-month-pruning, address/type/month-category and normalized-target trials. Earlier-selected compact ratio model yields blend annual 453.7023/June444.7842, both worse; retain the main result.

## Long-horizon checks and limits

The16-month check selects duration/weights on 2023-06–2023-11, then forecasts from 2023-12 through 2025-03. It does not reuse selection containing 2024 labels. Architecture/features still come from current development.
Beyond twelve months, 12-month cohorts cease to be observed history, while24-/36-month records remain. Evaluate long horizons separately.
This overlaps annual/June windows and is excluded from pooled rankings. Every period was previously examined: development, not independent final testing or Kaggle.

## Verification and reproduction

Independently recomputed106 predictions, checking raw labels, months, IDs, finite values and source hashes; 14 main-blend models reload consistently. Temporal tests: tests/test_second_wave.py.

Three-month block intervals describe fixed-prediction sensitivity in these months, excluding tuning, seeds and future markets. See monthly_comparison_*.csv and summary.json.

```powershell
python -m unittest tests.test_second_wave -v
python -m analysis.experiments.second_wave_anniversary screen
python -m analysis.experiments.second_wave_normalized screen
python -m analysis.experiments.second_wave_ensemble select
python -m analysis.experiments.second_wave_ensemble outer
python -m analysis.experiments.second_wave_ensemble june
python -m analysis.experiments.second_wave_stress
python -m analysis.experiments.second_wave_report
```

Models/row predictions:`output/models/round_02_anniversary_history/`. Main needs existing Conda modelling environment; TabICL dependencies:`.cache/foundation-deps/`,weights:`.cache/foundation-models/`.
TabICL:[author code](https://github.com/soda-inria/tabicl), [author weights](https://huggingface.co/jingang/TabICL). All inference is local.
