[Chinese version](README.zh-CN.md)

# Cleaning strategies: matched comparisons

| Material | Purpose |
|---|---|
| [Matched comparison table](matched_cleaning_comparison.csv) | Nine MLP, TabM and hybrid results with paired differences against each family's uncleaned baseline |
| [Compact row-level predictions](matched_predictions.npz) | IDs, labels and predictions for uncleaned MLP, mildly cleaned MLP and uncleaned hybrid; independently recomputable |
| [Prediction provenance](prediction-provenance.json) | Original CSV paths, commit, hashes and lossless-conversion notes |
| [Full original protocol](source/protocol.json) | Seeds 42/43/44, MLP 3 epochs, TabM 4 epochs, rules, calibration and aggregation |
| [Sample changes](source/cleaning_summary.json) | 105,837 fitting rows; mild cleaning removes 1,332, highest-value retention removes 15,537 |
| [Independent verification](../evidence-audit.json) | Three row-level recomputations; the other six checked against original summaries and source-file identity only |

Source: `mlp-eda-ensemble @322fa4c20d83f20e19328823203873a0f81e0f4a`. Full cleaning exploration belongs to appendix X02. These selections support M01 data-retention decisions and M04 error interpretation.

| Protocol | Fixed configuration |
|---|---|
| Fitting | 2021-01–2023-12, 105,837 rows; rebuild history, vocabulary, imputation and scaling after cleaning |
| Calibration | 2024-01–2024-03, unfiltered |
| Outer validation | 2024-04–2025-03, all 35,158 rows; do not delete low-rent validation records |
| Blend | Equal average over three seeds per family; hybrid then equally averages the two families |

Mild-cleaning MLP RMSE is 471.260898 versus 471.804200 uncleaned, an improvement of about 0.543301 SGD. The same rule worsens TabM and hybrid RMSE; `highest_only` worsens all three. Whether low rents represent different rental scope is unverified. Main currently keeps all records. This evidence supports caution with deletion rules, cannot replace F1's own cleaning ablation, and must not be ranked alongside F1 scores using different fitting cutoffs.
