# TabPFN on <=S$1,500-filtered context

Chronological origin: June 2024. Filtered context: 117,560 through May 2024; all 28,995 June 2024-March 2025 validation rows retained. The same 77 static, spatial, causal rent-history and local temporal forecast features as the strongest retained frozen TabPFN baseline were used. Every target-derived history and baseline was rebuilt using only the filtered pre-origin context. Frozen TabPFN-3.5 Fast weights, two estimators, full context, seed 42, no subsampling; fit computes a context cache rather than updating neural weights.

| Validation group | Model | Rows | RMSE (S$) | MAE (S$) | Bias (S$) |
| --- | --- | ---: | ---: | ---: | ---: |
| all | unfiltered_77 | 28,995 | 455.396 | 327.794 | 10.792 |
| all | filtered_1500_77 | 28,995 | 463.194 | 330.072 | 28.895 |
| low_rent_validation | unfiltered_77 | 323 | 1669.336 | 1603.253 | 1603.253 |
| low_rent_validation | filtered_1500_77 | 323 | 1745.386 | 1683.941 | 1683.941 |
| other_validation | unfiltered_77 | 28,672 | 422.290 | 313.425 | -7.147 |
| other_validation | filtered_1500_77 | 28,672 | 427.373 | 314.821 | 10.250 |

Filtered minus unfiltered all-row RMSE: **S$+7.799** (negative favors filtering). The reference is the saved native-default two-view 77-feature frozen baseline on the identical validation rows. These development labels have been examined in earlier experiments; this is a controlled development comparison, not an untouched test.

The price rule does not establish that any removed record is a bedroom-only lease. Validation includes rows that would meet the rule, preserving a realistic population metric.

Artifacts: [scores](scores.csv), [paired predictions](comparison_predictions.csv), [feature manifest](feature_manifest.json), [GPU timing](timing.json).

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_filtered_1500_80_20.py --stage all`.
