# TabPFN-3 model-ready rental features

Numeric columns retain their values and missingness; categorical columns use pandas `category` with levels learned from fitting rows. TabPFN applies its own numeric and categorical transforms. Each rental-history input uses only preceding months; query histories freeze at the validation origin.

Bundles are compressed trusted local joblib files under `.cache/tabpfn_v3_features/`. Load with `joblib.load`, then use `X_train`, `y_train`, and `X_query` directly in `TabPFNRegressor`. `X_query` retains source row IDs as its index. Outer validation bundles also include `y_query` for evaluation. The final strict bundle has no query labels.

| Split | Train rows | Query rows | Last label month | First query month | File |
| --- | ---: | ---: | --- | --- | --- |
| 2023-01 | 70,183 | 17,754 | 2022-12 | 2023-01 | `.cache/tabpfn_v3_features/fold_2023-01.joblib` |
| 2023-07 | 87,937 | 26,905 | 2023-06 | 2023-07 | `.cache/tabpfn_v3_features/fold_2023-07.joblib` |
| 2024-04 | 114,842 | 35,158 | 2024-03 | 2024-04 | `.cache/tabpfn_v3_features/fold_2024-04.joblib` |
| 2025-03 | 146,968 | 50,000 | 2025-02 | 2025-03 | `.cache/tabpfn_v3_features/final_strict_2025-03.joblib` |

See `manifest.json` for ordered features, category names, hashes, and library versions.
