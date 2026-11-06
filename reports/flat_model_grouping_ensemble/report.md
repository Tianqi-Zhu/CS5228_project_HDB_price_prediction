# Flat-model categories and the proposed 20-to-16 grouping

All 150,000 source training rows are retained. Fit on 121,005 January 2021-May 2024 rows; evaluate all 28,995 June 2024-March 2025 rows. No low-rent filtering is used.

A: original model categories (saved baseline). B: replace the direct `FLAT_MODEL_CLEAN` feature with the proposed group. C: retain the original and add categorical `FLAT_MODEL_GROUP`. B has 87 TabPFN columns; C has 88. Category levels are learned from fitting rows only. No other input, target, row, temporal baseline or historical grouping key is changed. Thus B tests the direct category feature while retaining the model-specific signature histories used by the existing ensemble.

Merges: type s1 / type s2 -> type s; model a maisonette / premium maisonette / improved maisonette -> maisonette. All other labels stay separate, including Model A2, DBSS, Apartment and Premium Apartment. This is the user's proposed grouping, not an official HDB classification.

The ensemble keeps 50% zch and 50% anniversary TabPFN. All seven zch weights, earlier-selected rounds, TabPFN checkpoint, two estimator views, seed 42 and full context remain fixed. There is no validation early stopping or reweighting.

## Per-category source statistics

Raw rents combine different years, towns and flat types; differences are descriptive, not estimates of a causal model premium. Full quantiles, standard deviations, areas, lease years, fitting/validation/test counts are in `flat_model_statistics.csv`.

| Model | Rows | Median rent | Mean rent | Min | Max | Fit rows | Validation rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| improved | 44,008 | 2,700 | 2,754.39 | 350 | 7,400 | 35,803 | 8,205 |
| model a | 42,745 | 2,700 | 2,743.19 | 300 | 6,000 | 34,144 | 8,601 |
| new generation | 27,380 | 2,450 | 2,483.68 | 490 | 5,100 | 22,130 | 5,250 |
| premium apartment | 12,312 | 2,900 | 2,857.14 | 600 | 5,500 | 9,872 | 2,440 |
| simplified | 6,794 | 2,500 | 2,582.52 | 500 | 5,000 | 5,514 | 1,280 |
| standard | 5,903 | 2,500 | 2,541.87 | 500 | 6,300 | 4,794 | 1,109 |
| apartment | 4,724 | 3,000 | 3,051.66 | 750 | 6,050 | 3,894 | 830 |
| maisonette | 2,346 | 3,000 | 3,141.89 | 900 | 6,150 | 1,918 | 428 |
| model a2 | 1,553 | 2,550 | 2,590.68 | 700 | 4,500 | 1,246 | 307 |
| dbss | 1,090 | 3,400 | 3,369.93 | 600 | 5,600 | 848 | 242 |
| type s1 | 299 | 4,600 | 4,356.69 | 1,500 | 6,000 | 237 | 62 |
| 2 room | 224 | 2,300 | 2,138.48 | 480 | 3,000 | 112 | 112 |
| adjoined flat | 218 | 3,325 | 3,648.03 | 800 | 7,600 | 178 | 40 |
| type s2 | 160 | 4,950 | 4,696.97 | 3,000 | 6,250 | 125 | 35 |
| model a maisonette | 140 | 3,175 | 3,300.09 | 2,000 | 5,600 | 114 | 26 |
| premium apartment loft | 51 | 3,800 | 3,676.47 | 1,300 | 5,000 | 36 | 15 |
| premium maisonette | 21 | 3,300 | 3,514.29 | 1,900 | 6,600 | 19 | 2 |
| terrace | 19 | 4,200 | 3,726.32 | 700 | 5,200 | 12 | 7 |
| 3gen | 12 | 3,500 | 3,550.00 | 2,500 | 4,500 | 9 | 3 |
| improved maisonette | 1 | 3,780 | 3,780.00 | 3,780 | 3,780 | 0 | 1 |

Top four categories cover 84.30%; 1,145 rows occur in categories with fewer than 500 rows. Only 621 source labels change under the proposed mapping.

## Validation performance

| Model | RMSE | MAE | Bias |
|---|---:|---:|---:|
| zch_original | 443.8631 | 322.9188 | -4.9029 |
| tabpfn_original | 441.9593 | 318.2158 | +4.1084 |
| blend_original | 440.9520 | 318.8879 | -0.3973 |
| zch_grouped | 443.9559 | 322.9574 | -3.9561 |
| tabpfn_grouped | 442.0895 | 316.5159 | +23.4519 |
| blend_grouped | 440.7707 | 317.7625 | +9.7479 |
| zch_original_plus_group | 444.0387 | 323.2272 | -7.0542 |
| tabpfn_original_plus_group | 443.1036 | 319.6094 | +0.7910 |
| blend_original_plus_group | 441.5420 | 319.7364 | -3.1316 |

grouped: ensemble RMSE change -0.1813 SGD; improved 6/10 months. Month-cluster bootstrap delta-MSE descriptive 95% interval [-645.096401569, 261.8665659005459]. Among 126 validation rows whose label changes, RMSE 457.379 -> 460.258.

original_plus_group: ensemble RMSE change +0.5900 SGD; improved 2/10 months. Month-cluster bootstrap delta-MSE descriptive 95% interval [186.20129657981272, 954.2968089156147]. Among 126 validation rows whose label changes, RMSE 457.379 -> 459.952.

`per_model_scores.csv`, `town_scores.csv` and `monthly_scores.csv` contain all component and ensemble errors. The same validation set has been used repeatedly for development. Any observed gain needs confirmation on additional chronological folds; these results do not establish Kaggle accuracy.

## Model-label reliability

The official [rental dataset](https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view) exposes six columns and no flat model; the official [resale dataset](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) includes flat model. Both schemas were checked for this audit. This establishes that the public rental table alone cannot supply that field, but does not establish how the supplied competition training table was built.

FLAT_MODEL is supplied in train.csv; original resale enrichment table/join code not found in available data files. Multiple models per observed key do not prove incorrect assignments.

Observed block_type: 0/14,828 keys have multiple model labels, covering 0 training rows. See `multiple_models_block_type.csv`.
Observed block_type_area_lease: 0/14,828 keys have multiple model labels, covering 0 training rows. See `multiple_models_block_type_area_lease.csv`.

Different units can share these keys; this audit cannot identify incorrect assignments without the original property-level enrichment source. These diagnostics do not relabel or exclude any row.

Original source files and existing submissions are preserved. This evaluation does not generate a replacement full-data submission.

Reproduce: `.venv/Scripts/python.exe analysis/compare_flat_model_grouping.py --stage all`.
