# Frozen TabPFN local economy and landscape experiments

**Retain the original 77-feature baseline.** The selected combination changes RMSE by only S$-0.051; its paired building-cluster 95% interval is [-0.705, +0.599], crossing zero. The strongest standalone economic result is the published census/income block, but it also fails to improve baseline RMSE. The selected park representation worsens RMSE. Fixed-radius commercial counts outperform KDE on the full holdout, reversing their earlier screening order; neither beats X0. There is no demonstrated robust improvement from these tested additions. Existing production/submission artifacts remain unchanged.

Selected economy: RMSE S$457.089, change +1.276 against frozen S$455.814. Selected landscape: RMSE S$456.762, change +0.949 against frozen S$455.814. Combined: RMSE S$455.763, change -0.051 against frozen S$455.814. Lagged rental fields: RMSE S$457.326, change +1.513 against frozen S$455.814. Published census including income: RMSE S$455.936, change +0.122 against frozen S$455.814.

The existing 77 features, rent residuals, pretrained checkpoint, full context, categorical encodings and inference settings are preserved. No TabPFN parameter is optimized. This is a repeatedly examined development holdout, not a new untouched test.

## Data and information policy

Public government sources acquired: Census 2020 (planning-area population and age composition), NEA market/food-centre points, NParks managed-park polygons, and URA MP2019 subzone polygons. Their source files total approximately 6.1 MiB; manifests preserve source URLs, observation/publication information and checksums. Existing mall coordinates are reused. Every one of 9,464 canonical buildings has one polygon match, agreeing with supplied planning area and subzone. No rows are removed.

A fifth 3.6 KB source provides Census 2020 household-income bands by planning area in a dataset last updated 28 July 2022. The separate regional_income ablation adds population/elderly share plus shares of households with no employed person, employed income below S$3,000, and income at least S$10,000, with availability flags (seven dimensions). Income from work includes employer CPF contributions, per the source. These are actual published band proportions, not estimated building incomes. Income features are unavailable before August 2022 forecast origins; the first earlier fold has no training-origin income coverage and that limitation is preserved. This predeclared regional ablation does not alter the already locked factorial selection. Its own source/settings/coverage/cache manifests are in ../tabpfn_local_economy_landscape_80_20_income/. The separate source revised in 2026 is excluded.

Census values enter only at origins from July 2021 after the 18 June 2021 release. Suppression/unmatched regions remain missing. Added rental fields use same flat type and labels observed by origin minus one month, with no self/current/future labels. Training features share X0's pseudo-origins; query origins remain fixed. The original 77 histories keep their existing availability policy.

POIs and park geometries are current mirrors without verified opening/closing or archived boundary versions. All experiments using these additions are **retrospective geography experiments**, including the factorial; they do not establish historical point-in-time gains. `strict_incremental` is the earlier-selected lagged rental-field addition; `regional_income` is the predeclared published-census ablation. Their availability claim concerns additions, since inherited X0 already includes static amenities.

Food features count food-service centres (not individual stalls/restaurants); malls are not a complete business register. KDE uses the normalized sum of Gaussian kernels in centres/km². EPSG:3414 is used for all distances and areas. Park fractions measure designated/managed park area, not satellite tree, grass, water or built-up probabilities. Polygon unions prevent overlap double counting. Landscape structure uses park-patch support only.

Commercial radius counts use 300/500/1,000 m; selected non-redundant inputs add food-centre counts at all three radii and mall counts at 300 m, since X0 already has mall counts at 500/1,000/2,000 m. Gaussian KDE uses sigma 200/500/1,000 m: sum(exp(-d²/(2 sigma²)))/(2 pi sigma²), converted to centres/km², without boundary correction. All category points contribute. Count-derived densities and nearest distances are cached but not concatenated into every candidate.

Park fractions use union area intersected with 100/300/500 m buffers, divided by actual buffer area. The Gaussian park feature approximates a sigma-200 m radial integral with piecewise-constant weights on these annuli. Structure adds largest connected park-patch fraction at 500 m and binary park/nonpark entropy. These are deterministic representations, not fitted encoders.

Rental fields cover 6/12-month windows at 500/1,000/2,000 m. Raw comparisons use means/counts (neighborhood medians were not computed). Selected shrinkage uses 20 pseudo-observations toward the same-flat-type national history. The continuous field uses sigma 500 m truncated at 2 km and a 6-month temporal half-life; its weighted support mass is not an effective sample size. A historical rent-per-square-metre field supplies a limited composition adjustment, not a full hedonic adjustment. Raw means remain missing with zero local observations; shrunk estimates may back off to national history. No temporal model uses query-period labels.

## Earlier representation comparisons

| model | rmse_sgd | mae_sgd | query_rows |
| --- | --- | --- | --- |
| park_structure | 608.311 | 481.858 | 1200 |
| market_smoothed | 608.353 | 478.305 | 1200 |
| park_basic | 609.900 | 475.936 | 1200 |
| commercial_kde200 | 610.290 | 476.732 | 1200 |
| commercial_kde1000 | 610.442 | 476.589 | 1200 |
| regional | 610.477 | 480.807 | 1200 |
| commercial_kde500 | 610.480 | 476.723 | 1200 |
| economy_all_selected | 612.583 | 480.543 | 1200 |
| regional_kde | 615.651 | 481.732 | 1200 |
| baseline | 615.701 | 480.872 | 1200 |
| park_multiscale | 616.594 | 483.909 | 1200 |
| commercial_fixed | 618.388 | 483.778 | 1200 |
| regional_density_retro | 619.341 | 484.576 | 1200 |
| regional_income | 621.501 | 485.525 | 1200 |
| market_raw | 624.615 | 488.501 | 1200 |

Two earlier ten-month folds keep all 60,079 / 87,937 context rows. Each fold has a fixed nested sample of 60 queries/month, chosen without labels and weighted to monthly population. All feature representations use the same sampled IDs. These are screening estimates, not complete-fold scores. Regions/KDE/market combinations are built only after individual screening; no outer labels choose the representation.

Frozen earlier selections: economy `commercial_kde200`, landscape `park_structure`, market `market_smoothed`. Complete column lists and selection audit: selection.json.

## Full chronological 80/20 results

| model | n | rmse_sgd | mae_sgd | bias_sgd | r2 | delta_rmse_sgd | relative_rmse_improvement_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0_baseline | 28995 | 455.814 | 329.748 | 0.768 | 0.516 | 0.000 | 0.000 |
| commercial_fixed | 28995 | 456.812 | 327.232 | 28.080 | 0.513 | 0.998 | -0.219 |
| F1_economy | 28995 | 457.089 | 327.718 | 24.970 | 0.513 | 1.276 | -0.280 |
| F2_landscape | 28995 | 456.762 | 332.666 | -12.192 | 0.514 | 0.949 | -0.208 |
| F3_combined | 28995 | 455.763 | 329.573 | -0.932 | 0.516 | -0.051 | 0.011 |
| regional_income | 28995 | 455.936 | 329.151 | 3.901 | 0.515 | 0.122 | -0.027 |
| strict_incremental | 28995 | 457.326 | 330.111 | 6.663 | 0.512 | 1.513 | -0.332 |

Context: 121,005 January 2021–May 2024 rows; evaluation: all 28,995 June 2024–March 2025 rows. Every prediction file has identical unique IDs, months and actual rents. F0 is loaded after a full-context 400-query replay agreeing within S$0.00001. Shared selected columns reuse predictions through explicit aliases.

## Factorial and uncertainty

Negative-MAE interaction `Score(F3)-Score(F1)-Score(F2)+Score(F0)` = 1.0626554864070954 SGD. Positive values indicate descriptive complementarity. This interaction alone does not prove statistical significance.

`cluster_bootstrap.json` contains 5,000 paired resamples clustered separately by building and month. Grouped results cover town, flat type, rent ranges defined by training-target quartiles, and high/low park/commercial bins defined by training-location medians. They do not adjust for repeated feature selection or development-holdout inspection. Entire-neighborhood spatial holdout was not run; the claim is chronological generalization only.

## Coverage and compute

| model | added_dimensions | fit_seconds | predict_seconds | peak_vram_gib | context_rows |
| --- | --- | --- | --- | --- | --- |
| commercial_fixed | 4 | 14.276 | 528.243 | 7.180 | 121005 |
| F1_economy | 2 | 14.044 | 501.799 | 7.199 | 121005 |
| F2_landscape | 7 | 14.417 | 507.533 | 7.153 | 121005 |
| F3_combined | 9 | 14.619 | 531.217 | 7.136 | 121005 |
| regional_income | 7 | 14.390 | 527.321 | 7.153 | 121005 |
| strict_incremental | 12 | 14.794 | 529.070 | 7.113 | 121005 |

VRAM figures are peak PyTorch-allocated memory, not total driver reservation. One nvidia-smi observation during the economic run reached 15,874 MiB of device usage; driver memory was not sampled continuously. All complete runs fit the RTX 4080 without a context reduction or memory retry.

Full feature/town missingness is recorded in feature_coverage.csv / coverage_by_town.csv. Source coverage for context, holdout and unlabelled test locations is in source_coverage_by_split.csv; test census availability is assessed at the last labelled month, not with future observations. New test rental fields/predictions were not generated. Unknown imagery/census is not zero-filled. No training-context reduction or prediction-row omission is permitted. Each inference job is isolated in a fresh process and saves resumable 200-row batches. The unlabelled competition test and existing submission are unchanged.

## Unavailable sources and next steps

Dynamic World probabilities and AlphaEarth embeddings were not extracted: Earth Engine credentials/project and imported exports are unavailable. Export scripts, dated import validation, train-only PCA at configurable 8/16/32 components, missingness handling and setup instructions are implemented; no satellite performance result is fabricated. The live AlphaEarth catalog lists observation years 2017–2024, but the product was publicly released on 30 July 2025, after this entire holdout. Strict historical AlphaEarth features would therefore be unavailable even with credentials. Retrospective observation-lagged embedding experiments must be labelled separately. Dynamic World likewise requires its actual 2022 product release as well as image dates.

Income/employment at building resolution, complete commercial establishments, historical park snapshots, optional supervised economic encoders/GPs, extra landscape fragmentation indices, permutation importance and a spatially held-out study were not run. The targeted deterministic/KDE comparisons precede those optional costs. Fixed-radius and KDE candidates add four versus two columns, respectively; this compares the declared representations rather than proving a dimension-matched intrinsic advantage of continuous fields. Additional simple/continuous-economy × simple-landscape combinations were not swept; the complete selected 2×2 factorial is prioritized.

Primary-source references: [Census table/publication](https://data.gov.sg/datasets/d_d95ae740c0f8961a0b10435836660ce0/view), [archived household-income table](https://data.gov.sg/datasets/d_2d6793de474551149c438ba349a108fd/view), [NEA](https://data.gov.sg/datasets/d_a57a245b3cf3ec76ad36d55393a16e97/view), [NParks](https://data.gov.sg/datasets/d_77d7ec97be83d44f61b85454f844382f/view), [URA](https://data.gov.sg/datasets/d_8594ae9ff96d0c708bc2af633048edfb/view), [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1), [AlphaEarth catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL), [AlphaEarth release](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/).

Reproduce main study: `.venv/Scripts/python.exe analysis/acquire_local_spatial_sources.py`; `.venv/Scripts/python.exe analysis/compare_local_economy_landscape.py --stage all`. Regional income extension: `.venv/Scripts/python.exe analysis/acquire_local_spatial_sources.py --config analysis/configs/local_income_extension.json`; `.venv/Scripts/python.exe analysis/compare_local_income_extension.py --stage all`. Final verification/report: `.venv/Scripts/python.exe analysis/summarize_local_economy_landscape.py`. Use `--stage dry-run` on the main runner for access/configuration inventory. See inventory.md, feature_dictionary.json, both source manifests and SATELLITE_SETUP.md.
