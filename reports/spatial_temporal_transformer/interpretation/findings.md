# Spatial, temporal and feature-effect diagnostics

The inputs have sensible geography and strict historical cutoffs, but these diagnostics do not establish that the model adequately learns the spatial–temporal relationship. The additional sequence branch did not improve consistently over the feature transformer.

[Open the interactive dashboard](dashboard.html). All figures and CSVs are standalone local artifacts; no external map or online service is required.

## Completed model comparison

| variant | mean | std |
| --- | --- | --- |
| T0 | 461.164 | 1.544 |
| T1 | 465.309 | 3.163 |
| T2 | 466.777 | 3.881 |
| T3 | 465.086 | 2.534 |

Means/std are across the three declared seeds, in individual-rental SGD RMSE. All use the same repeatedly examined June 2024–March 2025 development rows. T0 already has spatial, calendar and engineered rent-history inputs.

## What is encoded and what is missing

The property encoder attends across feature tokens, including a coordinate/Fourier token, location IDs, amenities, calendar and history summaries. The sequence encoder attends across historical monthly rent/COE/stock tokens and a known forecast-context token. Its local rent panel is selected using planning area and flat type.
The property hidden state does not condition the sequence encoder; outputs combine through a single scalar gate. Direct interactions between the two encoded input blocks are absent by construction. This does not mean all spatial–temporal interactions are absent: the property branch mixes location with time/history, and the sequence already contains local rent histories. Comparable-transaction attention and distance/time attention biases were not tested.

## Spatial and temporal errors

Block residual Moran's I exceeds the support-stratified permutation reference upper band in T0 3/10 months and T2 2/10 months. This is a descriptive warning about remaining spatial structure, not a causal test or a multiplicity-adjusted significance claim.
The graph uses at most eight other blocks within 2 km, Gaussian distance weights with 1 km scale, row normalization, and monthly blocks with at least two rentals. The 199 permutations shuffle residual means within count strata (2, 3, 4, 5+) to partly preserve support. They do not account for fitted-model or search uncertainty.
Town/month heatmaps and monthly bias plots show where prediction errors persist. Raw observed-rent maps remain confounded by property composition; they should not be read as pure location premiums.

## Which input bundles the prediction pipeline relies on

| variant | group | mean | min | max |
| --- | --- | --- | --- | --- |
| T0 | Geography + amenities | 21.247 | 17.851 | 26.610 |
| T0 | Geography + rental history | 135.379 | 122.940 | 144.704 |
| T0 | Property attributes | 6.621 | 4.270 | 9.972 |
| T0 | Rental-history bundle | 83.616 | 71.140 | 92.459 |
| T2 | Geography + amenities | 20.158 | 16.178 | 26.141 |
| T2 | Geography + rental history | 144.206 | 130.442 | 156.262 |
| T2 | Property attributes | 6.973 | 4.679 | 11.460 |
| T2 | Rental-history bundle | 83.643 | 72.808 | 91.183 |

Grouped permutation uses a label-independent 200-query-per-month sample, population weights, and donors within month and flat type; three shuffles for each of three seeds. Flat type itself is held fixed, so its importance cannot be ranked by these shuffles. Geographic columns move together, and the Fourier token is recomputed. History shuffles move the external baseline and sequence together. A joint geography/history shuffle reduces, but cannot eliminate, incompatible feature combinations. The four groups overlap and their importances must not be added. Correlated predictors can substitute for one another, so a small marginal importance does not establish irrelevance.

The large history-bundle effect includes the external forecast baseline, and cannot be attributed solely to the new transformer or sequence encoder. Property shuffles retain the same flat type and rent-history proxies, so their smaller effect is not evidence that property characteristics are unimportant. These group tests do not isolate the benefit of Fourier coordinates versus building IDs or other geographic encodings.

## Direct floor-area response

ALE is computed within 3/4/5-room types, starting with six sampled-feature quantile bins and merging adjacent sparse bins until each has at least 20 rows. It averages adjacent-endpoint changes and accumulates/centers them. The historical summaries and external baseline stay fixed, so this measures direct neural sensitivity, not the complete pipeline effect of changing a property. Shading is seed range, not a confidence interval. [ALE methodology](https://arxiv.org/abs/1612.08468).

The additional observed-feature plot compares actual and predicted means against floor area, flat age, nearest MRT distance and CBD distance among 4-room rentals. Those are descriptive associations: other attributes, location and time still vary. They are deliberately distinguished from the model's direct area-response curve.

## Sequence contributions and lags

| variant | sequence_gate | mean_sequence_sgd | mean_abs_sequence_sgd |
| --- | --- | --- | --- |
| T0 | 0.000 | 0.000 | 0.000 |
| T1 | 0.099 | 51.972 | 51.972 |
| T2 | 0.100 | 22.625 | 27.652 |
| T3 | 0.100 | 3.632 | 23.125 |

The saved branch decomposition exactly reconstructs each sampled prediction. The temporal term includes BF16 addition rounding. A small gate is not an importance score because learned output scale can compensate. Compare actual SGD branch contributions and occlusion responses instead.
Channel/lag masking removes values and their presence flags in the frozen model. These are sensitivity probes that can be outside the training distribution; they do not replace matched refitting comparisons T2–T1 or T3–T2. On the weighted sample, disabling the sequence branch changes RMSE on average: T1 -3.51 SGD, T2 -2.98 SGD, T3 -0.24 SGD; negative means improvement. This is a post-hoc diagnostic, not a newly validated deployment choice. Removing COE from a model trained with COE can hurt even though adding COE through matched retraining did not reliably help: reliance and incremental predictive value are different questions. Attention weights are not presented as causal feature contributions. [Attention interpretation caution](https://aclanthology.org/N19-1357/).

## Economic effects are not identified by this holdout

At the fixed May 2024 rental origin, every main-holdout row shares the same national COE and stock sequence (through April 2024). Row permutation cannot identify their effects: there is no across-row macro variation. Learned responses come from the historical training origins, not 28,995 independent economic paths. A separate COE response curve from this single origin would be poorly supported. Masking and matched refit comparisons are reported instead; no causal COE/stock effect or company-local effect is claimed.

Target month and forecast horizon also move together in this fixed-origin evaluation. Their separate effects cannot be inferred from monthly error curves. More independent origins/new rental months and an explicitly tested joint spatial–temporal architecture are needed before claiming the relationship is adequately captured.

## Verification and scope

Full saved GPU prediction replay: maximum absolute discrepancy 5.09e-11 SGD. All input, original checkpoint/preprocessor, source-code and submission hashes are checked. No model is retrained and no original experiment artifact is overwritten.
Plots use primary seed 42 for geographic errors and all three declared seeds for reliance/response stability. No best-seed selection. Reproduce with `.venv/Scripts/python.exe analysis/interpret_spatial_temporal_transformer.py`; CUDA inference uses the existing capacity gate. See verification.json for exact inputs, settings, devices and replay tolerances.
