# Continuous HDB rental-expensiveness surface

[Open the interactive heatmap](rent_heatmap.html).

Reference: 4-room, 95 sqm, model a, 25-year flat/building age, maximum floor 12, approval 2025-03. Alternative: same type/size/model, actual lease/completion year and maximum floor retained.
Existing T0 transformers, declared seeds 42/43/44; average predictions at 3,553 real block anchors. Each has at least 5 same-type labeled fitting transactions. All labels and histories end May 2024. No 2024-06 or later rent labels enter this map. No refitting.

## Meaning of the continuous surface

These are predicted monthly rents for hypothetical flats at supported existing HDB locations. They are not parcel/land prices, private-property rents or a causal location premium. Building identities and local rental histories vary even in the standardized view, so some unmeasured building characteristics can remain.
Every one of the 46 target-derived features and the external baseline is recomputed for the reference property and common cutoff. Exact-property signature histories may be missing; the trained preprocessor handles them. Available block/type histories and local forecast baselines remain. No arbitrary coordinate edits with stale geographic attributes are used.
The grid is a normalized compact Gaussian smoother over real-block model predictions, bandwidth 300 m, radius 1 km, with a smooth taper to zero at the boundary. Display requires an anchor within 500 m and effective support at least two. All anchors inside the kernel radius contribute. Values between blocks are interpolation, not direct transformer predictions; smoothing can hide block-level discontinuities. The 100 m default grid is rendering resolution, not prediction precision.
The existing 2021 ESA WorldCover raster masks nodata and permanent water (class 80), only for display. It is not a current coastline, administrative border or construction-eligibility layer. Grey land includes unsupported locations, forests and industrial/private-property areas; no rent is inferred there. [WorldCover legend](https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf).

## Supported-town averages of the standardized scenario

| town | blocks | reference_mean_sgd | existing_building_mean_sgd | seed_range_mean_sgd |
| --- | --- | --- | --- | --- |
| central | 30 | 3960.804 | 3981.733 | 275.482 |
| queenstown | 78 | 3817.517 | 3916.736 | 198.205 |
| bukit merah | 153 | 3670.817 | 3709.757 | 217.464 |
| clementi | 77 | 3603.721 | 3584.868 | 116.073 |
| kallang/whampoa | 106 | 3591.072 | 3614.106 | 144.743 |
| bishan | 121 | 3472.745 | 3427.493 | 129.377 |
| toa payoh | 83 | 3462.074 | 3468.193 | 130.668 |
| serangoon | 105 | 3395.504 | 3340.884 | 111.765 |
| geylang | 85 | 3392.813 | 3362.376 | 112.628 |
| ang mo kio | 133 | 3362.450 | 3319.828 | 100.695 |
| jurong east | 86 | 3308.763 | 3288.955 | 106.546 |
| tampines | 285 | 3288.256 | 3251.668 | 81.849 |
| bedok | 147 | 3249.240 | 3209.193 | 78.238 |
| pasir ris | 123 | 3206.581 | 3196.847 | 104.964 |
| jurong west | 246 | 3193.899 | 3184.170 | 101.106 |
| bukit batok | 120 | 3168.934 | 3141.965 | 128.224 |
| hougang | 225 | 3168.789 | 3148.446 | 76.120 |
| punggol | 191 | 3140.176 | 3201.341 | 106.023 |
| sengkang | 278 | 3114.738 | 3163.329 | 98.312 |
| choa chu kang | 181 | 3088.392 | 3086.339 | 93.196 |
| yishun | 238 | 3067.111 | 3038.322 | 103.182 |
| sembawang | 94 | 3030.720 | 3049.264 | 105.374 |
| woodlands | 227 | 3019.208 | 3005.910 | 105.588 |
| bukit panjang | 113 | 2962.619 | 2972.271 | 107.273 |

Town summaries average supported block predictions, not population-weighted observed rents. They are conditional model scenarios; do not interpret the ordering as a causal town premium or a price for every spot in that town.

## Limits and verification

Seed range describes optimization disagreement only; it is not a calibrated forecast interval. The selected T0 architecture has the best primary mean validation score among these transformer variants, but its development period has been repeatedly inspected. The underlying forecast baseline plateaus after horizon ten. Static geography/amenity snapshots and land cover retain their original vintage limitations.
Original model weights, inputs and Kaggle submissions remain unchanged. Source and node artifacts, scenario cutoffs and interpolation settings are recorded in verification.json. Reproduce using `.venv/Scripts/python.exe analysis/plot_transformer_rent_surface.py`; use a new --output directory for a different scenario.
