# Initial implementation inventory

Predictand: monthly rent in SGD, with an additive temporal residual target; no logarithm.

X0: frozen TabPFN-3.5 Fast, TabPFN 9.2.0, checkpoint tabpfn-v3.5-fast-20260909.safetensors; 77 features, one estimator, seed 42, 200-row query batches, full cached context with no subsampling. No DiagScale/LoRA/adapters.

Chronological 80/20: 121,005 January 2021–May 2024 context rows and 28,995 June 2024–March 2025 development rows. Unlabelled 50,000-row test remains untouched. The supplied test extends to July 2026; this comparison does not validate that longer horizon.

Frozen categorical baseline: RMSE S$455.81357012; existing adapted models are outside this frozen-model experiment. X0 has property, spatial, amenity, 37 historical and 9 temporal features. It does not contain national economic features; earlier economic experiments are separate.

Preserved baseline inputs:

FLAT_TYPE_CLEAN, FLAT_MODEL_CLEAN, FLOOR_AREA_SQM, APPROVAL_MONTH_INDEX, APPROVAL_MONTH_OF_YEAR, FLAT_AGE_YEARS, TOWN_CLEAN, LATITUDE, LONGITUDE, MAX_FLOOR, YEAR_COMPLETED, BUILDING_AGE_FROM_COMPLETION, SUBZONE, PLANNING_AREA, REGION, NEAREST_MRT_KM, NEAREST_MALL_KM, NEAREST_SCHOOL_KM, DISTANCE_TO_CBD_KM, AMENITY_DENSITY_1KM, MRT_COUNT_0.5KM, MRT_COUNT_1.0KM, MRT_COUNT_2.0KM, MALL_COUNT_0.5KM, MALL_COUNT_1.0KM, MALL_COUNT_2.0KM, SCHOOL_COUNT_0.5KM, SCHOOL_COUNT_1.0KM, SCHOOL_COUNT_2.0KM, HIST_SIGNATURE_MEAN, HIST_SIGNATURE_COUNT, HIST_SIGNATURE_LAST_MEAN, HIST_SIGNATURE_RECENCY_MONTHS, HIST_BLOCK_FLAT_MEAN, HIST_BLOCK_FLAT_COUNT, HIST_BLOCK_FLAT_LAST_MEAN, HIST_BLOCK_FLAT_RECENCY_MONTHS, HIST_AREA_FLAT_MEAN, HIST_AREA_FLAT_COUNT, HIST_AREA_FLAT_LAST_MEAN, HIST_AREA_FLAT_RECENCY_MONTHS, BLOCK_ID, STREET_ID, RECENT_BLOCK_FLAT_3M_MEAN, RECENT_BLOCK_FLAT_3M_COUNT, RECENT_BLOCK_FLAT_6M_MEAN, RECENT_BLOCK_FLAT_6M_COUNT, RECENT_BLOCK_FLAT_12M_MEAN, RECENT_BLOCK_FLAT_12M_COUNT, RECENT_AREA_FLAT_3M_MEAN, RECENT_AREA_FLAT_3M_COUNT, RECENT_AREA_FLAT_6M_MEAN, RECENT_AREA_FLAT_6M_COUNT, RECENT_AREA_FLAT_12M_MEAN, RECENT_AREA_FLAT_12M_COUNT, RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN, RECENT_BLOCK_FLAT_6M_SHRUNK_MEAN, RECENT_BLOCK_FLAT_12M_SHRUNK_MEAN, RECENT_BLOCK_FLAT_3M_STD, RECENT_BLOCK_FLAT_6M_STD, RECENT_BLOCK_FLAT_12M_STD, RECENT_AREA_FLAT_3M_STD, RECENT_AREA_FLAT_6M_STD, RECENT_AREA_FLAT_12M_STD, AREA_RECENT_MINUS_LONG, BLOCK_RECENT_MINUS_LONG, BLOCK_MINUS_AREA_RECENT, AREA_LAST_MINUS_RECENT, TEMP_FLAT_RECENT3_MEAN, TEMP_FLAT_RECENT12_MEAN, TEMP_FLAT_DAMPED_FORECAST, TEMP_FORECAST_HORIZON, TEMP_FLAT_HISTORY_COUNT_3M, TEMP_FLAT_LAST_OBSERVATION_AGE, TEMP_LOCAL_FORECAST, TEMP_LOCAL_WEIGHT, TEMP_FORECAST_RMSE

Available additions: published 2020 Census planning-area totals/elderly share; food-service-centre and mall radius/KDE intensities; publication-lagged flat-type-specific rental fields; current NParks managed-park polygon statistics.

Unavailable: building/subzone income, complete retail/restaurant establishments, historical establishment openings/closures, archived park boundaries, authenticated Dynamic World/AlphaEarth extracts. Mall/food-centre counts are not total commercial establishments or employment measures. Park area is not tree/grass cover.

Census publication: 18 June 2021; new features conservatively enter July 2021 forecast origins onward. Newly added rental fields use origin minus one month. Existing X0 histories are preserved at their original information policy; incremental availability does not certify the inherited amenity snapshots as historically available.

Original source metadata, retrieval timestamps, hashes, resolution and policies are in source_manifest.json. Distances/areas use EPSG:3414. Canonical location table uses exact normalized address and supplied coordinates; no postal-only merging.

Predeclared screening: two earlier ten-month folds, full context, nested label-free 60 query rows/month with population weights. Choose representations solely on earlier pooled RMSE; prefer fewer added columns within 0.1%. The baseline competes for the final retained recommendation; forced nonempty economy/landscape groups permit a meaningful factorial even if they lose screening.

Strict-incremental and retrospective results must be reported separately. AlphaEarth's July 2025 public release means even 2017–2023 image-year embeddings could not have been available during this 2024–25 holdout.
