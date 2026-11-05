# Satellite feature setup and availability

Update, 30 October 2026: a credential-free ESA WorldCover route is now implemented
in the separate [tree-cover experiment](../tabpfn_worldcover_tree_cover_80_20/report.md).
The 2020/2021 Singapore rasters are cached and 100/300 m tree-class fractions
have been extracted for all 9,464 canonical buildings. This route does not
require Earth Engine and does not change the original park experiment.

Earth Engine is not installed/authenticated in this project. No Earth Engine
cloud exports or paid services were submitted. The original experiment used
government park polygons; those are not a substitute for measured tree/grass
cover. The setup below describes the optional Dynamic World/AlphaEarth route.

1. Register an Earth Engine research account and an enabled Google Cloud
   project at https://developers.google.com/earth-engine/guides/access.
2. Upload `canonical_buildings.csv` as a table using `LONGITUDE`, `LATITUDE`
   for geometry. Only the unique building locations need to be uploaded.
3. Inspect `export_dynamic_world_300m.js` / `export_alphaearth_100m.js` in the
   Earth Engine Code Editor; replace the placeholder asset. The scripts print
   the actual available observation years. Creating scripts does not start
   exports. Review account quota/cost policy before manually starting tasks.
4. Generate other scales with `analysis/export_satellite_features.py` using
   `--radius 0|100|300|500`; `--source dynamic_world|alphaearth`; `--years ...`;
   `--building-asset ...`; `--output ...`. Dynamic World is already masked for
   cloud/shadow. Exported mean class probabilities, valid-pixel fraction and
   mean valid observation counts retain missing coverage.
5. Add `available_at` and `availability_source_url` from verified source release
   records, not by guessing from observation year. Preserve `observation_end`,
   `observation_year`, `building_id`, `radius_m`, `valid_pixel_fraction`,
   `valid_observation_count`, and all 9 Dynamic World or 64 `A00`–`A63` bands.
6. Import using `validate_export` / `join_satellite` in
   `analysis/experiments/satellite_feature_import.py`. This enforces unique
   building/year/scale keys, completed prior years, actual publication cutoff,
   valid coverage and missing indicators. Strict mode is default. Retrospective
   mode must be explicitly named and reported separately.
7. Fit `TrainingPCA(8|16|32)` only on the training-location feature matrix and
   transform queries with the same instance. All-unavailable bands are rejected;
   all-missing query embeddings remain missing. Evaluate raw 64D and single
   scales first; do not concatenate every scale. The shared TabPFN feature
   interface is `matrix_pair`: append the imported columns to its `added`
   matrices and record a new prepared-cache manifest/experiment directory.

The live [AlphaEarth catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
checked on 30 October 2026 lists observation years 2017–2024. Catalog image years
are not historical publication dates. [Google released the dataset on 30 July
2025](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/),
after every outer validation transaction. A strictly available AlphaEarth
experiment therefore has no added information for this holdout even with
credentials. An observation-lagged retrospective study is possible and must be
labelled accordingly. Do not claim point-in-time improvement from it.

[Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1)
contains observations from June 2015 onward, but was publicly launched on 9 June
2022. Enforce the later of observation completion and documented publication;
historical imagery alone does not imply that the product was available then.
