# Frozen TabPFN amenity-quality augmentation

The combined dated-source augmentation changes RMSE by S$+0.316, with paired building-cluster 95% interval [-0.293, +0.908]. Retain the 77-feature control; no robust gain is established.

This experiment adds source-backed school demand, mall scale/activity and MRT connectivity/passenger measurements to all 77 existing baseline features. No TabPFN training, adapters, blending or context subsampling is used. The same checkpoint, single estimator, seed 42, residual targets, reconstruction and 200-row query batches are retained.

## Evaluation and experiment definitions

Chronological 80/20: 121,005 context rows (January 2021–May 2024) and the identical 28,995 June 2024–March 2025 development queries. The 50,000 unlabelled competition rows and existing submission are unchanged. Two earlier ten-month folds retain every context row and use the existing fixed label-free 600-query samples per fold, weighted to monthly populations. The four feature blocks and full evaluations were defined before new outer scoring. No outer targets set source values or weights.

| Group | Added dimensions | Total input columns |
| --- | ---: | ---: |
| Baseline | 0 | 77 |
| Schools | 5 | 82 |
| Malls | 7 | 84 |
| Transport | 9 | 86 |
| Combined | 21 | 98 |

These are logical input-column counts; the SDK can remove constant/all-missing columns in a particular fold.

Actual May 2024 per-location inputs are saved in school_model_weights_2024-05.csv, mall_weight_assignments_2024-05.csv and station_model_weights_2024-05.csv. The older school_weight_assignments.csv gives within-year reference ranks, not the mixed-latest-record ranks used by the current model. Per-property features apply the documented distance kernels to these source values.

## Sources and measurements

Schools: 182 existing primary-school locations joined to 729 historical school-year records: 186 schools in 2020 and 181 in each of 2021–2023. 181 of the 182 sites match the 2023 archive. Source: [SG Schooling historical tables](https://sgschooling.com/year/2020/all), tabulating MOE registration facts. This is a secondary archive, not an official ranking; historical versions are not independently preserved here. P1 demand is a proxy for parental demand, not a teaching-quality assessment or a secondary-school league table.

The school block contains nearest-school log((Phase-2C applicants+1)/(vacancies+1)), maximum latest-available demand percentile within 1 km, distance-weighted above-median demand access, source coverage within 1 km, and nearest-source unknown indicator. Same-year figures enter conservatively in October after Phase 2C. Percentiles rank the latest available record per archived school at each cutoff. Schools missing from a newer archive can retain earlier records, so this is not strict within-registration-year normalization; the 2022 reserved-seat change can still affect comparisons for such records. Distance decay is exp(-d/500 m). Primary admissions source columns are parsed separately across old/new phase structures and tested against sample rows.

Malls: published measurements for 22 of the 113 existing mall locations, extracted from CMT/CICT and FCT 2019–2023 owner reports plus Link REIT's dated Jurong Point disclosure. [CICT reports](https://investor.cict.com.sg/ar.html), [FCT reports](https://fct.frasersproperty.com/ar.html), [Link disclosure](https://www.linkreit.com/en/media/news-releases/link-enters-singapore-through-the-acquisition-of-jurong-point-and-swing-by-thomson-plaza/). Source page numbers, units, property scope and extraction method are in mall_records.csv. Representative pages were rendered and visually checked, exposing and excluding grouped Bugis+/Bukit Panjang Plaza revenue.

Mall features: nearest reported retail NLA, distance-weighted reported NLA, owner gross revenue and annual footfall within 2 km, NLA coverage, revenue coverage and nearest record-publication age. Sums use exp(-d/750 m) and log1p; NLA is converted to square metres. Gross property revenue is not tenant retail sales or net property income. Mixed-use office/warehouse area, partial Northpoint City wings, combined-property revenue and Waterway Point/NEX stakeholder-share revenue are excluded. Unknown observations are not assigned arbitrary tiers or zero quality. A weighted sum is missing when malls exist nearby but none has that measurement; actual absence of any mall within the buffer has zero access. The 2019 retail-component areas for Funan/IMM/Raffles City are retained as dated measurements with newer mixed-use revenue omitted. Jurong Point NLA is approximate as disclosed. Mall levels are not used because physical, retail and carpark storey definitions differ.

Transport: an operational-date-filtered MRT graph, distinct operating MRT-line count, normalized harmonic graph centrality, graph hops to Raffles Place, same-name/nearby bus services and station entry/exit volume. [LTA TEL dates](https://www.lta.gov.sg/content/ltagov/en/upcoming_projects/rail_expansion/thomson_east_coast_line.html/1000). The inherited station table contains future stations marked open. Its original 77 features are preserved for control; only the new graph filters those records by verified operating dates. Historical CE1/CE2 branch links are used for source labels CC34/CC33, and the post-2026 Circle Line closure is not invented for 2021–2024. LRT topology is incomplete and excluded from the graph. This block combines an operating-station distance correction with quality/connectivity measures; without a distance-only ablation, any gain cannot be attributed solely to crowd volume or network centrality.

Bus connections use dated public [BusRouter SG / cheeaun archives](https://github.com/cheeaun/sgbusdata) derived from LTA. Same-name matching uses normalized station/stop names, stop/interchange descriptors and a 500 m distance guard; a separate 400 m proximity count covers connections without a matching name. Counts are unions of unique service IDs across stops and both directions, not frequencies or passenger counts. The audit lists matched stop names/codes and distances. Archive schema changes (additional road-name field) are handled explicitly. Underlying data attribution remains LTA/OneMap/OSM; the repository's software license is not treated as a blanket data license.

Ridership is from the public LTA-derived [station passenger archive](https://www.kaggle.com/datasets/yorkyong/singapore-passenger-volume-by-train-stations), containing August–October 2023 hourly entry/exit totals, uploaded 6 December 2023. Monthly totals are divided by calendar days and averaged across the three months. Values are available from January 2024 conservatively; missing-source indicators are retained. Daily tap volume is an activity proxy, not train occupancy, platform density or observed waiting time. Only 2.52% of the main fitting rows have dated ridership coverage, versus full holdout coverage, limiting what this experiment can establish about crowd-volume usefulness. The official LTA API key was not configured; no private key was requested or exposed.

## Information policy and limitations

Dated-source runs use the latest observation available at each context pseudo-origin and the fixed May 2024 query origin. Label-free January 2021 warmup rows without a target-history origin use December 2020 for external data. FCT fiscal reports enter in January of the next year; CICT calendar-year reports enter in May of the next year, conservatively after publication. There is no future-source backfill in dated mode. Underlying geocoding/location snapshots are inherited; a perfectly archived historical geography is not claimed.

The separately labelled combined_canonical_static_weights sensitivity fixes external measurements at May 2024 for all rows while retaining the contemporaneous operational rail graph. This explicitly retrospective representation tests source-coverage asymmetry; it is not evidence of point-in-time performance and must not replace the dated result. Historical-school archive revisions and incomplete mall reporting remain limitations in both modes.

Before full transport/combined scoring, a source audit exposed abbreviated stop names such as S'goon, Botanic Gdns and 6th Ave. A separate reproducible correction expands these published abbreviations and changes only the same-name bus-service column, with all other 20 added columns asserted identical. Nineteen of 134 station counts change in May 2024, including Serangoon (37 services). The raw source files, original prepared artifact and initial earlier-fold diagnostic outputs are preserved. Primary transport and combined results below use canonical names; station_bus_match_audit_canonical_2024-05.csv and bus_stop_abbreviation_audit.csv contain the evidence. Remaining zero same-name matches are retained alongside the independent nearby-stop count.

## Earlier diagnostic results

| model | query_rows | rmse_sgd | mae_sgd |
| --- | --- | --- | --- |
| baseline | 1200 | 615.701 | 480.872 |
| schools | 1200 | 622.593 | 485.896 |
| malls | 1200 | 608.325 | 481.047 |
| transport | 1200 | 619.573 | 484.837 |
| combined | 1200 | 613.359 | 481.376 |

## Complete 80/20 results

| model | n | rmse_sgd | mae_sgd | bias_sgd | delta_rmse_sgd | relative_rmse_improvement_pct | r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 28995 | 455.814 | 329.748 | 0.768 | 0.000 | 0.000 | 0.516 |
| schools_dated | 28995 | 457.220 | 329.373 | 12.329 | 1.406 | -0.309 | 0.513 |
| malls_dated | 28995 | 456.596 | 331.469 | -5.922 | 0.782 | -0.172 | 0.514 |
| transport_canonical_dated | 28995 | 457.453 | 335.359 | -30.721 | 1.639 | -0.360 | 0.512 |
| combined_canonical_dated | 28995 | 456.130 | 331.445 | -10.885 | 0.316 | -0.069 | 0.515 |
| combined_canonical_static_weights | 28995 | 457.455 | 331.241 | 1.542 | 1.641 | -0.360 | 0.512 |

Intervals in cluster_bootstrap.json use 5,000 paired resamples separately by building and month. This holdout has been inspected in previous project work; intervals are descriptive and not adjusted for repeated development/model selection. Grouped scores by town and flat type, all monthly results and per-transaction errors are saved alongside the report.

## Coverage and computation

| split | feature | missing_fraction |
| --- | --- | --- |
| train | AQ_SCHOOL_NEAREST_LOG_DEMAND | 0.000 |
| train | AQ_SCHOOL_MAX_PERCENTILE_1KM | 0.036 |
| train | AQ_SCHOOL_EXCESS_ACCESS_LOG | 0.000 |
| train | AQ_SCHOOL_COVERAGE_1KM | 0.036 |
| train | AQ_SCHOOL_NEAREST_UNKNOWN | 0.000 |
| train | AQ_MALL_NEAREST_LOG_NLA | 0.770 |
| train | AQ_MALL_NLA_ACCESS_LOG | 0.471 |
| train | AQ_MALL_GROSS_REVENUE_ACCESS_LOG | 0.604 |
| train | AQ_MALL_FOOTFALL_ACCESS_LOG | 0.481 |
| train | AQ_MALL_COVERAGE_2KM | 0.011 |
| train | AQ_MALL_REVENUE_COVERAGE_2KM | 0.011 |
| train | AQ_MALL_NEAREST_RECORD_AGE_YEARS | 0.770 |
| train | AQ_MRT_NEAREST_OPERATIONAL_KM | 0.000 |
| train | AQ_MRT_NEAREST_LINE_COUNT | 0.000 |
| train | AQ_MRT_NEAREST_HARMONIC_CENTRALITY | 0.000 |
| train | AQ_MRT_NEAREST_CBD_HOPS | 0.000 |
| train | AQ_MRT_NAMED_BUS_SERVICES_LOG | 0.251 |
| train | AQ_MRT_NEARBY_BUS_SERVICES_LOG | 0.251 |
| train | AQ_MRT_NEAREST_DAILY_TAPS_LOG | 0.975 |
| train | AQ_MRT_BUS_SOURCE_UNKNOWN | 0.000 |
| train | AQ_MRT_PASSENGER_SOURCE_UNKNOWN | 0.000 |
| query | AQ_SCHOOL_NEAREST_LOG_DEMAND | 0.000 |
| query | AQ_SCHOOL_MAX_PERCENTILE_1KM | 0.039 |
| query | AQ_SCHOOL_EXCESS_ACCESS_LOG | 0.000 |
| query | AQ_SCHOOL_COVERAGE_1KM | 0.039 |
| query | AQ_SCHOOL_NEAREST_UNKNOWN | 0.000 |
| query | AQ_MALL_NEAREST_LOG_NLA | 0.707 |
| query | AQ_MALL_NLA_ACCESS_LOG | 0.378 |
| query | AQ_MALL_GROSS_REVENUE_ACCESS_LOG | 0.549 |
| query | AQ_MALL_FOOTFALL_ACCESS_LOG | 0.414 |
| query | AQ_MALL_COVERAGE_2KM | 0.011 |
| query | AQ_MALL_REVENUE_COVERAGE_2KM | 0.011 |
| query | AQ_MALL_NEAREST_RECORD_AGE_YEARS | 0.707 |
| query | AQ_MRT_NEAREST_OPERATIONAL_KM | 0.000 |
| query | AQ_MRT_NEAREST_LINE_COUNT | 0.000 |
| query | AQ_MRT_NEAREST_HARMONIC_CENTRALITY | 0.000 |
| query | AQ_MRT_NEAREST_CBD_HOPS | 0.000 |
| query | AQ_MRT_NAMED_BUS_SERVICES_LOG | 0.000 |
| query | AQ_MRT_NEARBY_BUS_SERVICES_LOG | 0.000 |
| query | AQ_MRT_NEAREST_DAILY_TAPS_LOG | 0.000 |
| query | AQ_MRT_BUS_SOURCE_UNKNOWN | 0.000 |
| query | AQ_MRT_PASSENGER_SOURCE_UNKNOWN | 0.000 |

| model | added_dimensions | fit_seconds | predict_seconds | peak_vram_gib | context_rows |
| --- | --- | --- | --- | --- | --- |
| combined_canonical_dated | 21 | 15.781 | 503.974 | 7.120 | 121005 |
| combined_canonical_static_weights | 21 | 15.258 | 508.124 | 7.132 | 121005 |
| malls_dated | 7 | 14.584 | 585.396 | 7.153 | 121005 |
| schools_dated | 5 | 14.666 | 406.295 | 7.180 | 121005 |
| transport_canonical_dated | 9 | 15.024 | 489.276 | 7.136 | 121005 |

GPU figures are peak PyTorch-allocated memory, not total driver reservation. All runs retain the same full context; none changes the original labels or residual baseline. Source hashes, parsed records, coordinate joins, source-quality limits and match audits are stored separately from model outputs.

Validation: 25 tests pass across the new source/feature cases and existing spatial/temporal safeguards (test_results.txt). Baseline replay agrees within 1e-5 SGD; all 77 control columns, targets, residual baselines and holdout row IDs are checked, and checkpoint/source/code/submission hashes are verified.

Reproduce: install requirements-amenity-quality.txt; run analysis/acquire_amenity_quality.py --stage all and --stage followups; then analysis/compare_amenity_quality.py --stage prepare, --stage smoke and --stage inner; run analysis/refine_amenity_bus_names.py --stage all for corrected earlier diagnostics and full outer evaluations; finally run analysis/summarize_amenity_quality.py. Each stage is a separate invocation. Source URLs are in source_manifest.json and bibliography.json. Existing baseline/submission outputs are preserved.
