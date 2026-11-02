# Data-cleaning audit of the MSE-only DiagScale submission

**Conclusion: basic cleaning, row alignment and submission formatting pass, but the spatial source is not fully cleaned for historical availability. The current77-feature cache includes future JRL/CRL sites as open MRT access.**

## Verified input and cache integrity

| split | rows | raw_missing_cells | raw_exact_duplicate_excess | canonical_duplicate_excess | canonical_duplicate_affected_rows | flat_type_alias_rows | invalid_lease_age_rows | missing_block_metadata_rows | unseen_block_id_rows | constants_excluded | minimum_floor_area | maximum_floor_area | minimum_rent | maximum_rent | nonpositive_rent_rows | rent_at_most_500_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 150000 | 0 | 425 | 641 | 1261 | 35331 | 1 | 159 | 0 | FURNISHED,FEE | 31.000 | 215.000 | 300.000 | 7600.000 | 0.000 | 49.000 |
| test | 50000 | 0 | 5508 | 7764 | 13973 | 0 | 0 | 0 | 102 | FURNISHED,FEE | 31.000 | 215.000 | nan | nan | nan | nan |

Raw train/test have no missing cells, invalid approval dates, nonpositive floor areas or infinite model inputs. Known source hashes match the full-data training settings. All150,000 train and50,000 test rows retain their raw order; cached labels match raw rent exactly. Flat-type aliases such as4 room/4-room are canonicalized, and case/whitespace normalized address keys give100% many-to-one block joins. Cached coordinates and building metadata match an independent join. Constant FURNISHED=yes and FEE=0 are excluded from model inputs. No current-rent, log-rent or rent-per-area target column appears among the77 inputs.

## Findings and their treatment

### High priority: inaccurate MRT operating-status source

The source has 34 distinct coordinates markedopen whose codes consist exclusively of future JRL/CRL lines. The pipeline filters STATUS=open and deduplicates exact coordinates, but has no station-opening date filter. Independent nearest-neighbour calculation reproduces the saved NEAREST_MRT_KM values, confirming this source issue is present in the submitted model inputs.

| split | nearest_jrl_crl_only_rows | nearest_jrl_crl_only_pct | jrl_crl_in_1km_count_rows | jrl_crl_in_1km_count_pct |
| --- | --- | --- | --- | --- |
| train | 20557 | 13.705 | 29181 | 19.454 |
| test | 7153 | 14.306 | 9899 | 19.798 |

[LTA’s March2025 announcement](https://www.lta.gov.sg/content/ltagov/en/newsroom/2025/3/news-releases/expanding_our_rail_network_and_strengthening_rail_reliability.html) gives JRL opening stages in2027-2029 and CRL openings from2030, later than every train/test transaction in these files (latest July2026). The counts above are a conservative subset; they do not count other station-opening-date issues. Sites sharing a coordinate with an operating-line source code are excluded from this flag.

This is temporally inaccurate infrastructure availability, not direct use of hidden rental labels. It can make access look better than it was. Correct operating stations and planned-station access should be separate features in a controlled rerun. Existing77-feature checkpoints and the current CSV are preserved; silently changing these inputs at inference would break training consistency.

### Repeated records: retained, not automatically established as errors

Raw training has425 exact duplicate excess rows; canonicalization raises this to641. Test has5,508 raw duplicate excess rows and7,764 after canonicalization. These counts include all observed fields (including rent for training). There is no individual flat-unit or tenancy identifier, so identical rows can represent distinct contracts in the same block/month. Keep every test row and its Kaggle ID. Any train deduplication or reweighting needs a validation sensitivity experiment and a documented identity rule. Repeated observed property inputs alone do not establish target leakage. Duplicate-case exports are provided for review.

### One invalid lease-based age: already masked

Training row31739 has approval2021-11 and lease commencement2022. Its derived FLAT_AGE_YEARS is correctly NaN. Missing block metadata (-1 in the source) becomes NaN for159 training rows and zero test rows. This is not a missing-address problem: the coordinate joins are complete. Completion year and lease commencement are different fields and are retained separately.

### Missing histories and new categories: expected and explicit

The history features intentionally preserve NaN when no observations, no enough observations for a sample standard deviation, or no supported forecast are available at the permitted origin. Recent counts remain nonnegative, zero-count means are missing, and sample deviations are never negative. TabPFN receives its native missing-value representation; there is no future-target backfill.102 test rows have unseen BLOCK_ID categories and are deliberately encoded as missing while their town, coordinates and other features remain available. Other categorical columns are complete and use known train levels. Detailed missing counts are in feature_missingness.csv.

### Redundant histories: data structure, not a join failure

All four HIST_SIGNATURE summaries equal their HIST_BLOCK_FLAT counterparts, including missingness. In the combined raw data, floor area, model and lease year are each constant within a normalized block x flat-type group, so the finer signature creates no additional partition. Removing these four redundant columns is a feature-ablation question requiring retraining rather than a silent cleaning edit.

### Extreme rents and areas: flagged for review, not assumed erroneous

Training rents range fromS$300 toS$7,600; floor areas range31-215sqm. Nonpositive inputs are absent. Low/high prices and unusually large sizes are not dropped simply for falling outside a statistical range; source confirmation is needed to call them corrupt. The current pipeline has no blanket outlier deletion or winsorization.

## Temporal policy and CSV checks

March2025 test histories use February2025; later test histories use March2025. The actual forecast horizon matches each row’s calendar month. 18,077 test rows exceed the validated ten-month forecast range and inherit the documented month10 forecast plateau. That extrapolation is a modelling limitation, not a raw-value cleaning operation. Existing causal-history tests are rerun separately; this audit verifies cached date/origin consistency and arithmetic without claiming that a static audit proves every possible leakage path absent.

The final CSV has50,000 unique, sample-aligned Id values and two columns Id,Predicted. All predictions are finite and positive. Source data, cached features, trained adapter and the saved submission are hash-checked before/after this read-only audit and remain unchanged.

## Next action

Prioritize an opening-date-corrected MRT feature experiment with the identical chronological80/20 split before changing the Kaggle submission. Deduplication policy and four redundant history columns can then be tested independently. Basic categorical/date/sentinel cleaning does not need a wholesale rewrite.

Reproduce: `.venv/Scripts/python.exe analysis/audit_diagscale_submission_cleaning.py`. Audit tables and issue row IDs are stored next to this report.
