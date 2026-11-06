[Chinese version](README.zh-CN.md)

# Feature engineering: from basic attributes to anniversary histories

Read the [27-input baseline report](BASELINE_27_FEATURES.md),then use controlled fixed-model ladders to interpret added information.

| Content | File |
| --- | --- |
| Basic cleaning, hypotheses and 27-input policy | [BASELINE_27_FEATURES.md](BASELINE_27_FEATURES.md) |
| Baseline27-field definitions/provenance | [feature_dictionary.csv](feature_dictionary.csv) |
| Single-field deletion/refit | [FEATURE_ABLATION.md](FEATURE_ABLATION.md), [single_feature_effects.csv](single_feature_effects.csv) |
| Whole-group deletion/refit | [group_feature_effects.csv](group_feature_effects.csv) |
| Fixed-model19/27/47/62-input comparison | [controlled_feature_ladder.csv](controlled_feature_ladder.csv) |
| Ladder settings/results interpretation | [Full report section2](../01_final/EXPERIMENT_REPORT.md#2-explain-feature-engineering-with-fixed-learners-and-iterations) |

19 inputs cover properties, time, location and amenities; 27 adds past rents; 47 adds market-relative histories;
62 adds12/24/36-month cohorts. Document settings per table; no direct cross-table attribution.

Deletion/refit effects are conditional predictive contributions, not additive or causal shares.
Histories respect cutoffs without validation-rent updates.
Fold metrics, settings and diagnostics:`output/experiment_runs/features_*/`.
