[Chinese version](PROTOCOL.zh-CN.md)

# Model comparison with identical inputs

All models use default27 inputs, 114,842 fitting and 35,158 validation rows,
April2024–March2025. Select settings/duration from original compact inner-time scores only,
then refit. No residual offset/warm-up deletion. Reuse exactly matched prior outer results;
refit others. Each model's `results.json` records selection/provenance.

`model_metrics.csv` is the common table;`results.json` explains protocol/limits.
Preprocessing adapts to models (e.g.,neural numerical transforms/embeddings),
with unequal tuning budgets, one seed, one outer window and prior label inspection.
Paired three-month block percentiles describe month sensitivity, not full model significance.

Unlike `comparison_model_search/model_metrics.csv`,which permits model-specific features/residuals
and compares complete pipelines, this table fixes27 raw inputs to better examine model differences.

```powershell
python -m analysis.main common-models
```

Initial execution requires original `model-search` inner trials; commands/deep dependencies:
[model guide](../../docs/methods/model_comparison.md). Full analysis:[current report](../02_feature_engineering/BASELINE_27_FEATURES.md).
