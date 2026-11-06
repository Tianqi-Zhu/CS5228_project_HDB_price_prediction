[Chinese version](model_comparison.zh-CN.md)

# Early model search: protocol, configuration and reproduction

This page retains the complete-pipeline search in which models choose their own features/residual representations.
The newer [common-27-input comparison](../../experiment/main/M05/common_27/PROTOCOL.md) fixes inputs and fitting rows
and is better suited to architecture comparisons. See the [project report](../../experiment/main/M03/README.md) for the current narrative; do not merge these tables into one architecture ranking.

This round predicts individual HDB monthly rents; all final RMSEs are calculated after reversing transforms into SGD.
Regional means, standardized-target errors and log errors are different metrics.

Stage results are in `output/experiment_runs/comparison_model_search/model_metrics.csv`;
`summary.json` records pooled results and goal status, and `model_comparison.png` is the comparison chart.
Each model directory retains selected parameters, feature version, inner-validation score, fixed refit epochs and outer score.

This stage completed 30 inner candidate configurations and outer fitting for ten model families. Use the corresponding results table for scores; this page records protocol and reproduction settings.

## Chronological validation protocol

The primary outer validation is **all 35,158 records from 2024-04 through 2025-03**, with fitting data through 2024-03.
The 17,977 rows from 2023-10 through 2024-03 are used only for inner selection of features, hyperparameters and training length;
after selection, refit all data before the outer origin for the selected fixed duration.
Outer validation neither stops training nor selects blend weights.

The annual windows starting 2023-04 and 2022-04 and the sixteen-month window starting 2023-12 check selected tree models, with inner selection
in the six months before each origin. Retain annual scores separately; an easier window is not representative overall performance.

These outer months have been examined previously; this is still policy-selection evidence, not a new independent test.
Nesting reduces direct tuning bias in this round but cannot erase previous repeated selection.

## Models and feature adaptation

| Model | Main configurations compared |
| --- | --- |
| LightGBM | 31/63 leaves; compact history/historical premiums/market residuals; inner early stopping |
| CatBoost | Depth 6/8; native categories, address categories and market residuals; CPU training with inner early stopping |
| XGBoost | Depth 4/6; native categories; compact/premium/market-residual inputs; CPU hist with inner early stopping |
| HistGradientBoosting | 400 iterations; compact or market-residual inputs; only low-cardinality categoricals |
| ExtraTrees | 300 trees; minimum leaf size 5/10; one-hot categories and fit-only numerical imputation |
| Embedding MLP | Categorical embeddings + three-layer 256-wide MLP; compare compact, entity and market-residual inputs |
| FT-Transformer | Official implementation, three layers of width 96; categorical/numerical tokens and the same feature variants |
| TabNet | Official PyTorch implementation; decision/attention widths 24/32, four steps, category embeddings and entmax |
| Embedding ResNet | Categorical embeddings + three 256-wide residual blocks; try rare-level merging for address categories |
| FT-Transformer 192 | Official width-192 configuration; learning rate 1e-4, weight decay 1e-5 and smaller batches |

FT-Transformer uses the [authors' implementation](https://github.com/yandex-research/rtdl-revisiting-models),
and TabNet uses [DreamQuark's implementation](https://github.com/dreamquark-ai/tabnet).
ResNet is implemented in this project, which also manages neural training and preprocessing consistently.

See `analysis/experiments/model_comparison/model_search.py` for all candidates/settings, and see neural interfaces/preprocessing in
`analysis/pipeline/neural_models.py`.

### Feature variants

- `compact`: the previously selected 27 fields.
- `premium`: research variant combining history means and historical same-type monthly premiums.
- `entities`: the 27 fields plus street, address key and Subzone categories for embedding/native-category models.
- `entities_residual`: entity inputs plus earlier same-type market context, predicting residuals from the latest market baseline.
- `compact_residual`: low-cardinality compact market-residual variant compatible with sklearn category-count limits.

Neural vocabularies, numerical medians, standardization and quantile transforms fit only the corresponding fitting partition.
Apply log1p to history counts and add indicators for numerical columns missing during fitting. Preserve a linearly standardized time index,
avoiding quantile transforms that clip future time to the fitting-period maximum.
Unknown categories encode as 0; some new ResNet/large-FT candidates also map categories with fewer than ten fitting occurrences to 0.

Market-residual variants drop the first six fitting months for historical warm-up; those known labels can still support later
histories. Each model records its actual fitting-row count separately; every validation row is retained.
These tables compare complete pipelines selected by inner validation, so differences cannot all be attributed to architecture.

## Blending and error diagnostics

`blend` takes the three completed models with best inner scores and fits nonnegative, sum-to-one weights to their inner predictions.
Apply those weights directly to outer predictions; do not repeatedly optimize against outer scores.

`report` summarizes stage scores, generates charts and reports the best observed model's errors by area, type and rent band,
plus the largest-error records. Row predictions and binary models stay local and outside Git.

`data_diagnostics.json` separately examines distinguishability:

- Identical raw inputs in the latest outer set can have different rents. Even fitting each group's optimal mean to those validation labels
  leaves in-sample RMSE 150.86. This is diagnostic only, never a validation score.
- Pooled within-group standard deviation for duplicate inputs is about 459.96. This suggests substantial unobserved variation, but is descriptive
  evidence for repeated samples, not a theoretical population error floor of 460 or proof that a target RMSE is impossible.

These records may represent different contracts in the same block; unit floor, interior condition and similar distinctions are unobserved.
Investigate missing information, temporal change and underfitting together; increasing network size alone cannot establish that a target error is achievable.

## Environment and rerunning

`environment.yml` declares core packages, including RTDL, TabNet and XGBoost.
Existing `.cache/model-deps/` remains compatible; install new environments from the main YAML.

```powershell
# Create the core environment; existing environments need not be recreated
conda env create -f environment.yml
conda activate cs5228

# Prepare strictly separate inner/outer feature caches
python -m analysis.main model-search prepare --fold 2024-04

# Train all models or select a subset with --families; completed tasks are reused
python -m analysis.main model-search run --fold 2024-04
python -m analysis.main model-search run --families embedding_mlp ft_transformer tabnet

# Inner-selected blending and final report
python -m analysis.main model-search blend --fold 2024-04
python -m analysis.main model-search diagnose
python -m analysis.main model-search report
```

Use a new `--output-dir` and corresponding caches when changing search settings or training budgets, so completed tasks
are not mistaken for results of a new configuration. Results save after each candidate/model, allowing interrupted runs to resume.
