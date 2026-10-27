# TabPFN features and frozen residual heads

This plan is fixed before scoring the new models. Existing outer windows are
2023-01–2023-06, 2023-07–2024-03 and 2024-04–2025-03. They are development
windows, not an untouched final test. Raw rent in SGD and RMSE remain the target
and decision metric. The existing full-history CatBoost predictions are a
reference control.

## Feature and backbone controls

All new backbones use the same frozen TabPFN-3.5 checkpoint, one estimator,
10,000 context rows, seed 42 and 500-row prediction batches. The smaller matched
configuration makes historical embedding generation affordable; comparisons
against the earlier four-estimator/50,000-row run are reported separately.

1. Original 41 property/spatial/history columns.
2. Add normalized composite block ID and street ID, explicitly categorical.
3. Add earlier 3/6/12-month block × flat-type and area × flat-type statistics,
   shrunken block means, rent dispersion and recent-versus-longer-term contrasts.
4. Replace the absolute approval-month index and ordinal month with sine/cosine
   seasonality. This is a separate direct-prediction ablation.

Heads use variant 3 regardless of outer scores. COE and stock observations are
excluded because future realized observations are unavailable at a fixed origin.

## Chronological representation learning

Generate quarterly forecast origins from July 2021 to January 2024. At each
origin, fit the frozen backbone using only earlier labels and generate query
rows at horizons 1, 3, 6 and 12 months. Each query month's random sample is capped
at 600 rows, selected without looking at rents. Query dates end before April
2024. Histories and recent windows freeze at the origin; query labels never
enter the labelled context. The context and query row IDs must be disjoint.

Episodes can be cached globally, but a head for an outer fold may see only query
labels strictly before that fold's origin. Repeated appearances of a source row
under different earlier forecast origins are valid forecast tasks; every copy
is assigned to the same side of the inner split by query month.

For each outer fold, the last three fitting months are the inner chronological
validation period. Ridge alpha is selected from 10, 100 and 1000 on this period.
MLPs use hidden widths 64 and 32, GELU, dropout 0.1, AdamW (learning rate 0.001,
weight decay 0.001), up to 40 epochs and patience 6. Heads learn corrections to
the frozen TabPFN mean. Targets are scaled by 1000; input imputation/scaling is
fit only on the respective head-training data. After selecting alpha/epoch,
refit on all eligible pre-outer episodes.

Compare Ridge and MLP both with selected numeric features alone and with those
features plus the 1024-dimensional query embedding. This distinguishes a useful
representation from a generic calibration head. Store embeddings in the ignored
project cache, with source/code/settings hashes and origin/query audits.

## Decision and optional gradient pilot

Report all folds, monthly errors, pooled RMSE/MAE/bias, head inner-validation
scores, row predictions, runtime and GPU allocated memory. A candidate replaces
CatBoost only with at least a 1% pooled RMSE gain and improvement on every fold.

A backbone gradient pilot is triggered only if an embedding head improves the
matched unmodified recent-feature TabPFN on pooled RMSE and on at least two of
the three outer folds. It begins with small chronological context/query episodes,
one estimator and activation checkpointing. It is a separate resource/accuracy
pilot, not a full fine-tuning sweep. If the gate fails, record that outcome and
retain the feature/head results without claiming fine-tuning was evaluated.
