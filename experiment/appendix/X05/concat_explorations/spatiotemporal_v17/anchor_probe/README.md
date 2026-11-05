# Frozen screen anchor probes

All scores use the same 30,000 fit and 5,000 development-validation rows as v17 no-MRT F0; F0's saved best epoch scores 481.36 SGD. This is a screening comparison, not full-split confirmation. Models fit or calibrate on training labels only, with fixed horizon ridge penalty 1,000 and group pseudo-count 20. Validation labels were used only for reporting. The known-anchor-only original-anchor sensitivity was added after the initial calibration results, and the known-anchor refreshed-12 sensitivity after observing the all-train refreshed results; both are post-hoc, although their coefficients still use training labels only.

| Probe | Train RMSE | Validation RMSE | Validation bias (pred−actual) |
|---|---:|---:|---:|
| Original causal anchor | 806.10 | 719.00 | −528.55 |
| Original + all-train mean residual | 642.84 | 489.25 | −42.19 |
| Original + train-fitted horizon ridge | 631.31 | 494.08 | −47.97 |
| Original + horizon + shrunk town×flat-type residual | 625.27 | 489.00 | −49.74 |
| Original + mean from known-anchor training rows only | 644.37 | 495.06 | −86.60 |
| Refreshed 6-month anchor | 708.56 | 501.85 | −45.61 |
| Refreshed 6-month + all-train mean residual | 626.91 | 575.13 | +284.61 |
| Refreshed 12-month anchor | 744.05 | 494.99 | −54.91 |
| Refreshed 12-month + all-train mean residual | 631.78 | 596.92 | +338.10 |
| Refreshed 12-month + known-anchor-only train mean residual | 633.53 | 571.58 | +291.04 |

Refreshed anchors compute count-weighted monthly town×flat-type and block×flat-type rent from frozen trailing sequences, with oldest-to-newest months, rent in kSGD and log1p(count). Block mean shrinks to town using fixed prior 10; town missing falls back to original anchor. Counts are reconstructed with `expm1`. The original anchor is cumulative through the origin; refreshed levels use only the 6 or 12 months ending at origin. The underlying sequences can use all legally available past source labels, including unsampled fit rows. No validation labels enter history or calibration.

The original mean estimate uses all fit rows, including 827 cold starts with zero anchor; the known-anchor estimate is a separate sensitivity, not a silent exclusion. The substantial positive bias from applying a single fit-period residual correction to the refreshed anchor shows poor transfer across this chronological split; it does not identify its cause. Raw refreshed anchors reduce error relative to the uncalibrated cumulative anchor but do not outperform the neural screen baseline. Avoid claiming an anchor improvement over the selected model without full-split and chronological robustness checks.

`results.json` records counts, coefficients, fit row/target/anchor hashes and train/validation metrics; per-arm predictions and `group_metrics.csv` include horizon, month, town, flat type, signed biases and paired MSE deltas versus F0. Source row ID, date-derived month, targets, town, flat type, horizon and reconstruction were checked; floor area was read for alignment/context but not fitted. Four focused tests cover validation-label independence, unseen-group fallback, causal sequence perturbation and missing-history fallback.
