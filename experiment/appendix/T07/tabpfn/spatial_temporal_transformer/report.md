# Spatial and temporal transformers from scratch

All twelve configured variants completed: 24 calibration fits, 20 main evaluation fits, and three long-horizon fits. Saved checkpoints, predictions and original source/submission hashes were verified.

All models start from random weights. Original labels and source/submission files are preserved.
Training episodes cover horizons 1–16. All rent-derived inputs share the row cutoff; macro inputs have an extra month buffer.
The inherited baseline plateaus beyond month ten; the neural horizon input does not.

## Individual-rental development results

| variant | seed | rmse_sgd | mae_sgd | bias_sgd | best_epoch | peak_allocated_vram_gib |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 42 | 471.7953 | 332.2414 | 106.1181 | 5 | 0.3045 |
| C2 | 42 | 461.6822 | 331.2485 | 34.7803 | 11 | 0.3045 |
| C3 | 42 | 468.6879 | 331.9466 | 86.8806 | 5 | 0.3044 |
| C4 | 42 | 464.4261 | 332.6780 | 47.7276 | 4 | 0.3044 |
| S1 | 42 | 473.4028 | 334.1015 | 106.0000 | 5 | 0.3044 |
| S2 | 42 | 468.2890 | 334.6784 | 65.2233 | 4 | 0.3044 |
| S3 | 42 | 460.9765 | 335.9138 | -7.9768 | 8 | 0.3046 |
| S4 | 42 | 462.4788 | 339.3260 | -20.7308 | 8 | 0.3046 |
| T0 | 42 | 462.3362 | 332.9666 | 31.1014 | 8 | 0.3044 |
| T0 | 43 | 459.4150 | 333.9359 | -6.0553 | 8 | 0.3044 |
| T0 | 44 | 461.7403 | 337.5896 | -20.8896 | 8 | 0.3044 |
| T1 | 42 | 461.9014 | 330.1934 | 43.2053 | 7 | 0.3046 |
| T1 | 43 | 465.8754 | 332.3756 | 62.2526 | 7 | 0.3046 |
| T1 | 44 | 468.1514 | 330.4112 | 89.2394 | 7 | 0.3046 |
| T2 | 42 | 470.8563 | 331.1156 | 100.5105 | 5 | 0.3046 |
| T2 | 43 | 466.3439 | 330.9712 | 74.3410 | 5 | 0.3046 |
| T2 | 44 | 463.1303 | 328.4731 | 54.7039 | 5 | 0.3046 |
| T3 | 42 | 465.9602 | 329.9099 | 74.6290 | 5 | 0.3046 |
| T3 | 43 | 467.0677 | 332.2460 | 72.6033 | 5 | 0.3046 |
| T3 | 44 | 462.2303 | 329.0487 | 46.1297 | 5 | 0.3046 |

Primary auxiliary comparison: T2 (rent + COE) versus T1 (rent only), matched by seed.
T3 adds the supplied Singapore median stock-return proxy. Company/worksite interactions are excluded.

## Limits

June 2024–March 2025 and the overlapping 16-month stress window are repeatedly examined development periods, not untouched tests.
There are only 51 distinct labeled months, with serial dependence. Row count does not multiply independent macro observations.
Intervals are descriptive, conditional on fitted models, and omit selection/fitting uncertainty.
All runs are fixed-origin. Monthly-updated forecasts are a different information setting and are not silently mixed here.
Linear controls use a jointly optimized linear lag head with weight decay; they are not closed-form Ridge fits.

See settings.json, availability.csv, training_horizons.csv, per-run checkpoints/audits, comparisons.csv and long16_scores.csv.
