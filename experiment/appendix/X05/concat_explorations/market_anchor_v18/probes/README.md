# v18 market-anchor screen

These fixed 30,000/5,000 chronological screen results are exploratory development results, not a full-split confirmation or unseen test. No validation labels were used to build the anchors or estimate the training-mean corrections. Saved F0 neural screen RMSE is 481.36 SGD; prior original and refreshed local-comparable controls are linked and hashed in `results.json`.

| Anchor or correction | Train RMSE | Validation RMSE | Validation bias (pred−actual) |
|---|---:|---:|---:|
| v17 original cumulative anchor | 806.10 | 719.00 | −528.55 |
| v17 original + train mean residual | 642.84 | 489.25 | −42.19 |
| v17 refreshed local 12-month anchor | 744.05 | 494.99 | −54.91 |
| v18 market level | 749.77 | 510.86 | −55.57 |
| v18 market level + train mean residual | 637.12 | 610.96 | +339.68 |
| v18 market trend | 652.57 | 516.74 | −1.06 |
| v18 market trend + train mean residual | 629.08 | 544.75 | +172.44 |
| Original + cold flag, horizon and group correction | 576.02 | 495.25 | −73.01 |

The final correction is a supervised train-only diagnostic, not a historical anchor fed into another learner; its cold flag and fixed penalties were added after the earlier v17 probe results. Market-level and market-trend settings were fixed before their own evaluation. The trend anchor's nearly zero aggregate bias does not imply lower RMSE or a better full predictor. Adding a global training-mean correction transfers poorly here.

The market level is a transaction-weighted trailing 12-month town×flat-type mean, shrunk toward flat-type then global prices with pseudo-count 10. The trend fits weighted lines using only the trailing window available at each individual origin, shrinks town slope toward flat-type and flat-type toward global with pseudo-count 50, and caps the combined intercept-and-horizon adjustment at ±20% of the past group level. The no-trend counterpart isolates this extrapolation choice. These are forecasts, not observed future group means. Historical query origins are rebuilt separately, so no final-cutoff trend coefficients are used for earlier queries. Global no-history fallback is 0 SGD with `missing=True`; 827 fit rows and no validation rows have no history, and all are retained.

Observed group composition can change over time: sampled floor-area means are 93.58 sqm in fit and 92.36 in validation, with medians 92 in both. `composition_by_group.csv` gives group-level counts and area summaries; changes in other property attributes can also masquerade as market movement.

Run the probes with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-model-benchmark/bin/python -m experiments.market_anchor_v18.probe`. Anchor build took 1.20 seconds in the observed run; probes took 0.12 seconds before the additional cold correction. `results.json`, saved predictions and `group_metrics.csv` provide paired comparisons and subgroup errors. Data manifests record bank/source hashes and as-of rules; `data/schema.json` is the Phase A screen-only gate. Five feature tests passed, including hand-computed parent smoothing, no-history fallback, future-label perturbation on synthetic origins and real validation rows, row alignment and target reconstruction.
