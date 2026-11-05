# TabFM v10: v4 anchor-residual feature bank

This package adds a local Google TabFM benchmark to the existing HDB rent project. It reads only the frozen orig/A fit and holdout matrices:56 numeric features,4 categorical columns, and the historical anchor used by v4. It predicts the residual in thousands of SGD and adds the per-row anchor back before scoring. Context sampling uses training rows only; validation labels never select contexts.

Historical scores467.40/469.01 belong to the2024-04..2025-03 historical benchmark. The older development window2023-04..2024-03 has v4 RMSE581.0164 on35,882 rows. Do not compare scores from these two windows as improvements/regressions.

## Runtime

- Official Google source1.0.1, commit `fbb665569425fd2f490c6576b3af967876fe11ff`; archive commit verified.
- Official regression checkpoint `google/tabfm-1.0.0-pytorch`,6,591,243,724 bytes, downloaded to the user's Downloads folder and linked under `reports/tabfm_v10/weights/regression/`. Safetensors header/file length checked.
- Runtime source and dependencies are isolated under `reports/tabfm_v10/runtime/`; shared environments unchanged. PyTorch2.8.0,jaxtyping0.2.36,typeguard2.13.3,SciPy1.17.1. Runtime provenance records wheel digests.
- SciPy1.15.x failed to import its PROPACK extension on this machine. An isolated official1.17.1 wheel imports successfully.
- This process sees no available MPS backend; real inference uses CPU with2 threads.
- Default TabFM internal NNLS weighting is explicitly disabled; no random-CV tuning or paid APIs.

## Commands

Run from the project root using `.venv/bin/python`. When needed set `DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib` and unset PYTHONPATH/PYTHONSTARTUP as in existing project commands.

```sh
.venv/bin/python -m pytest experiments/tabfm_v10/tests/test_benchmark.py -q
.venv/bin/python -m experiments.tabfm_v10.benchmark --run-id UNIQUE_ID --prepare-only --usage-log CURRENT_SESSION_JSONL
.venv/bin/python -m experiments.tabfm_v10.benchmark --run-id UNIQUE_ID --context-rows 512 --validation-rows 512 --estimators 1 --predict-batch 32 --usage-log CURRENT_SESSION_JSONL
```

`--validation-rows 0` requests every orig/A validation row; do not run automatically as an escalation from the screening. Existing run directories are never overwritten. The quota guard requires a recent genuine token_count/rate_limits event in the specified current-session log and stops if either subscription window reaches90% used or evidence is stale/missing. It cannot change account billing settings. No background agent or paid API fallback is used.

## Evidence

- `reports/tabfm_v10/runs/v10-prepare-orig-s42/`: fixed512-row validation selection, baseline RMSE570.7602.
- `reports/tabfm_v10/runs/v10-real-smoke-orig-s42/`: real32-context/8-query inference; smoke only, not an accuracy ranking.
- `reports/tabfm_v10/runs/v10-screen-orig-s42/`: first512-context/512-query,1-estimator screening; consult manifest status and predictions, not the existence of a directory.

The runner saves IDs, horizons, source/input hashes, same-row baseline, metrics and a fixed equal-blend diagnostic. Small sampled development scores do not replace the historical benchmark or establish hidden-test performance.
