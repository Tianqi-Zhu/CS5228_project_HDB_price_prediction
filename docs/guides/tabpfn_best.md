# Reproduce the best observed standalone TabPFN model

The strongest saved standalone TabPFN development result is `anniversary87`:
RMSE **441.95926939 SGD**, MAE **318.21583739 SGD**, bias **+4.10837143 SGD**.
It uses 121,005 fitting rows through May 2024 and all 28,995 validation rows
from June 2024 through March 2025. The previously examined holdout supports
development comparisons; it is not an untouched estimate of generalization.

The 87 inputs are the categorical-month, aligned temporal/residual setup's
77 features plus ten same-block/flat-type anniversary fields: count, mean rent,
mean market-normalized rent, normalized-rent standard deviation and low-rent
fraction, at 12- and 24-month lags with a one-month window on each side.
The 92-feature extension did not improve RMSE. Tree/TabPFN blends are separate
models and their scores must not be attributed to standalone TabPFN.

## Run from main

Use the existing CUDA environment or install the archived dependencies in
`workflows/tabpfn/environment.yml` and `workflows/tabpfn/requirements-tabpfn.txt`.
The tested local SDK is TabPFN 9.2.0 with PyTorch 2.11.0+cu128.
The runner uses local inference, with no paid API calls.

The checkpoint is ignored by Git. The default location is
`.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors`; its presence in the source
author's recorded environment does not imply it exists in another checkout.
Run `preflight` first, or supply the same checkpoint with `--checkpoint PATH`.
SHA256 must be
`14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550`.
The runner does not download a substitute checkpoint or authenticate for you.

```powershell
git switch main
python run_tabpfn_best.py preflight
python run_tabpfn_best.py validate
python run_tabpfn_best.py predict
```

`validate` rebuilds features from the raw CSVs when they are absent, fits the
87-feature context and predicts the full validation set. `predict` builds the
corresponding features for all 150,000 labeled rows, fits that full context,
and predicts all 50,000 unlabeled test rows. Both use the frozen Fast checkpoint,
two estimator views, seed 42, batches of 200 and no context subsampling.
Fitting prepares the model's in-context cache; it does not fine-tune its weights.
The model predicts rent minus a temporal baseline; that baseline is added once.

The full-data builder uses strictly prior histories for March 2025 test rows,
and holds temporal forecasts flat beyond their fitted ten-month horizon.
Test performance cannot be computed without the hidden test labels.

Outputs are confined to `output/tabpfn-best/`:

| File | Purpose |
|---|---|
| `reports/tabpfn_anniversary_history_80_20/scores.csv` | New validation metrics |
| `reports/tabpfn_anniversary_history_80_20/predictions_anniversary87.csv` | Validation row IDs, labels and predictions |
| `reports/tabpfn_anniversary_history_80_20/timing_anniversary87.json` | Runtime, GPU memory and prediction hash |
| `submission_tabpfn87_150k.csv` | Standalone TabPFN test submission (`Id,Predicted`) |
| `run-provenance/` | Raw input hashes, checkpoint, package versions and run configuration |

The default workspace can resume its own partial prediction batches. To prove
a new fit and inference run, choose a new workspace; the runner never imports
old predictions from the project root:

```powershell
python run_tabpfn_best.py validate --workspace output/tabpfn-validation-fresh
python run_tabpfn_best.py predict --workspace output/tabpfn-test-fresh
```

`prepare` rebuilds validation features without GPU fitting. For faster replay
of trusted existing feature bundles, explicitly add `--assets-from .`.
Only checkpoint and feature assets whose hashes match their manifests are
copied. Saved prediction CSVs and partial predictions are never copied.
The historical sources and root F1 analysis package remain intact.

## Verification on main

See [replication evidence](../../review/checks/tabpfn-replication/summary.json)
for the tested main snapshot, package versions, numerical comparisons and scope.
The first fresh validation run on an RTX 4080 took 28.35 seconds to fit and
122.70 seconds to predict, with 9.13 GiB peak allocated CUDA memory. That replay
used verified historical feature bundles and reproduced every historical
validation prediction exactly, including the CSV SHA256. Raw feature rebuilding
is checked separately in the evidence record. Timing varies by hardware and
feature generation adds CPU time.

The current local RTX 4060 Laptop has 8 GiB VRAM and lacks the default SDK/checkpoint;
the recorded 9.13 GiB validation allocation and 11.22 GiB full-data allocation are
from a different machine. CPU component reconstruction via
`python -m analysis.main final-pair` remains available without those assets.

The new CLI was also checked with a second fresh validation fit/inference run;
its prediction CSV again matched exactly. Full-data raw features were rebuilt
and matched historical values, but a fresh 150,000-row GPU fit and 50,000-row
inference run were not completed in this audit; validation was prioritized.
