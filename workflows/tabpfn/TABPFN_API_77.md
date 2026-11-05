# TabPFN API with the original 77 features

The completed [TabPFN 3.5 and Thinking comparison](reports/tabpfn_api_77/report.md)
records full validation results, monthly errors and server timings for both models.

`analysis/tabpfn_api_77.py` uses the same saved categorical-month feature matrices
as the original frozen Fast baseline and the current DiagScale submission. It
does not change features or transfer the locally trained DiagScale adapter to
the API. Managed API `fit()` conditions a pretrained model on the supplied data;
Thinking mode also searches configurations. This is distinct from local weight
fine-tuning.

The official client supports model paths such as `v3_default`, `v3.5_default`,
`v3.5-fast_default`, and `auto`. `auto` follows the server's latest default;
explicit names select a model family but its default checkpoint can change.
Account access and dataset limits are checked by the service. List the names
accepted by your installed client with `--stage models`.

## Install and authenticate

The project bundles were saved with pandas 3. The API client currently requires
pandas 2.3, so use a separate environment and export verified Parquet/NumPy
inputs with the original environment. This preserves all numeric values, missing
values, row IDs, categorical values, order and unused categories. The API runner
automatically selects these portable inputs when they exist. Exporting needs
`pyarrow` in the original environment (`python -m pip install pyarrow` if absent).

```powershell
.venv/Scripts/python.exe -m venv --system-site-packages .cache/tabpfn_api_env
.cache/tabpfn_api_env/Scripts/python.exe -m pip install -r requirements-tabpfn-api.txt
.venv/Scripts/python.exe analysis/tabpfn_api_77.py --stage export
.venv/Scripts/python.exe analysis/tabpfn_api_77.py --stage export --split submission
```

Supply `TABPFN_TOKEN`, use the client's existing login cache, or let the runner
read the key already saved by `analysis/authenticate_tabpfn.py` at
`~/.cache/tabpfn/auth_token`. `--token-file PATH` selects a different existing
key file. Tokens are never included in reports or copied into the repository.
Create a key at https://platform.priorlabs.ai/account/api-keys if needed.

## Inspect and estimate

```powershell
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage inspect
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage models
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage quote --model v3.5_default
```

Inspection is offline. A quote sends only dimensions and configuration, and does
not consume prediction quota. Quotes are in API quota units, not monetary prices.
They cover all configured prediction batches. A Thinking quote includes the fit
estimate, but its prediction estimate is provisional until the fitted ensemble
configuration is known. Cache quotes assume cache hits. If quoting is unavailable,
the runner reports the server error rather than inventing a price.

## Compare on the original chronological holdout

```powershell
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage run --model v3.5-fast_default
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage run --model v3.5_default
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage run --model v3.5_default --thinking-effort medium
```

Default inputs are
`.cache/tabpfn_v3_features/calendar_80_20_categorical.joblib`: all 121,005
January 2021–May 2024 context rows and 28,995 June 2024–March 2025 validation rows.
All 77 columns, categorical values, intentional missing features, original row
IDs and order are preserved. Only features and residual training labels are
uploaded; held-out rents remain local for scoring. Block/street identifiers are
among the existing predictors. The nine categorical column positions are sent
explicitly so integer-valued categorical month is retained as a category.

The target is `y_train - train_baseline`. Predictions use `output_type="mean"`
and add `query_baseline` exactly once. Default inference uses one estimator,
seed 42, full context (`SUBSAMPLE_SAMPLES=None`), and up to 50,000 rows per API
prediction job. This avoids multiplying the service's minimum per-request charge.
Use `--n-estimators 4` or `8` to request a larger ensemble. No local GPU is needed.

Thinking uses MSE as its optimization metric and the existing numeric
`APPROVAL_MONTH_INDEX` as its time column. The outer validation labels never
enter the fit. The provider controls Thinking's internal search; it does not
rebuild the historical features for each internal fold, so its internal scores
are not a substitute for the chronological outer result. Thinking may select
its own ensemble configuration. `--thinking-timeout 600` sets its fit budget.

For a small connectivity pilot, explicitly limit rows and give it a separate name:

```powershell
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage run --model v3.5_default --train-rows 1000 --query-rows 20 --run-name pilot_v35
```

Pilots use the first rows in the saved source order; their metrics are not the
full benchmark. Without row-limit flags every context and query row is used.

## Fit once, infer later, resume after interruption

```powershell
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage fit --model v3.5_default
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage predict --model v3.5_default
```

Re-running the same command resumes saved fits, accepted prediction jobs, and
completed batches. Fitted model references and progress live under ignored
`.cache/tabpfn_api_77/<run>/`; metrics, monthly metrics, predictions and settings
live under `reports/tabpfn_api_77/<run>/`. Resume requires the same account,
inputs, runner version and settings. Use a new `--run-name` for changes.
Deleted server fits and failed jobs surface as errors rather than silently
starting another chargeable fit. Avoid simultaneous processes using the same run.

`--fit-mode fit_with_cache` requests server-side KV caching for repeated
inference; it cannot be combined with Thinking. The default `fit_preprocessors`
uses background prediction jobs, which can be resumed from another process.

## Generate a new submission

```powershell
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage quote --split submission --model v3.5_default
.cache/tabpfn_api_env/Scripts/python.exe analysis/tabpfn_api_77.py --stage run --split submission --model v3.5_default
```

This loads `.cache/tabpfn_fast_final_submission/features.joblib`, with 150,000
labelled rows and 50,000 test rows. It retains the original March overlap cutoff
and long-horizon forecast policy. The output is a separate
`reports/tabpfn_api_77/<run>/submission_tabpfn_api.csv` with the template's
`Id,Predicted` format. Query-limited pilots do not write a submission.

The saved bundles must already exist for the export step. Rebuild validation inputs through the
existing calendar experiment, or submission inputs with
`analysis.train_tabpfn_fast_submission.prepare_features()`; the API runner does
not silently regenerate features or train a local model. Export again after rebuilding
the source bundles; the manifest records source and exported-file SHA-256 hashes.

Source: [official Prior Labs API client](https://github.com/PriorLabs/tabpfn-client).
