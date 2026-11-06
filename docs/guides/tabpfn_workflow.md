[Chinese version](tabpfn_workflow.zh-CN.md)

# TabPFN isolation and execution

| Entry point | Purpose |
|---|---|
| `python review_tabpfn.py list` | Inspect original entry points and F2 status in the current branch |
| `python review_tabpfn.py verify` | Verify original code, results and derived-data hashes without training |
| `python review_tabpfn.py metrics` | Recompute existing pooled RMSE, MAE and bias from monthly summaries; no model inference |
| `python review_tabpfn.py preflight` | List local packages, checkpoints, missing resources and historical hash differences; no downloads or authentication |
| `python review_tabpfn.py test` | Run bundled original tests in an independent working directory |
| `python review_tabpfn.py prepare` | Copy original code, shared data and results into `output/tabpfn-workspace/` |
| `python review_tabpfn.py run aligned -- --help` | Inspect original arguments without training |

See [split decisions](../planning/tabpfn-branch-split.md) and the [directory classification](../planning/tabpfn-classification.csv). Actual verification records are in `review/checks/tabpfn-split/`.

## Environment and architecture

`workflows/tabpfn/` preserves the original TabPFN `analysis/`, `tests/`, `environment.yml` and `requirements-*.txt`. `environment.anniversary-original.yml` preserves the merged branch's environment; existing root environments continue to serve F1. Original README/API documentation is retained as source material; interpret its commands relative to the assembled isolated workspace.

```powershell
conda env create -f workflows/tabpfn/environment.yml
conda activate cs5228-hdb
python -m pip install -r workflows/tabpfn/requirements-tabpfn.txt
```

Prepare CUDA following the original files. A fresh environment installation was not verified in this round; recorded GPU and NumPy/Pandas versions differ from this machine, so full inference needs separate verification. The API route requires `requirements-tabpfn-api.txt` and is not invoked here.

Same-named original modules remain inside the isolated workflow. `review_tabpfn.py` places them under `output/` and runs them in a separate process, preserving root `analysis/`, the fixed F1 package and raw data. Outputs/caches stay in that workspace. Use `--workspace output/tabpfn-new-run` for a new directory; the entry point refuses reuse when an existing workspace differs from the new snapshot.

## Data and results

| Content | Action |
|---|---|
| Nine original competition CSVs | Continue using root `data/`, copied during workspace assembly; preserve all 150,000 original training samples |
| Cleaned, deduplicated and other derived data | Stored in appendix's `data/processed/` and explicitly identified as alternative data definitions |
| Original result directories | Retained by classification; restored as `reports/<original-directory-name>/` in the assembled workspace for relative-path traceability |
| Original code and provenance hashes | `workflows/tabpfn/manifest.json`; historical source/code hashes are not rewritten to imply a new run |
| Final submission CSVs and member predictions | Files absent from the source Git repository remain marked missing; predictions are not fabricated from summary tables |

`metrics` checks summary consistency only. If component monthly tables omit row counts, it weights months by corresponding counts in the raw training data; earlier sampled validations are excluded from complete-holdout checks. Without original row predictions, this cannot verify inference, row alignment or absence of leakage.

## Execution boundaries

`run` requires explicit original arguments; omitting them does not trigger a script's default training. Full inference requires the appropriate packages, checkpoint, feature/adapter caches and hashes. `preflight` lists current gaps; restore resources before running the original recipe. Main includes essential entry points; appendix retains full exploration and historical submission commands.

The candidate export, generator source and provenance for `submission_zch_tabpfn87_equal_150k.csv` are now integrated from the anniversary branch. Its file identity relative to the historical screenshot score remains unverified. Existing ABCDE/ABDE are different candidates and cannot replace or confirm F2. See the [anniversary integration guide](anniversary_merge.md) for new commands and verification.

## Validation dependency snapshot

`workflows/tabpfn/requirements-validation-extra-lock.txt` records the repository-local supplemental packages used for original test validation. It complements the existing cs5228 numerical/PyTorch environment; it is not a complete Conda lock. Original requirement files remain unchanged. The tests are run offline.

One appendix integration test needs two untracked archived school HTML inputs (2020/2023). `test-resources.json` declares their exact paths. The test wrapper skips only that named test when the files are absent, reports the skip, and leaves every original test file unchanged. Supplying the real cached files enables the original assertion.

Model checkpoints and legacy caches belong inside the generated workspace, e.g. `output/tabpfn-workspace/.cache/tabpfn/`. `preflight --workspace output/another-run` inspects that exact execution directory. It does not silently borrow the root F1 model cache.

On Windows, original experiment paths can exceed the default Git path limit. This repository enables `git config --local core.longpaths true`. When cloning into a deep directory, use `git -c core.longpaths=true clone <repository-url>` so the archived paths are preserved.

## Best observed standalone model

Use [the 87-feature reproduction guide](tabpfn_best.md) for the tested local validation and full-data inference commands. `run_tabpfn_best.py` rebuilds missing features from raw data and separates archived reference predictions from newly generated outputs.

For regenerated original and grouped-category tree/TabPFN submissions, see [the ensemble guide](tabpfn_ensemble.md).
