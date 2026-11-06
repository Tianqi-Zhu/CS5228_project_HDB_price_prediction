# Latest TabPFN anniversary integration

The merge imports `experiment/tabpfn-anniversary-history` at commit `7c01b8bc2bf499b38d341d40960cc9d59d362c7e` into main. Main's active `analysis/`, tests, environments, and validated F1/Concat entry points retain their canonical implementations. The incoming branch's exact source bytes are available through the existing isolated TabPFN workflow rather than shadowing those modules.

## Run the integrated workflow

From the repository root, with the numerical/test dependencies installed:

```powershell
python review_tabpfn.py verify
python review_tabpfn.py prepare --workspace output/tabpfn-anniversary-merged
python review_tabpfn.py test --workspace output/tabpfn-anniversary-merged
python review_tabpfn.py run --workspace output/tabpfn-anniversary-merged anniversary -- --help
python review_tabpfn.py run --workspace output/tabpfn-anniversary-merged anniversary-blend -- --help
```

The new named entry points are `anniversary`, `anniversary-blend`, `half-median`, and `flat-model-grouping`. Explicit arguments are required, so listing or preparing the workflow does not start training. Full inference still requires the original licensed checkpoint, feature bundles and, for tree refits, the documented zch worktree. This merge does not retrain research models, download checkpoints, authenticate an API, or upload a Kaggle submission.

The original seven selected test files remain active, with the three new feature/filter/category test files. A small unittest adapter runs the unchanged function-style anniversary test. Other original branch tests remain under `workflows/tabpfn/archived_tests/` for explicit use; they are not silently counted as executed tests.

## Source and results

- The latest analysis, documentation, requirements and test source mapping is in [integration.json](../../review/checks/anniversary-merge/integration.json).
- Per-file byte hashes and executable entries are in [the workflow manifest](../../workflows/tabpfn/manifest.json).
- Latest validation and candidate export records are in [the anniversary study](../../reports/tabpfn_anniversary_history_80_20/report.md), [tree/TabPFN blend](../../reports/zch_anniversary_tabpfn_blend/report.md), [half-median ablation](../../reports/half_median_zch_tabpfn_ensemble/report.md), and [category-grouping ablation](../../reports/flat_model_grouping_ensemble/report.md).

The candidate `submission_zch_tabpfn87_equal_150k.csv`, generator and provenance are now present in the imported branch. Their availability does not by itself verify that the candidate is the file associated with the previously recorded screenshot score of 413.19635. That identity and test-score linkage remain unverified; no development RMSE is relabelled as a Kaggle score.

Main's original raw CSVs, registered frozen F1/Concat evidence and model recipes remain unchanged. Older incoming read-only reports retain their historical paths for provenance, while main's generated-output defaults stay under ignored `output/`. Tests and integrity checks are recorded separately from historical model-quality results.

## Verification and runtime limitation

The merge checks passed 78 main tests and 38 isolated TabPFN tests, with no skips. Main's existing analysis/test files, F1/Concat source and frozen artifacts are unchanged. The original 50,000-row F1 submission replay passes its numerical tolerance (maximum absolute difference about 9.1e-13 SGD). See [the current validation record](../../review/checks/anniversary-merge/summary.json).

The stronger fresh-refit CSV byte-hash check used by `python -m analysis.main planned` does **not** pass in this runtime, including after trying its pinned NumPy/pandas versions. Recorded Y/J RMSE and MAE match exactly in the separate numerical replay, with bias differences below 4e-15 SGD. The original byte checker remains unchanged; numerical agreement is not reported as byte-identical reproduction. This is a recorded verification limitation, not a newly measured model-quality result. No scientific training is performed during this merge.
