[Chinese version](README.zh-CN.md)

# Tests and execution checks

Both merged workflows are covered. Original TabPFN checks retain their module
names; overlapping zch checks are also available as `test_zch_*.py`.
`test_merged_workflows.py` checks namespace separation and output paths.

Activate `cs5228` and run from the repository root:

```powershell
conda activate cs5228
python -m unittest discover -s tests -v
python -m analysis.main verify
```

The first command discovers all `test_*.py`, checking data handling, temporal boundaries, model adapters and outputs.
The second checks the environment, saved report metrics, source hashes and final blend components.
They serve different purposes; passing both does not establish retraining all ten waves or GPU compatibility on another machine.

## Find tests by topic

| Issue | Test files |
| --- | --- |
| Shared pipeline, model saving and output locations | `test_pipeline.py`, `test_project_paths.py`, `test_output_defaults.py` |
| Spatial auxiliaries, economic signals and feature selection | `test_auxiliary_features.py`, `test_economic_feature_ablation.py`, `test_economic_signals.py`, `test_feature_selection.py`, `test_feature_evidence.py` |
| Model search, TabM and spatiotemporal hybrids | `test_model_search.py`, `test_neural_tabm.py`, `test_spatiotemporal_hybrid.py` |
| Time cutoffs, features and blending across ten waves | `test_frontier_search.py`, `test_second_wave.py` through `test_tenth_wave.py` |
| Panel, spatial and correlation studies | `test_group_temporal_panel.py`, `test_spatial_controls.py`, `test_temporal_correlations.py` |
| Source-audit key matching | `test_source_overlap.py`, a separate study that supplies no test-rent labels to the final method |
| Experiment index, reading reports and component completeness | `test_experiment_catalog.py`, `test_report_bundle.py`, `test_artifact_verification.py` |
| Missing resources, changed hashes, path escapes and preflight without modeling dependencies | `test_resource_preflight.py` |
| Artifact coverage, invalidation status, metric context and final-component tracing | `test_artifact_catalog.py` |

Existing test modules stay in place to preserve historical commands. Use a discovery pattern for one category:

```powershell
python -m unittest discover -s tests -p test_project_paths.py -v
python -m unittest discover -s tests -p 'test_*wave.py' -v
python -m unittest discover -s tests -p test_artifact_verification.py -v
```

## Dependencies and interpretation

- Full tests require competition data and the core environment. Saved-result consistency is a local integration check: a new clone without `output/experiment_runs/` explicitly skips it; an existing but incomplete directory or inconsistent metrics still fail. Prepare artifacts via the [running guide](../docs/guides/running_experiments.md).
- Optional TabM, RealMLP, ModernNCA and TabDPT adapter tests explicitly skip missing packages/author sources with setup guidance, but fail when existing files import or behave incorrectly. Missing core dependencies such as RTDL do not count as optional skips.
- Some adapter tests use small temporary samples or mock trainers; they check implementation behavior, not actual model RMSE.
- Reloading and hashes establish artifact integrity, not feature generalization, temporal-validation performance or Kaggle scores.
- Tests use controlled small samples and saved results; they neither upload to Kaggle nor launch ten-wave full training.

Environment: [dependencies](../requirements/README.md); data roles: [data guide](../data/README.md).
