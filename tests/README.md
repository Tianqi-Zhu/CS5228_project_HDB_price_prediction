[Chinese version](README.zh-CN.md)

# Main tests and execution verification

After installing the corresponding environment, run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m analysis.validation.repository_audit
python -m analysis.main planned
```

| Scope | Currently available test files |
|---|---|
| Data, shared pipeline, common modules and saved results | `test_pipeline.py` |
| Spatial auxiliary features and feature selection | `test_auxiliary_features.py`, `test_feature_selection.py`, `test_feature_evidence.py` |
| History, anniversary windows and blend boundaries | `test_second_wave.py`, `test_third_wave.py`, `test_ninth_wave.py`, `test_tenth_wave.py` |
| Row IDs, labels, grouped errors and metrics | `test_main_validation.py` |
| F2 component order, numeric validity and line-ending identity | `test_final_pair.py` |
| MLP isolation, data identity and four submissions | `test_mlp_integration.py`; standalone runtime tests run through `mlp test` |
| Feature-metadata corrections and matched cleaning comparisons | `test_main_evidence_completion.py` |
| F1 fixed weights, member order and support groups | `test_f1_validation.py` |
| CLI help and argument forwarding | `test_main_cli.py` |
| Concat integration, gradients and Windows/POSIX compatibility | `test_concat_integration.py`, `test_concat_portability.py` |
| Separation of generated outputs and frozen evidence | `test_output_layout.py` |

For a focused check use `python -m unittest discover -s tests -p test_output_layout.py -v`. Tests for other experiments in the old full repository are outside current main coverage and must not be counted from old documentation.

TabPFN uses its own working directory to avoid shadowing main's `analysis` package:

```powershell
python review_tabpfn.py prepare --workspace output/tabpfn-tests
python review_tabpfn.py test --workspace output/tabpfn-tests
```

Original TabPFN tests require workflow dependencies; missing external source-branch resources are controlled by its declaration file and explicitly reported. They neither download weights nor call APIs. Keep main's temporary third-party paths separate from the TabPFN SDK; see the [running guide](../docs/guides/running_experiments.md).

Tests check implementation and information boundaries, not generalization or full retraining. `review.py verify` and `repository_audit` check file identity, syntax and navigation; `planned` recomputes frozen predictions and checks completed F1 generation. See [reproduction](../reproduction/f1/README.md) for full training, Y/J fixed recipes, original replay and large-model checks.

Data, model results and final conclusions rely on their experiment evidence. Incomplete F2 work remains marked pending. Historical acceptance records: [review/CHECKS.md](../review/CHECKS.md).
