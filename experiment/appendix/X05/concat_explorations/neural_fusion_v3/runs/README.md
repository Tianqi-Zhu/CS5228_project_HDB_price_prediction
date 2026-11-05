# neural_fusion_v3 runs

Check `result_status` in each `run_manifest.json` before citing anything.

| status | meaning |
|---|---|
| `REAL_CONTRACT_TOKEN_ABLATION` | block mode is v2 itself; per_feature changes only the numeric block |
| `MISMATCHED_ARCHITECTURE` | block mode also altered the context pathway; exploratory only, never a tokenisation result |
| `FAILED_LAUNCH` | the run never trained a model |
| `SYNTHETIC_FIXTURE_SMOKE` | generated inputs; code-path verification only |

No directory is deleted. `tok-probe-orig-s42` is `MISMATCHED_ARCHITECTURE`.
