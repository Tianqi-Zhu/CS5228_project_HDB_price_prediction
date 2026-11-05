# Historical concat exploration archive

This package preserves historical local Python source and small experiment evidence for appendix X05/X06. Source is isolated under `source/experiments` and `source/analysis` so it does not replace the curated branch’s active analysis modules. The snapshot includes dependent earlier experiments for source inspection. It excludes the final main integration’s feature builders and capacity runs.

## Read and verify

From the repository root:

```sh
python3 reproduction/concat_explorations/verify.py
python3 review.py verify
```

These commands use the standard library and perform no training, installation or network operation. The source manifest records the original relative path and SHA256 for every copied file. Historical README commands, queue status files and absolute paths are retained verbatim as evidence and must not be interpreted as current operational instructions.

## Scope and limits

The archive includes source, configuration, available selected summaries and small training curves. It does **not** include prepared feature banks, large checkpoints, TabFM upstream weights/runtime, raw predictions, deployment archives or all intermediate logs. Historical code can depend on those omitted artifacts, old relative paths and environment-specific libraries; it is not wired into the active CLI and is not an end-to-end replay bundle. Available results are archived rather than rerun or independently rescored in this integration.

[Spatial and temporal evidence](../../experiment/appendix/X05/concat_explorations/README.md) · [Architecture and TabFM evidence](../../experiment/appendix/X06/concat_explorations/README.md)

The final 128-dimensional, six-layer model and its submitted CSV/weights belong to the separate main branch integration. Do not add its public score to an earlier experimental run.
