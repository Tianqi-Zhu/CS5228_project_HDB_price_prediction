# Spatial and temporal exploration before the final concat model

This archive adds Ziyuan’s local explorations to the existing appendix. It records earlier work, not training performed during integration. The final same-feature Transformer belongs to main/M05 and its separate reproduction package.

## What was explored

- **Macro sequence tokens:** `macro_encoders_v3` implements a DBS LSTM (63 × 7 input window, hidden 32, output 48) and a COE GRU (12 × 24 input window, hidden 32, output 48). Fit-only scaling, missing masks and padding are part of the input contract. `neural_fusion_v3` records downstream fusion experiments. These external macro sequences are different from the final model’s two MLP summary tokens derived from rental-history columns.
- **Spatial tokenization:** `spatial_token_ablation_v3` compares a grouped spatial block with individual feature tokens. In the archived fixed-seven-epoch comparison, grouped tokens scored 475.5987 and individual tokens 474.4912. The one-epoch individual-token probe scored 543.9991; it must not be used to claim spatial tokenization failed because the training budgets differ. See `refit7/summary.json`.
- **Rental history and temporal encoding:** `history_fusion_v9`, `temporal_fusion_v4` and `target_asof_v8` retain available historical run summaries. Read each record’s period, availability policy and row count before comparing scores.
- **Richer spatial-temporal inputs:** `spatiotemporal_v17` includes local-history/support, retrieval and architecture trials. Small screens and follow-up results are retained with their own configurations.
- **Market-anchor residuals:** `market_anchor_v18` explores direct versus residual prediction and alternative market anchors. Some records use a 5,000-row screening population. These are not full-validation scores and cannot be ranked directly against main’s 459.2189.

## Evidence and reuse

The subdirectories contain the original available small configuration, summary and curve files. Smoke/probe records remain named as such and are not final accuracy results. Files were copied without changing historical values. Historical instructions inside snapshots are records of past operations, not instructions to launch a queue now.

[Source snapshot and verification](../../../../reproduction/concat_explorations/README.md). Large weights, feature banks, prediction arrays and deployment bundles are not included. This is an inspectable methods/evidence archive, not a promise that every old run can be reproduced from one command.
