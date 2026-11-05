# Frozen TabPFN-3.5-Fast embedding MLP on the categorical-month 80/20 split

The backbone uses the same one-estimator checkpoint, 77 features, all 121,005 historical context rows, and the categorical month encoding as the saved S$455.81 RMSE experiment. It is frozen. Every head training embedding is a query from an earlier forecast origin; its rent label is absent from that origin's context. The 28,995 June 2024-March 2025 holdout rows are scored only after the head settings are selected from December 2023-May 2024 historical queries.

Historical query cases: 10,800 (8,200 head training, 2,600 calibration). The head preprocessors fit on head training cases only; selected heads are retained without refitting.

| Model | Validation RMSE (S$) | Validation MSE (S$²) | MAE (S$) |
|---|---:|---:|---:|
| tabpfn | 455.814 | 207766.0 | 329.748 |
| mlp_numeric | 473.275 | 223989.5 | 362.531 |
| mlp_raw | 460.903 | 212431.4 | 340.874 |
| mlp_pca32 | 473.174 | 223893.2 | 360.293 |
| mlp_pca32_only | 467.612 | 218661.4 | 352.667 |
| ridge_pca32 | 467.999 | 219022.6 | 354.644 |
| mlp_numeric_unshrunk | 473.275 | 223989.5 | 362.531 |
| mlp_raw_unshrunk | 460.903 | 212431.4 | 340.874 |
| mlp_pca32_unshrunk | 473.174 | 223893.2 | 360.293 |
| mlp_pca32_only_unshrunk | 467.612 | 218661.4 | 352.667 |
| ridge_pca32_unshrunk | 501.428 | 251429.6 | 396.980 |
| selected | 460.903 | 212431.4 | 340.874 |

Preselected result: **mlp_raw**, outer RMSE **S$460.903**, change **+5.089** from frozen TabPFN.

The outer holdout has been inspected in prior project experiments, so this is a development comparison, not an untouched estimate of generalization. Historical calibration and outer validation have different origin dates and query populations. PCA32, numeric-only, Ridge, and unshrunk heads are controls. The selected head uses a nonzero correction only if an embedding MLP cleared the prespecified 1% historical calibration gain gate.

## Interpretation and diagnostic checks

The selected raw-embedding MLP worsens RMSE in **all ten** outer months. Its
historical calibration baseline overpredicts by S$124.08 on a weighted average,
and the head learns a mean S$128.10 downward correction. On the outer period,
the unchanged TabPFN baseline has only S$0.77 mean prediction bias, while the
head still subtracts S$48.96 on average. The correction moves predictions in
the wrong direction on this later period. A post hoc outer shrinkage calculation
would choose zero among nonnegative weights; it was **not** used to select the
head or claim an improvement.

The raw embeddings add signal relative to the numeric-only MLP (S$460.90
versus S$473.28 outer RMSE), but neither beats frozen TabPFN. This supports
**retaining the original categorical-month TabPFN prediction**. The failure is
consistent with a changing backbone bias across forecast origins and with
head transfer across different context sizes and query populations. These
checks do not identify one cause conclusively. The September 2022 historical
prediction replay matches the prior categorical-month artifact to within
5×10⁻⁹ SGD with identical row IDs, and the outer first batch also replays.
The full outer embedding extraction used at most 7.23 GiB GPU memory.
