# Matched historical 467 architecture comparison

Historical reference: 467.4027 SGD RMSE on MPS. Fresh arms ran on CPU with two threads, so exact historical reproduction is not expected. The fresh concat arm is the matched baseline.

| Arm | RMSE SGD | MAE SGD | Bias SGD | Epochs | Steps | Replay max delta SGD |
|---|---:|---:|---:|---:|---:|---:|
| concat | 467.6968 | 342.6043 | -10.4742 | 23 | 3979 | 0.000000 |
| summary | 469.4115 | 337.8739 | 37.2092 | 23 | 3979 | 0.000000 |
| sequence | 468.6702 | 337.9441 | 26.7377 | 23 | 3979 | 0.000000 |

Fresh concat minus historical RMSE: +0.2941 SGD.

Rank, fresh arms only: concat, sequence, summary.

All arms used the same fresh initial weights, post-construction training RNG state, fixed 23-epoch budget and 60-epoch cosine schedule. Validation was scored after all training, using batch size 4096. Horizon-level metrics and prediction archives are stored in each arm directory.

Historical guard: 94 files unchanged. The benchmark was previously viewed, uses one seed and one validation window, and is not an independent test.
