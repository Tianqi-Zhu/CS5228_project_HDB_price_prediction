# Cross-attention v11: orig/A screening

All results use the same orig/A fit (52,310 rows) and holdout (35,882 rows), seed 42, CPU, and fit-only preprocessing. Stage B was not scored. The historical 467.4027 result uses a different evaluation window and is not comparable to these holdout scores.

| Arm | Best epoch | RMSE SGD | MAE SGD | Bias SGD | Parameters | Seconds | Replay max delta SGD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| concat | 23 | 584.7115 | 454.2965 | 42.7067 | 87249 | 1722.2 | 0.000000 |
| summary | 23 | 580.3508 | 451.3627 | 38.9998 | 87249 | 1521.8 | 0.000000 |
| sequence | 23 | 583.6198 | 452.2960 | 52.0160 | 87249 | 1723.3 | 0.000000 |

Horizon breakdown (months):

| Arm | Horizon | n | RMSE SGD | MAE SGD | Bias SGD |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat | 1 | 2964 | 617.8171 | 506.9863 | -52.5273 |
| concat | 2 | 2978 | 610.4051 | 496.9265 | -61.8717 |
| concat | 3 | 3035 | 624.3674 | 508.5861 | -64.7799 |
| concat | 4 | 3207 | 616.4337 | 501.2962 | -98.2574 |
| concat | 5 | 2974 | 595.5397 | 472.2522 | -21.5073 |
| concat | 6 | 2747 | 582.4059 | 453.4914 | 24.4521 |
| concat | 7 | 3039 | 566.0139 | 435.9382 | 65.1372 |
| concat | 8 | 2925 | 574.3664 | 434.4482 | 112.1979 |
| concat | 9 | 3008 | 556.1330 | 416.9994 | 130.7143 |
| concat | 10 | 3171 | 556.9567 | 416.1373 | 163.4943 |
| concat | 11 | 2650 | 556.8841 | 406.4148 | 167.1387 |
| concat | 12 | 3184 | 550.2363 | 399.0563 | 157.0914 |
| summary | 1 | 2964 | 614.1931 | 501.3045 | -9.0596 |
| summary | 2 | 2978 | 606.4692 | 490.7449 | -22.0362 |
| summary | 3 | 3035 | 623.1991 | 504.1342 | -29.6840 |
| summary | 4 | 3207 | 613.7675 | 496.8050 | -73.4197 |
| summary | 5 | 2974 | 596.1746 | 470.8224 | -7.2150 |
| summary | 6 | 2747 | 581.5212 | 453.1229 | 23.9439 |
| summary | 7 | 3039 | 565.2673 | 437.5540 | 49.1031 |
| summary | 8 | 2925 | 570.7060 | 434.0121 | 86.0479 |
| summary | 9 | 3008 | 550.8892 | 416.5285 | 100.0652 |
| summary | 10 | 3171 | 548.8001 | 411.6632 | 129.9778 |
| summary | 11 | 2650 | 546.3444 | 403.1261 | 123.1197 |
| summary | 12 | 3184 | 536.4866 | 393.9819 | 104.5101 |
| sequence | 1 | 2964 | 616.9850 | 505.7187 | -45.9865 |
| sequence | 2 | 2978 | 609.7197 | 495.6552 | -46.8958 |
| sequence | 3 | 3035 | 622.7535 | 505.5727 | -44.1219 |
| sequence | 4 | 3207 | 613.1389 | 497.2077 | -75.4539 |
| sequence | 5 | 2974 | 595.2140 | 469.8500 | -3.0606 |
| sequence | 6 | 2747 | 581.8245 | 452.6827 | 40.5816 |
| sequence | 7 | 3039 | 567.4876 | 436.4479 | 75.3592 |
| sequence | 8 | 2925 | 574.9547 | 433.6847 | 115.4727 |
| sequence | 9 | 3008 | 556.7403 | 416.6153 | 132.9987 |
| sequence | 10 | 3171 | 555.3751 | 412.9343 | 165.8844 |
| sequence | 11 | 2650 | 554.3131 | 403.1544 | 165.5799 |
| sequence | 12 | 3184 | 546.2111 | 395.2936 | 152.0958 |

The concatenation arm retains the original 3-layer, 63-token FT plus DBS/COE architecture. Both cross arms replace only the final attention block, use the first two self-attention layers for tabular queries and share all initial weights. Sequence memory uses 63 DBS steps and 12 COE months, with sinusoidal lag and existing source embeddings, and learned missing-window fallback. All arms have 87,249 parameters.

The historical outer model has 87,297 parameters because its B/fit flat-model vocabulary has 20 rows versus 19 in A/fit; the other vocabulary sizes are 27, 7, and 2 in both. The additional 48-dimensional embedding row explains the difference.

This is one development window and one seed, used for screening. Early-window, multiple-seed, and independent-test confirmation remain outstanding. Independent specification and code-quality reviews found no blocking issues. The root reviewer ran all six tests and checked the 58-file historical guard. Full training curves, preprocessing, source and input hashes, predictions with IDs/origins/horizons and reconstructed-checkpoint replay checks are stored alongside this report. Checkpoints support prediction reconstruction but do not contain optimizer/scheduler state and cannot resume training. Peak memory was not measured: process inspection was denied by the environment.
