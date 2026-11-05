# Town-specific TabPFN on the chronological 80/20 holdout

Each of 26 towns is fitted separately using its January 2021-May 2024 labeled rentals as context; all June 2024-March 2025 rentals are inferred. The pretrained TabPFN-3.5 Fast weights are frozen: `fit` supplies town-specific labeled context, not gradient training. One estimator, seed 42, no context subsampling, 77 unchanged features, causal rental histories and original temporal residual baseline. The pooled control is the existing same-feature one-estimator run. This is a previously examined development split.

## Overall performance

| Model | Rows | RMSE (S$) | MAE (S$) | Bias (S$) | Δ RMSE (S$) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pooled_77 | 28,995 | 455.814 | 329.748 | +0.768 | +0.000 |
| town_context_77 | 28,995 | 464.242 | 339.794 | -18.286 | +8.429 |

## Per-town performance

Positive bias means overprediction. Δ RMSE is town-context minus pooled.

| Town | Context | Validation | Pooled RMSE | Town RMSE | Δ RMSE | Pooled bias | Town bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ang mo kio | 6,682 | 1,601 | 439.34 | 446.78 | +7.44 | -1.81 | -44.28 |
| bedok | 7,188 | 1,719 | 449.90 | 454.29 | +4.39 | -6.15 | +9.76 |
| bishan | 2,678 | 628 | 486.17 | 492.24 | +6.07 | -0.30 | +2.25 |
| bukit batok | 4,469 | 1,136 | 432.47 | 451.29 | +18.82 | +14.99 | -104.43 |
| bukit merah | 6,573 | 1,697 | 506.60 | 530.93 | +24.33 | -38.40 | -120.34 |
| bukit panjang | 3,027 | 690 | 386.68 | 394.62 | +7.94 | +52.73 | +62.80 |
| bukit timah | 355 | 83 | 580.38 | 593.56 | +13.18 | -30.79 | +75.80 |
| central | 1,855 | 423 | 457.40 | 481.78 | +24.38 | -55.06 | -59.10 |
| choa chu kang | 4,272 | 1,028 | 423.19 | 426.44 | +3.25 | +71.51 | +58.82 |
| clementi | 4,404 | 1,025 | 486.79 | 493.57 | +6.77 | -55.81 | -71.77 |
| geylang | 3,668 | 836 | 478.66 | 483.27 | +4.62 | -31.64 | -62.78 |
| hougang | 5,746 | 1,369 | 430.56 | 439.77 | +9.21 | +50.48 | +82.60 |
| jurong east | 3,268 | 756 | 457.35 | 462.15 | +4.80 | -15.48 | -38.74 |
| jurong west | 8,441 | 2,041 | 568.39 | 569.11 | +0.71 | -66.78 | -77.75 |
| kallang/whampoa | 4,565 | 1,124 | 524.66 | 539.34 | +14.68 | -5.80 | +29.58 |
| marine parade | 1,333 | 304 | 451.67 | 483.73 | +32.06 | -47.14 | -126.53 |
| pasir ris | 3,353 | 736 | 470.54 | 479.70 | +9.17 | +29.79 | -69.50 |
| punggol | 5,300 | 1,368 | 360.82 | 360.44 | -0.38 | +73.01 | +44.94 |
| queenstown | 4,905 | 1,210 | 489.59 | 494.21 | +4.62 | -5.43 | -29.72 |
| sembawang | 2,816 | 737 | 395.91 | 395.36 | -0.55 | +57.34 | +64.31 |
| sengkang | 7,837 | 1,936 | 363.87 | 369.42 | +5.55 | +25.46 | +50.46 |
| serangoon | 2,752 | 646 | 442.79 | 453.10 | +10.31 | +2.22 | -68.17 |
| tampines | 8,113 | 1,829 | 427.10 | 433.54 | +6.44 | -6.21 | +10.09 |
| toa payoh | 4,868 | 1,120 | 470.56 | 489.71 | +19.15 | -32.56 | -119.57 |
| woodlands | 6,122 | 1,455 | 451.58 | 460.58 | +9.00 | +25.73 | +27.12 |
| yishun | 6,415 | 1,498 | 433.85 | 436.18 | +2.33 | -9.65 | -22.01 |

## Six persistent Plus overprediction towns

These towns were identified using the earlier one-estimator Plus analysis, so this subgroup is descriptive rather than an independent selection test.

| Model | Rows | RMSE (S$) | MAE (S$) | Bias (S$) |
| --- | ---: | ---: | ---: | ---: |
| pooled_77 | 7,128 | 391.230 | 270.995 | +51.967 |
| town_context_77 | 7,128 | 395.735 | 275.295 | +59.406 |

Runtime: 33.8 s fit plus 13.4 s inference; maximum PyTorch peak allocation 3.01 GiB. Process startup and feature loading are excluded.

The town context changes which rentals TabPFN sees and refits its native preprocessing. The 77 features and historical rental baseline were constructed globally and remain unchanged. No town-specific weight fine-tuning or hyperparameter selection was performed.

Artifacts: [scores.csv](scores.csv), [scores_by_town.csv](scores_by_town.csv), [monthly_scores_by_town.csv](monthly_scores_by_town.csv), [predictions_combined.csv](predictions_combined.csv), [verification.json](verification.json).

Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_by_town.py --stage all`. Resume is supported per town.
