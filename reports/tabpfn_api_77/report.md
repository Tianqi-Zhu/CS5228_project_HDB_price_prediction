# TabPFN 3.5 API comparison on the original 77 features

All models use the same 121,005 January 2021–May 2024 context rows and 28,995 June 2024–March 2025 validation rows. All 77 original predictors and missing values are retained, with nine explicitly categorical columns. Training targets are rental residuals; the temporal baseline is added once.

| Model | RMSE (SGD) | MAE (SGD) | Bias (SGD) | RMSE change vs local Fast |
| --- | ---: | ---: | ---: | ---: |
| Local frozen 3.5-Fast (original 77) | 455.814 | 329.748 | +0.768 | +0.000 |
| TabPFN 3.5 API | 456.144 | 328.344 | +16.836 | +0.331 |
| TabPFN 3.5 Thinking API (medium) | 457.712 | 329.449 | +10.622 | +1.899 |

Server-reported timings (seconds; upload and client polling excluded):

| API model | Fit elapsed | Prediction compute |
| --- | ---: | ---: |
| TabPFN 3.5 API | 24.697 | 8.294 |
| TabPFN 3.5 Thinking API (medium) | 561.421 | 17.156 |

The standard API run requests one estimator and full context with seed 42. Thinking uses medium effort, a 600-second fit budget, MSE optimization, and the existing numeric approval-month index for time-ordered validation. The provider controls its internal search and fitted ensemble; its internal folds do not rebuild the precomputed historical features.

These runs use managed API models, without the locally trained DiagScale adapter. The local frozen 3.5-Fast result is the directly aligned feature baseline. Managed default checkpoints and preprocessing can change, so the comparison does not isolate architecture alone. This is a repeatedly inspected development holdout, not a new generalization estimate.

Each run directory contains settings, predictions, pooled and monthly metrics. Saved fit references, prediction jobs and server timings are in the matching ignored `.cache/tabpfn_api_77/` directory. No final 150,000-row refit or competition submission is produced by this validation comparison.
