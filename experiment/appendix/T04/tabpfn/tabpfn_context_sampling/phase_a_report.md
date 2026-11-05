# Inference-only context retrieval ablation

The saved corrected two-view DiagScale adapter, TabPFN-3.5-Fast checkpoint, 77 model features, residual reconstruction, one inference view, preprocessing seed, calibration rows and outer validation rows are unchanged. Only which **already labeled** rows are supplied as TabPFN context changes. The full-context S0 prediction is reused from the validated saved run. Among sampled strategies, all use the same **10,000-row** budget and `historical_only` policy. S1 uniform is the matched-budget control; S2 weights by recency with six-month exponential decay; S3 selects spatial/property neighbors with month and property dissimilarity; S4 allocates 40% local, 35% recent, 25% diverse history. The query batch shares one selected context. No validation label, query label or query duplicate group enters retrieval. Model rental-history features continue to use the complete eligible earlier fitting period, as in the original pipeline.

## Results

| Strategy | Context rows | Earlier-month weighted calibration RMSE (S$) | Outer RMSE (S$) | Outer MAE (S$) | Outer bias (S$) | Outer fit + prediction time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 full context | 121,005 outer | **520.848** | **455.212** | **330.625** | −2.798 | 1,185 s saved run |
| S1 uniform | 10,000 | 530.508 | 461.266 | 333.116 | +16.417 | 15.56 s |
| S2 recency | 10,000 | 523.683 | 460.799 | 342.428 | −52.429 | 13.52 s |
| S3 local/property | 10,000 | 523.857 | 457.317 | 330.850 | +5.453 | 13.62 s |
| S4 mixture | 10,000 | 524.526 | **456.390** | 333.575 | −15.516 | 14.03 s |

S4 improves outer RMSE by **S$4.876** over the **same-budget** S1 uniform control, but remains **S$1.178** above S0 full context. S3 improves over uniform by S$3.948. S4 has lower monthly RMSE than uniform in nine of ten months, but beats full context in only one. S2 ranked best among sampled strategies on the earlier 600 calibration rows and scored poorly on the later outer period, so calibration alone would have selected the wrong retrieval strategy here.

Paired bootstrap resampling of the **ten months** (20,000 draws, seed 42) gives an outer RMSE difference S4−S1 of **−S$4.92**, with a 2.5–97.5% interval **[−6.02, −3.49]**, and S4−S0 of **+S$1.17 [0.40, 1.99]**. This describes variation across the already examined ten-month development period, not uncertainty on a new forecast period or Kaggle test score. The month-bootstrap inputs and result are saved in `phase_a_outer_budget10000_queriesfull_seed42/month_cluster_bootstrap.json`.

Sampled inference used about **3.62 GiB** peak PyTorch allocation versus **8.29 GiB** for the saved full-context adapter. S3/S4 retrieval took about four seconds for all 28,995 outer queries; S1/S2 retrieval took under 0.05 seconds. The full-context timing was reused and is not the zero placeholder in the new `scores.csv`. Context age, spatial coverage, monthly/town errors and cost plots are saved beside the outer score table. All five outer prediction files contain 28,995 unique row IDs, no missing values, and recomputed RMSE matching the recorded scores.

The full-context S0 remains the RMSE choice when compute is acceptable. S4 is the strongest tested option for a 10,000-row inference budget. The next experiment separately changes **training episode support** while keeping the month-coherent supervised queries and all other settings identical; its new chronological query construction must be compared with its own matched uniform control, not attributed solely to support retrieval relative to the historical random-episode adapter.

Reproduce the calibration/outer ablations with `python analysis/compare_tabpfn_context_sampling.py --stage calibration --context-budget 10000` and `python analysis/compare_tabpfn_context_sampling.py --stage outer --context-budget 10000`. A retrieval-only smoke test is `python analysis/compare_tabpfn_context_sampling.py --stage calibration --context-budget 1000 --query-limit 120 --dry-run`.
