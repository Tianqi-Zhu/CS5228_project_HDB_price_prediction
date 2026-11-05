# Spatial and temporal TabPFN context sampling experiment

## Setup and information boundary

This experiment reuses the saved 77-column categorical-month residual feature artifact, its exact column order and nine categorical indices, the TabPFN-3.5-Fast checkpoint, and the corrected DiagScale implementation. No feature or target transformation was changed. See [audit.md](audit.md) for hashes, paths, versions and the evidence for selecting the original two-view, learning-rate-0.01 adapter. The outer validation contains 28,995 labeled rows from June 2024 through March 2025; its 121,005 fitting rows end in May 2024. The earlier checkpoint-selection calibration uses 102,829 fitting rows before December 2023 and 600 sampled December 2023–May 2024 queries. Rent residuals are predicted and the saved temporal baseline is added once to recover monthly rent.

The main policy is `historical_only`: every directly selected labeled support row predates the earliest query month. The sampler also excludes matching raw transaction groups, repeated row IDs and query IDs. The existing rental-history feature pipeline builds training-row features from earlier forecast origins; calibration and outer query histories are rebuilt at their fixed origins. Label publication lag is unknown, so prior-month labels are treated as available. A separate `available_labeled_pool` mode exists for interpolation, but is not used in the chronological scores below. Hidden Kaggle test labels are unavailable; no test score is claimed.

S0 is the exact saved original adapter with its original full context. S1 is matched-budget uniform historical sampling; S2 weights history by a six-month exponential decay; S3 selects spatial/property neighbors; S4 allocates 40% local, 35% recent and 25% geographically and temporally diverse rows. S3 distributes local retrieval across query anchors rather than using one location. Retrieval coordinates, dates, area, town, flat type and building age are kept separate from the 77 model input columns. The tested sampled inference budget is 10,000 rows. No ratio, decay, learning-rate or view-count grid was selected on the outer period.

## Phase A: change inference context only

The original corrected DiagScale adapter is fixed in this table; no weights are updated. The same 28,995 outer validation rows, one inference view, seed and model features are used throughout.

| Inference context | Rows | RMSE (S$) | MAE (S$) | Inference GPU allocation | Fit + prediction time |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 full history | 121,005 | **455.212** | **330.625** | 8.29 GiB | 1,185 s, saved reference run |
| S1 uniform | 10,000 | 461.266 | 333.116 | 3.62 GiB | 15.6 s |
| S2 recency | 10,000 | 460.799 | 342.428 | 3.62 GiB | 13.5 s |
| S3 local/property | 10,000 | 457.317 | 330.850 | 3.62 GiB | 13.6 s |
| S4 mixture | 10,000 | **456.390** | 333.575 | 3.62 GiB | 14.0 s |

S4 improves RMSE by **S$4.876** over equal-budget uniform sampling and is better in nine of ten validation months. It remains **S$1.178 worse** than the full-context S0 result and beats S0 in only one of ten months. Resampling the ten validation months 20,000 times gives an S4–S1 RMSE difference median of −S$4.92 with a 95% interval of [−6.02, −3.49], and an S4–S0 median of +S$1.17 [0.40, 1.99]. These are descriptive intervals for this repeatedly examined development period, not confidence bounds for new forecast months. Retrieval for S3/S4 took roughly four additional CPU seconds across all outer queries. The original full-context timing came from its saved run; run-to-run timing variation prevents interpreting the time ratio as a controlled speed benchmark.

## Phase B: change fine-tuning support

The installed TabPFN fine-tuner originally forms random mixed-month episodes. Under a strict historical policy, that construction often puts an early query in a mixed episode and leaves no permissible support. This experiment therefore uses a separate opt-in path with **same-month supervised queries from January 2022 onward**. Within each epoch, all strategies use identical query IDs; only the 820 labeled support rows per episode change. The 204 query rows, 100 episodes per epoch, learning rate 0.01, two train/checkpoint views, one inference view, 8-epoch schedule, CRPS + MSE loss, model weights and 77 features are fixed. All four equal-update adapters completed eight epochs. Uniform and recency needed patience eight to avoid their original early stop; local and mixed completed eight epochs with the existing patience three. Their eight-epoch query-ID hashes match exactly across strategies. A separate four-epoch-cap attempt was stopped and excluded because TabPFN scales its learning-rate schedule by the configured epoch cap.

| Fine-tuning support, eight epochs | Full S0 inference RMSE | Mixed 10k S4 inference RMSE | Full S0 MAE | Train time |
| --- | ---: | ---: | ---: | ---: |
| S1 uniform | 459.627 | 460.786 | 332.773 | 958 s |
| S2 recency | 459.063 | 460.819 | 331.910 | 943 s |
| S3 local/property | **457.240** | 460.929 | 332.240 | 989 s |
| S4 mixture | 457.709 | **459.324** | 336.126 | 1,038 s |

The full-context S3 adapter is the best of the new equal-update fine-tuning runs, but is **S$2.028 worse** than the original DiagScale full-context result. In the 10,000-row mixed inference condition, S4 is best among the new equal-update adapters but is **S$2.935 worse** than the original adapter with that same S4 context. Thus the new fine-tuning support methods did not improve the deployment choice. Full-context inference of the new adapters took 552–584 seconds and peaked at 7.22 GiB PyTorch allocation. Fine-tuning peaked at 11.83 GiB on the 15.99 GiB RTX 4080. Total retrieval overhead over 800 training episodes was about 11 s (S1), 12 s (S2), 56 s (S3) and 105 s (S4).

With full inference context, paired bootstrap over the ten validation months gives S3–S1 median −S$2.40 [−3.47, −1.32] and S4–S1 −S$1.96 [−3.18, −0.51]. With mixed 10k inference, S4–S1 is −S$1.47 [−2.36, −0.47]. These differences describe the new month-coherent training setup only; the original S0 adapter used different query episodes, so its gap cannot be attributed solely to support selection. There is only one training seed, and this validation period has been consulted for many model choices.

The original early-stopping policy also produced four-epoch uniform/recency adapters. With S4 10k inference they scored 456.628 and **456.004** RMSE, respectively; with full S0 inference they scored 456.236 and 456.256. The S$0.625 recency gain at 10k has a month-bootstrap interval [−0.95, −0.30], but disappears with full context (difference +S$0.020; interval [−0.46, 0.49]). Longer training changed their scores substantially, including a deterioration of the eight-epoch uniform adapter under one-view deployment. The checkpoint criterion uses two views and does not guarantee the best one-view outer result. The earlier calibration ranking also failed to predict the later S3 result. These interactions and the one-seed design are reasons to avoid a new submission based on a small sampled-context gain.

## Verification and recommendation

Nine sampler/adapter unit tests pass, including historical cutoffs, duplicate exclusion, deterministic selection, fallback refill, unchanged model inputs, identical supervised queries and zero-initialized DiagScale layer identity. A separate full-model check with 2,048 context rows and 48 queries found **0.0 maximum absolute prediction difference** between frozen TabPFN and a zero-update DiagScale model. Each saved outer prediction file has 28,995 unique IDs, no missing predictions and recomputed RMSE matching its score. Model feature hashes and adapter hashes are recorded in run settings and training files. No outer or hidden-test target enters retrieval or fine-tuning.

**Keep the original full-context DiagScale setup** for RMSE-focused deployment. If memory or latency requires a 10,000-row context, reuse the original adapter with S4 retrieval: it incurs S$1.178 higher RMSE in this development period while reducing model context to 10,000 rows and inference allocation to about 3.62 GiB. The new adapters are research results rather than an improvement to the final submission. No further mixture-ratio or recency-decay tuning was launched because neither S3 nor S4 beat S0 on the primary metric; tuning those values on this already examined period would increase selection bias.

## Artifacts and reproduction

- `phase_a_outer_budget10000_queriesfull_seed42/`: fixed-adapter settings, context IDs, retrieval audits, 28,995-row predictions, scores, month/town diagnostics and context age/spatial coverage figures.
- `phase_b_S*_seed42_episodes100_epochs8*/`: strategy-specific training settings, query/support episode hashes, checkpoint validation trace, adapter hash, calibration result, outer scores and row-level predictions. Adapter weights live under `.cache/tabpfn_context_sampling/`.
- `combined_outer_scores.csv`, `combined_group_scores.csv`, `combined_outer_rmse.png`, `combined_monthly_rmse.png`, and `combined_town_rmse_difference.png`: verified combined comparisons.
- `phase_b_month_cluster_bootstrap_inferS0_8epochs.json` and `phase_b_month_cluster_bootstrap_inferS4_8epochs.json`: paired month-resampling summaries.
- `zero_adapter_identity.json`: full-model identity verification.

From the repository root with its CUDA-enabled virtual environment:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_tabpfn_context_sampler tests.test_diagscale_tabpfn
.venv\Scripts\python.exe analysis/compare_tabpfn_context_sampling.py --stage calibration --context-budget 10000
.venv\Scripts\python.exe analysis/compare_tabpfn_context_sampling.py --stage outer --context-budget 10000
foreach ($strategy in @('S1','S2','S3','S4')) {
    $patience = if ($strategy -in @('S1','S2')) { 8 } else { 3 }
    .venv\Scripts\python.exe analysis/compare_tabpfn_sampled_finetuning.py --strategy $strategy --epochs 8 --episodes 100 --patience $patience --outer --inference-strategy S4
    .venv\Scripts\python.exe analysis/compare_tabpfn_sampled_finetuning.py --strategy $strategy --epochs 8 --episodes 100 --patience $patience --outer --inference-strategy S0
}
.venv\Scripts\python.exe analysis/check_tabpfn_context_zero_adapter.py
.venv\Scripts\python.exe analysis/summarize_tabpfn_context_sampling.py
```

The sampled fine-tuning runner accepts `--strategy S1|S2|S3|S4`, `--patience`, `--outer`, `--inference-strategy S0|S1|S2|S3|S4`, and `--smoke`. Rerunning a completed checkpoint with `--outer` scores a new inference context without retraining. The original runners remain unchanged.
