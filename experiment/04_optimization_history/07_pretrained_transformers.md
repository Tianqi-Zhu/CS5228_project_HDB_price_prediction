[Chinese version](07_pretrained_transformers.zh-CN.md)

# Round 7:newer tabular Transformers

Fifteen complete earlier configurations, 25 full evaluations including later windows: 14 frozen-context and one early supervised TabDPT adaptation. Frozen inference calls are not training-from-scratch counts.

## Routes and versions

- [TabDPT1.3 / post-Turbo](https://github.com/layer6ai-labs/TabDPT-inference):author September2026 v1.3 weights,`tabdpt==1.3.2`. Compare random whole-pool context and per-query neighbors. Authors report real-world tabular pretraining; no extra HDB labels downloaded here.
- Custom supervised TabDPT trainer updates last two blocks, label encoder and regression head; others/kappa frozen. Not the author training recipe. Each fitting row is a query once per epoch, excluding its own label from 512-row context. BF16, MSE, AdamW0.00003, max8 epochs, patience3; matched epoch0 control. Later refits use earlier fixed epochs without later stopping.
- [Nori1.5](https://github.com/Synthefy/synthefy-nori):`synthefy-nori==0.22.0`,about 20.8M parameters, two-stage encoding/feature/sample attention; Nori-6M is a size control. Authors report synthetic pretraining.
- [TabICLv2](https://github.com/soda-inria/tabicl):existing February2026 regression weights,`tabicl==2.1.1`; new 62+five strong-history candidates/baseline differ from round 2's smaller inputs.
- [TabPFN-3.5](https://github.com/PriorLabs/TabPFN):requires user login/license acceptance. User chose available public weights first, so this version is untested here. Prior TabPFN scores retain their own versions/inputs.

All inference is local using downloaded author weights; no project data sent to model services. Independently screen raw, residual and ratio targets; restore historical baselines by addition/multiplication. Ratio denominators never use actual validation-month mean rents.

## Temporal, feature and hardware budgets

Earlier2023-10–2024-03:17,977 rows; annual 2024-04–2025-03:35,158; recent 2024-06–2025-03:28,995. Histories freeze per origin and retain all rows. Lock later settings, contexts, views, weights earlier. All windows are reused development; last two overlap.

Candidate62 histories/static + five past-market/shrinkage fields; drop missing/constants only from fitting context. TabDPT vocabularies, imputation/scaling fit source pools. Nori uses native handling with HighDimFeatureSelector fit_on_test disabled. TabICL uses native pipelines.

Each route adds a mature-context control: source rows at least24 months into history with positive24-month block support. Selection ignores rent values/retains all queries. Changed available pools/columns mean this is not architecture-only ablation.

RTX4060 Laptop8GB: TabDPT uses public use_flash=False because Windows lacks the Flash-only path; torch.compile disabled. Nori disables concurrent pipelines and fixes native chunk-element budget, without quantizing caches or silently dropping contexts. Initial full-query OOM runs are excluded; recompute with fixed chunks.

Nori: 2048 context/view; 1.5 defaults four pipelines, 6M capped four (default eight). TabDPT: two views, 2048 random context or128 neighbors/query; BF16 retrieval, FP32 random. All fitting rows remain in its source pool. TabICL: 24,000 random context, two views (default eight). Unequal budgets prohibit universal architecture claims. Sampling excludes query rents. Failed/time-stopped runs:`runtime_adjustments.json`.

## Later matched-window comparisons

| Route | Annual RMSE | Recent RMSE | Earlier-selected configuration |
|---|---:|---:|---|
| Frozen TabDPT1.3 | 462.9712 | 476.9916 | `tabdpt13_residual_global2048` |
| Supervised-adapted TabDPT1.3 | 464.1707 | 463.0065 | `tabdpt13_adapt_residual` |
| Nori | 459.5105 | 462.0088 | `nori15_raw_2048` |
| TabICLv2 | 460.6730 | 454.6275 | `tabicl2_raw_mature24000` |
| FT-Transformer (previous67-input policy) | 471.5338 | 458.6012 | — |
| Original history blend | 451.1810 | 442.9025 | — |
| Round 6 deep blend | 450.4924 | 444.6888 | — |
| Blend with new Transformers | 450.4823 | 442.9704 | — |

Full earlier-window scores, including negative results; 128-query smoke runs do not select candidates:

| Configuration | RMSE | MAE | Bias | Context/views | Seconds |
|---|---:|---:|---:|---:|---:|
| `tabdpt13_adapt_residual` | 530.2387 | 405.4116 | +21.9536 | 2048/2 | 223.0 |
| `tabdpt13_residual_global2048` | 533.0680 | 404.0540 | +50.7355 | 2048/2 | 5.1 |
| `tabicl2_raw_mature24000` | 534.0392 | 413.8928 | -16.2024 | 19863/2 | 14.3 |
| `tabdpt13_ratio_global2048` | 535.4776 | 404.3408 | +66.3935 | 2048/2 | 6.0 |
| `nori15_raw_2048` | 536.1248 | 407.6316 | +50.1221 | 2048/4 | 110.5 |
| `nori6m_residual_2048` | 536.7761 | 409.5743 | +45.6094 | 2048/4 | 34.9 |
| `nori15_ratio_2048` | 537.9307 | 408.5619 | +55.8705 | 2048/4 | 111.1 |
| `nori15_residual_2048` | 537.9709 | 409.5674 | +53.5413 | 2048/4 | 105.8 |
| `tabdpt13_raw_mature2048` | 537.9726 | 414.2037 | -4.7545 | 2048/2 | 7.1 |
| `tabdpt13_raw_global2048` | 539.0595 | 405.2803 | +73.1753 | 2048/2 | 5.7 |
| `nori15_raw_mature2048` | 539.5432 | 418.1984 | -5.6991 | 2048/4 | 121.3 |
| `tabicl2_raw_24000` | 549.7987 | 410.7461 | +118.9454 | 24000/2 | 13.4 |
| `tabdpt13_residual_retrieval128_bf16` | 552.3509 | 419.0147 | +92.4541 | 128/2 | 187.3 |
| `tabicl2_residual_24000` | 586.8685 | 433.0859 | +229.8313 | 24000/2 | 13.4 |
| `tabicl2_ratio_24000` | 592.3055 | 439.0568 | +195.5518 | 24000/2 | 14.2 |

### Matched supervised-adaptation control

Match original weights, inputs, BF16, two views and 2048 inference context; compare frozen epoch0 with earlier-selected adaptation. Preserve backbone inference-mode context normalization while differentiating designated parameters.

| Window | Frozen RMSE | Adapted RMSE | Selected epochs | Supervised rows per epoch |
|---|---:|---:|---:|---:|
| june | 475.3917 | 463.0065 | 1 | 121005 |
| outer | 463.2935 | 464.1707 | 1 | 114842 |
| screen | 533.1687 | 530.2387 | 1 | 96865 |

## Why these deep models do not consistently beat the reference

Separate observations from explanations below. Results do not establish useless capacity or attribute all gaps to noise.

### Historical-feature availability changes

| Recent-window cohort history | Available share in fitting source pool | Available query share | Available share in Nori random2048 context |
|---|---:|---:|---:|
| 12 months | 44.8% | 62.6% | 44.1% |
| 24 months | 31.9% | 71.3% | 31.6% |
| 36 months | 8.5% | 63.2% | 8.5% |

Fitting cold-start/missingness differs from later queries. Neural routes need a controlled check of this shift. Mature context improves TabICL early raw-target 549.7987→534.0392, but not Nori, so it is not the sole explanation.

### Some errors are broad underprediction

| Recent model | Actual RMSE | Mean bias | Centered error SD (decomposition only) |
|---|---:|---:|---:|
| history_ensemble | 442.9025 | -20.6662 | 442.4201 |
| old_ft_transformer | 458.6012 | -94.1569 | 448.8313 |
| tabm | 461.0924 | -116.2196 | 446.2053 |
| realmlp | 448.4527 | -67.2323 | 443.3843 |
| modernnca | 463.9001 | -69.5800 | 458.6523 |
| mlp_plr | 448.0631 | -47.7676 | 445.5096 |

Last column uses true validation labels to decompose MSE=variance+bias²,**not a valid calibrated prediction score**. TabM−116/RealMLP−67 biases motivate temporal calibration checks, not correction using validation means.

### Context, adaptation and reference choice

Frozen Nori uses 2048 source rows/view, unlike trees fitting over 100,000. TabDPT adaptation trains on every fitting row, but early loss0.8269→0.8149 coincides with temporal validation worsening after epoch1. More epochs do not consistently fix transfer. Ablate representation, drift and pretraining/task mismatch rather than blaming release date/parameter count.

Reference442.90 combines histories/ten models. First compare same-information standalones, then blend complementarity. Unequal support, contexts and preprocessing remain limitations.


## Blending and limits

Candidates include history blend, round 6 blend and complete new earlier runs. Nonnegative, sum1, new-member cap0.4,0.001 shrinkage toward history. No later reweighting; see `ensemble_selection.json`.

Target 430 unmet; new recent 442.9704 versus history 442.9025. Retain history and existing submissions.

Check25 full files'IDs, months, labels, counts/metrics; new weights match official LFS hashes, existing TabICL matches download manifest, blend reconstructs; seven tests pass. Two later supervised checkpoints reload all rows; other frozen runs verify predictions/recipes without claiming every checkpoint reload.

- `verified_metrics.csv`:complete configurations/context/time/VRAM budgets.
- `selected_model_comparison.csv`:earlier-selected annual/recent controls.
- `checkpoint_manifest.json` / `verification.json`:immutable weight commits/hashes/checks.
- `monthly_comparison_*.csv`:monthly blend errors.

## Reproduction

Preserve rounds1–6 row predictions; install isolated dependencies in existing CUDA environment.

```powershell
python -m pip install --no-deps --target .cache/transformer-deps -r requirements/optional_pretrained_transformers.txt
python -m analysis.experiments.seventh_wave_sources
python -m analysis.experiments.seventh_wave_transformers screen
python -m analysis.experiments.seventh_wave_ensemble select
python -m analysis.experiments.seventh_wave_ensemble outer
python -m analysis.experiments.seventh_wave_ensemble june
python -m unittest tests.test_seventh_wave
python -m analysis.experiments.seventh_wave_report
```
