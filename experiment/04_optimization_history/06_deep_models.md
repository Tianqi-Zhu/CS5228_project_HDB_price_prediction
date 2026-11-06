[Chinese version](06_deep_models.zh-CN.md)

# Round 6:measured stronger deep tabular models

Four families, 19 early fits(15 settings+four seeds),39 fits including later windows. Test numerical embeddings, parameter-sharing ensembles, optimized MLPs and learned retrieval using train/static-aux histories, without public rent targets or validation updates.

## Models and implementations

- [Official TabM](https://github.com/yandex-research/tabm):`tabm==0.0.3`,32 shared-parameter members, 3×256. Compare ordinary/piecewise-linear inputs, raw/residual targets, two learning rates. Train member MSE separately; average at inference.
- [Author numerical embeddings](https://github.com/yandex-research/rtdl-num-embeddings) + [RTDL MLP](https://github.com/yandex-research/rtdl-revisiting-models):MLP-PLR with periodic numerical/category embeddings and 3×256 MLP.
- [Author standalone RealMLP PyTorch](https://github.com/dholzmueller/pytabkit/tree/main/pytabkit/models/realmlp),`pytabkit==1.8.0`:native missing handling, robust scaling, smooth clipping, PBLD embeddings, parameter learning-rate factors/schedules; one/four members. Experimental standalone variant differs from Lightning in initialization/weight decay; no bitwise-equivalence claim.
- [Author ModernNCA](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py):fixed Git commit, unchanged source with custom training adapter. Learn numeric/category representations and neighbor weights from fitting rents. Compare ordinary/periodic embeddings/author defaults. Training pools4096/8192 exclude current batches; inference uses all fitting records with cached chunked embeddings.

All learn from random initialization without new rental data/pretrained targets. Prior TabPFN/TabICL remains separate evidence. Dependencies in `.cache/deep-deps`; existing Conda unchanged.

## Inputs and temporal protocol

Candidate inputs: 62 strong-history fields plus five past-market/cohort-shrinkage fields. Drop constant/all-missing fields using fitting data only, so actual widths vary by origin. Fit vocabularies, numerical transforms, PLE bins and RealMLP robust statistics only on matching fitting data. Restore residual baselines from earlier rents once.

Earlier17,977 rows(2023-10–2024-03) select settings, epochs and weights. Annual35,158 April2024–March2025 and recent 28,995 June2024–March2025 refit all pre-origin labels for fixed epochs; no later stopping. Retrieval may access fitting labels, never validation targets or a training query's own label.

TabM/MLP-PLR cap60 epochs; RealMLP128, retaining its128-epoch schedule for fixed-duration refits; ModernNCA50. Patience/context budgets:`plan.json`. Bounded measurements, not exhaustive tuning.

## Matched-window comparisons

Representatives selected earlier only; prespecified seeds 42/43/44 all enter means, without choosing later best seeds.

| Route | Annual RMSE | Recent RMSE | Earlier-selected configuration |
|---|---:|---:|---|
| Official TabM | 456.1612 | 461.0924 | `seedmean__tabm_ple_residual_lr0.0005` |
| Authors' standalone RealMLP | 458.0752 | 448.4527 | `seedmean__realmlp_residual_ens4` |
| MLP-PLR | 475.9378 | 448.0631 | `mlp_plr_residual` |
| Authors' ModernNCA | 461.3778 | 463.9001 | `modernnca_residual_plr` |

Prior round 3 strong-history FT annual/recent 471.5338/458.6012 is row-verified. It also uses 67 candidate inputs/residuals, but differs in preprocessing, network and duration; this compares training pipelines, not architecture-only changes.

### Every later-window refit

Retain all later runs, even those beating earlier representatives. Recent-score ordering documents results without reselecting blend members/weights.

| Configuration | Annual RMSE | Recent RMSE |
|---|---:|---:|
| `realmlp_residual_ens4_seed43` | 456.6087 | 447.8440 |
| `mlp_plr_residual` | 475.9378 | 448.0631 |
| `seedmean__realmlp_residual_ens4` | 458.0752 | 448.4527 |
| `realmlp_residual_ens4` | 459.8459 | 449.5217 |
| `realmlp_residual_ens4_seed44` | 459.9596 | 450.6923 |
| `tabm_ple_residual_lr0.0005` | 456.4600 | 460.4710 |
| `tabm_ple_residual_lr0.0005_seed43` | 457.4200 | 460.4771 |
| `seedmean__tabm_ple_residual_lr0.0005` | 456.1612 | 461.0924 |
| `modernnca_residual_plr` | 461.3778 | 463.9001 |
| `tabm_ple_residual_lr0.0005_seed44` | 456.4843 | 463.9748 |
| `tabm_ple_residual_lr0.002` | 453.6567 | 465.1498 |
| `tabm_ple_raw_lr0.002` | 455.2046 | 486.1951 |

### Earlier numerical-representation controls

TabM compares ordinary/PLE inputs with matched candidates, target and learning rate. Independent stopping and additional parameters make this a recipe comparison. Ordinary-input variants were not all rerun later; cross-window gains are unverified.

| Target / learning rate | Ordinary-input RMSE | PLE-input RMSE | Difference |
|---|---:|---:|---:|
| raw / 0.002 | 527.3965 | 523.9942 | -3.4023 |
| residual / 0.002 | 526.6436 | 523.9029 | -2.7406 |
| residual / 0.0005 | 526.8906 | 523.3821 | -3.5085 |

## Comparison with the original best blend

| Window | Original RMSE | New deep-model blend RMSE | Difference | New MAE | Months improved |
|---|---:|---:|---:|---:|---:|
| outer | 451.1810 | 450.4924 | -0.6886 | 329.9547 | 12/12 |
| june | 442.9025 | 444.6888 | +1.7863 | 328.8985 | 3/10 |

Because recent RMSE does not improve, retain history blend442.9025 and existing submission CSVs. Report the annual gain separately rather than claiming universal improvement.


Three-month block difference intervals describe development errors only; they do not correct repeated search/selection bias:

| Window | Descriptive95% new-minus-old RMSE interval |
|---|---:|
| outer | [-0.9041, -0.5015] |
| june | [-0.1952, +3.4086] |

Three-seed results (per-run RMSE SD differs from averaged-prediction RMSE):

| Window | Configuration | Mean single-run RMSE ± SD | RMSE of averaged three-seed predictions |
|---|---|---:|---:|
| screen | `tabm_ple_residual_lr0.0005` | 523.5183 ± 0.2076 | 523.1748 |
| outer | `tabm_ple_residual_lr0.0005` | 456.7881 ± 0.5473 | 456.1612 |
| june | `tabm_ple_residual_lr0.0005` | 461.6410 ± 2.0211 | 461.0924 |
| screen | `realmlp_residual_ens4` | 524.0615 ± 0.4386 | 523.3840 |
| outer | `realmlp_residual_ens4` | 458.8047 ± 1.9026 | 458.0752 |
| june | `realmlp_residual_ens4` | 449.3527 ± 1.4316 | 448.4527 |

Earlier weights are nonnegative, sum1, cap each new member0.5, and shrink toward the old blend by 0.001. No later reweighting. Report standalone/blend gains separately; different inputs/targets prevent architecture-only causal attribution.

Pool includes seed 42 and three-seed means, so effective seed weights may differ, all selected earlier. `effective_blend_weights.json` expands repeated members; the whole blend is not necessarily an equal three-seed average.

Target 430 unmet. Later windows were already examined; development scores remain. Seed SD describes initialization, not future markets. Overlapping windows do not add independent samples.

## Verification and files

Check45 files'IDs, months, labels, counts/metrics; 20 later models reload consistently; seed means/blend reconstruct. Six tests cover author interfaces, fit-only statistics, unknown categories, independent member losses, query exclusion and cached retrieval. Data/author-source hashes recorded.

- `selected_model_comparison.csv`:annual/recent architecture controls.
- `verified_metrics.csv`:all configurations, epochs, time and peak PyTorch allocation (not physical resident VRAM).
- `seed_variation.csv`:seed variability/equal averaging.
- `ensemble_selection.json`:earlier-locked weights/representatives.
- `author_source.json` / `verification.json`:source provenance, versions/full checks.

## Reproduction

Requires existing CUDA PyTorch/RTDL in `.cache/model-deps`. Install new author packages separately without replacing environments. Downloader pins this round's commits/hashes; see `author_source.json`.

```powershell
python -m pip install --no-deps --target .cache/deep-deps -r requirements/optional_deep_models.txt
python -m analysis.experiments.sixth_wave_sources
python -m analysis.experiments.sixth_wave_deep screen
python -m analysis.experiments.sixth_wave_ensemble select
python -m analysis.experiments.sixth_wave_ensemble outer
python -m analysis.experiments.sixth_wave_ensemble june
python -m unittest tests.test_sixth_wave
python -m analysis.experiments.sixth_wave_report
```

Keep rounds2–4 row predictions needed by blends. Completed results skip by default; later evaluation uses earlier saved epochs/weights only.
