# Ranked transformer experiment series

## Protocol and current decision

The original scratch transformer T0 has three-seed mean RMSE **S$461.164** on the repeatedly used June 2024–March 2025 development period (28,995 transactions). The original rent-sequence T1, rent+COE T2, and rent+COE+stock T3 means are S$465.309, S$466.777, and S$465.086. The separately trained TabPFN ABCDE equal ensemble is S$454.704. These are development comparisons, not an untouched final test.

New models use the original prepared chronological folds. Two earlier 2,000-case, population-weighted forecast windows (September 2022–June 2023 and July 2023–April 2024) select settings or stop a candidate. The later development window supplies evaluation only, usually at seeds 42, 43, and 44. Some posthoc studies use one earlier window to fit and the other to select. Original submissions, source checkpoints, and prepared folds were not replaced. New GPU runs wait for at least 8 GiB free and at most 35% utilization in two samples 15 seconds apart. The reports and checkpoints under this directory and `.cache/transformer_ranked_experiments` retain the run artifacts.

The most useful **development-window** finding is a fixed horizon-decayed blend of T0 and T1, rank 23. It improves both earlier windows and all three later seeds. It remains S$5.5 above TabPFN and slightly worsens the separate 16-month diagnostic. Training that decay inside T1 did not transfer on the earlier windows. Keep the existing submission unchanged pending a genuinely new temporal backtest.

## Sequential results

The deltas below are mean later RMSE minus matched T0 RMSE, in SGD; negative is better. `Gate stopped` means no later labels were scored for that candidate. Reused prior studies are identified explicitly. Frozen-backbone or posthoc tests are not claimed to be full end-to-end architecture tests.

| Rank | Test | Later result / decision |
| ---: | --- | --- |
| 1 | Independent versus property-conditioned rent-sequence correction, old versus stable initialization | Conditioned old mean Δ−0.177; conditioned stable Δ−0.092. Each improves only 2/3 seeds. No robust promotion. [Details](01_conditioned_correction/) |
| 2 | Hierarchical geographic calibration | Earlier-selected global shift Δ−0.504, +3.803, +5.485 across seeds. Unstable. [Details](02_hierarchical_calibration/) |
| 3 | Weighted local comparable control | Earlier selection chose no correction; all nine raw/premium Ridge settings worsened earlier transfer. [Details](03_comparable_control/) |
| 4 | Learned attention to causal comparable transactions | Mean Δ+0.759; one seed improved, two worsened. [Details](04_context_attention/) |
| 5 | Rent innovations and reliability | Mean Δ+0.903; no consistent gain. [Details](05_rent_innovations/) |
| 6 | Horizon-weighted supervised loss | Balanced weight selected earlier but mean Δ+5.524 later; full 16-month diagnostic RMSE S$494.18 versus T0 S$492.41. [Details](06_horizon_weighting/) |
| 7 | TabPFN forecast plus scratch-transformer correction | Earlier-selected Ridge worsened TabPFN by about S$7.1 on every later seed. [Details](07_tabpfn_correction/) |
| 8 | Lightweight TabPFN-to-T0 teacher distillation | Teacher-free student selected no change. Only 4,000 historical teacher forecasts exist versus 121,005 later-origin fit cases; this does not test full transformer distillation. [Details](08_tabpfn_distillation/) |
| 9 | Residual-quantile uncertainty calibration | Global quantiles selected earlier; nominal 80/90% intervals cover about 91/95% later, hence conservative. No point-RMSE claim. [Details](09_uncertainty_calibration/) |
| 10 | Learned town, planning-area, and block embeddings on frozen T0 | Mean Δ+0.558; one seed worsened by S$2.56. [Details](10_location_embeddings/) |
| 11 | Static and dynamic AR(1) latent residual factors | Earlier selection preferred static half-strength, not dynamic; later Δ−0.868, +1.732, +2.432. [Details](11_dynamic_latents/) |
| 12 | Town/month mean-residual training penalty | Stronger penalty selected earlier; later Δ−2.068, −0.114, +3.556, mean +0.458. [Details](12_group_calibration/) |
| 13 | Heteroscedastic Gaussian NLL | Earlier gains, later mean Δ+1.677. Learned scale correlates about 0.19–0.20 with absolute error; Gaussian nominal 80/90% bands cover 86–87/92–93%. [Details](13_gaussian_nll/) |
| 14 | Masked monthly-sequence pretraining before T1 fitting | Improved T1 by S$4.45 on the two earlier windows on average, but remained S$2.42 worse than T0; gate stopped. [Details](14_sequence_pretraining/) |
| 15 | Horizon-dependent T0/T1/T2/T3 ensemble | Every learned blend lost on the second earlier window; T0 alone selected. [Details](15_context_ensemble/) |
| 16 | Synthetic rental pretraining | **Not run.** No validated, causal rental simulator exists. A low-fidelity synthetic-label run would not test the proposed prior. |
| 17 | COE branch | Previously run T2: three-seed mean S$466.777 versus T0 S$461.164. [Original study](../spatial_temporal_transformer/) |
| 18 | Huber loss | Earlier-selected threshold 0.5; later mean Δ+11.896 and strong positive bias. [Details](18_huber_loss/) |
| 19 | Conditional VAE latent | **Not run.** A meaningful identifiable latent target and evaluation criterion remain unspecified; the scalar rental residual alone does not establish one. |
| 20 | Foundation-model-informed latent prior | **Not run as a latent-prior model.** Ranks 7–8 test direct TabPFN guidance. Broader chronological teacher targets or compatible embeddings are needed for a full prior. |
| 21 | Stocks and worksites | Previously run T3 mean S$465.086 versus T0 S$461.164; separate worksite stock study found no site passing both association and forecast checks. [Original model](../spatial_temporal_transformer/), [worksite study](../company_worksite_stocks/report.md) |
| 22 | Wider (192) or deeper (3-layer) transformer | Wider failed earlier gate. Deeper advanced but later mean Δ+8.477, all seeds worse. [Details](22_capacity_ablation/) |
| 23 | Fixed exponential decay of T1−T0 forecast difference | Chosen on earlier 2023: alpha 0.5, time constant 3 months. Later Δ−0.443, −0.803, −1.621; mean −0.956. [Details](23_forced_decay/) |

### Rank 23 audit

The selected rank-23 blend is `T0 + 0.5 exp(-(h−1)/3) (T1−T0)`, where `h` is the forecast horizon. Earlier-window RMSE changes were −S$0.879 and −S$0.698. Later seed RMSE values are S$461.893, S$458.612, and S$460.119 versus T0 S$462.336, S$459.415, and S$461.740. Month-level changes are negative in all 10 months for seeds 42 and 44 and in 9/10 months for seed 43. Three-month moving-block bootstrap intervals for the later RMSE delta are [−0.689, −0.248], [−1.430, −0.229], and [−2.739, −0.559]. These intervals condition on the selected setting and models and omit selection/training uncertainty; the later period has been repeatedly reused in this project.

On the separate 16-month diagnostic, full RMSE changes from S$492.408 to S$492.554 (+0.146); horizons 11–16 change from S$460.286 to S$460.207. This is a forecast-level proxy: T0 and T1 are independently trained, so their difference is not the exact T1 temporal branch. A direct T1 architecture with an internal 3-month decay gate was also tested; it improved the first earlier window but lost on the second, so it did not advance to later evaluation. [Internal-gate details](23b_internal_decay/).

## Next evidence needed

1. Check the fixed rank-23 recipe on a newly held-out set of later months or a genuinely untouched geographic/time split. Do not tune alpha or time constant on that set.
2. For rank 16, specify and validate a causal rental simulator against historical feature and target distributions before synthetic pretraining.
3. For ranks 19–20, define a measurable latent task and generate leakage-safe foundation-model guidance for enough training cases. Avoid interpreting an arbitrary latent coordinate as an economic factor.

The original 150k-row submission CSVs remain unchanged. No model from this series has been promoted to them.
