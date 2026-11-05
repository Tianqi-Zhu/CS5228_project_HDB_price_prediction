# Spatial and temporal transformer experiment

Implement and run randomly initialized models using the existing causal 77-feature schema. The primary question is whether a small rent+COE sequence encoder improves individual-rental SGD RMSE beyond an identical rent-only encoder. This plan is recorded before new model scoring. Existing Kaggle submissions are not overwritten.

## Fixed design

- Property FT-Transformer: two pre-norm blocks, dimension 128, four heads, dropout 0.1. Numeric missingness is encoded within tokens. Original categorical identities include block/street; unknown categories use bucket zero. A separate spatial token uses centered coordinates and Fourier wavelengths 0.25/0.5/1/2/4/8/16 km.
- Twelve monthly history tokens, with six-month matched sensitivity. One temporal transformer block, dimension 32, two heads, FFN 64; additive gated contribution initialized small. Economic response is shared rather than a company/location stock interaction.
- Sequence includes property-adjusted national/type/shrunken area-type monthly relative rents and changes, counts and observation age. A fixed 2021-only property reference is unavailable before December 2021. No future interpolation is permitted.
- COE A/C log premiums and monthly log changes. Secondary stocks: supplied .SI median monthly log-return proxy, not STI. No company/worksite stock features, sector search, or raw cross-company price levels.
- Every target-derived row input, baseline and sequence uses the same row information cutoff. COE/stocks have one additional calendar-month publication buffer. Sensitivity shifts the full rental information cutoff one more month and retains that macro buffer.
- Deterministic training origins cover horizons 1–16 where history permits. Source rows/labels are unchanged. The inherited local baseline plateaus beyond its validated horizon ten; actual horizon and observation age do not. This is explicitly a proposed extension, not validation of an extrapolation rule.
- Individual residual SGD MSE, standardized on fitting rows only. AdamW lr 0.0003, weight decay 0.001, batch 256, gradient clip 1, BF16 on CUDA, max 30 epochs and patience five. Sequence sigmoid gate has penalty 0.01 times its square. Per-run checkpoints include optimizer and shuffle/dropout states and verify exact prediction replay.

## Comparisons and selection

T0: property transformer and existing history summaries. T1: add rent sequence. T2: add COE. T3: add stock proxy. Primary incremental comparison is T2 minus T1, matched by seed; T1 minus T0 measures sequence value.

C1/C2 replace the temporal transformer with an eight-unit LSTM for rent/rent+COE. C3/C4 use a regularized linear lag head trained jointly with the same property backbone; they are **not** closed-form Ridge fits. S1/S2 use six-month windows; S3/S4 add one full month of delay. Architecture controls and sensitivities initially use seed 42; primary variants use fixed seeds 42/43/44 without seed selection or outer-optimized blending.

Earlier disjoint query windows September 2022–June 2023 and July 2023–April 2024 use the existing label-independent 200-per-month samples and population weights. All fitting rows are retained. Each variant's rounded median best epoch selects its fixed full refit length. Query months in these two windows are disjoint; their fitting histories expand normally. No June 2024–March 2025 label selects a setting or epoch.

Main development evaluation: all 28,995 June 2024–March 2025 rows, fitted on all 121,005 preceding rows. Separate 16-month stress window: December 2023–March 2025; its epoch comes only from the first earlier window, whose labels end June 2023. The stress window overlaps the main development window and is not independent confirmation.

Report individual-rental RMSE/MAE/bias, month scores, horizons 1–3/4–6/7–10/11–16, seeds, parameter count, wall time and peak allocated VRAM. Paired contiguous three-month block intervals are descriptive, conditional on fitted models, and omit fitting/search uncertainty. Only 51 distinct labeled months are available, with serial dependence. Require consistent incremental gains across origins/seeds and assess delay sensitivity before recommending auxiliary inputs. No automatic submission replacement.

## Capacity and operation

The worker is sequential and owns an exclusive experiment lock. Before every CUDA fit it polls `nvidia-smi`, requiring at least **8 GiB free** and utilization at most **35%** for two consecutive samples, 15 seconds apart. Telemetry errors keep it waiting. It never terminates another GPU process or bypasses the gate. Parameters may be made more conservative on the CLI. CPU preparation and tests do not initialize CUDA. GPU availability is not a reservation; other applications can still allocate memory after the check.

```powershell
.venv/Scripts/python.exe -m unittest tests.test_spatial_temporal_transformer
.venv/Scripts/python.exe analysis/run_spatial_temporal_transformer.py --stage prepare
.venv/Scripts/python.exe analysis/run_spatial_temporal_transformer.py --stage smoke --device cpu
.venv/Scripts/python.exe analysis/run_spatial_temporal_transformer.py --stage smoke --device cuda
.venv/Scripts/python.exe -u analysis/run_spatial_temporal_transformer.py --stage all
```

To run CPU smoke, gated CUDA smoke and the complete matrix sequentially in a hidden background worker:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File analysis/start_transformer_experiments.ps1
Get-Content reports/spatial_temporal_transformer/worker.stdout.log -Tail 20
```

`launcher.json` identifies the background wrapper PID; `worker.lock` identifies the current Python worker PID. `gpu_status.json` records the latest capacity check and `worker_status.json` records the active fit/epoch. Errors go to `worker.stderr.log`. GPU work starts only after CPU smoke succeeds; the full matrix starts only after CUDA smoke succeeds.

Rerun the identical command to resume. Source/code/settings hashes must agree; changed experiments need new `--output` and `--cache` directories. A terminated process can leave `worker.lock`; verify its recorded PID has exited before removing that exact file. Reports are in `reports/spatial_temporal_transformer/`, model artifacts in `.cache/spatial_temporal_transformer/`.

All experiments are fixed-origin forecasts. A monthly-updated production scenario requires a separately specified evaluation and is not mixed with this comparison.
