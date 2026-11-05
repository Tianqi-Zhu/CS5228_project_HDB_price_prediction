# feature_transformer_v4 — STATUS

Last updated: 2026-10-25.

## Task

Task 3 of `docs/plans/2026-10-25-feature-transformer-training.md`: a Transformer that keeps
every numeric feature as its own token, plus a bounded inner-only hyperparameter search.

I own only `experiments/feature_transformer_v4/` and `reports/feature_transformer_v4/`.
Every other package is imported read-only.

## Where it stands

| item | state |
|---|---|
| `SEARCH_SPACE.json` | written, 8 trials predeclared |
| model / data / training | implemented, **frozen** while the search runs |
| real-data smoke + checkpoint reload | **passed** (`runs/v4-smoke-s42`) |
| trial 1 (full A) | **done** — inner RMSE 587.4427, epoch 23 of 33, 54,097 params |
| trials 2–8 | **running** under `orchestrate.py --resume` |
| B evaluated | never — nothing has loaded B |

Trial 1 finished under an older in-`main` loop that had three orchestration defects; it is
**not** retrained. Instead it was verified by `resume_attestation.json` in its own directory:

| check | result |
|---|---|
| source hashes (`models/data/training/search` + v2 scaler/contract) vs the run manifest | match |
| checkpoint description (56 numeric tokens, width 48) and config vs summary | match |
| saved `StageState` rebuilds 52,310 A/fit rows identical to a fresh fit | identical |
| saved inner predictions: 35,882 row ids, labels, horizons | identical |
| RMSE recomputed from the saved predictions | 587.44270429, equals the summary |

The original `summary.json` and `run_manifest.json` were not modified; the sidecar records
what was and was not verified (the training curve itself is taken from the summary).

Smoke gate (`runs/v4-smoke-s42`, 1,500 fit / 1,500 hold rows, 3 epochs):

| check | value |
|---|---|
| real rental-loss forward/backward/optimizer step | yes, 54,097 parameters |
| checkpoint reload max abs delta | **0.0** SGD |
| state round-trip identical | true |

The smoke RMSE (875.9 at 3 epochs on a 1,500-row slice) is a gate result, not a score.

## Architecture as built

- **56 numeric tokens**, one per column: 21 static spatial/property + 24 history + 11
  context (anchor excluded). `token_j = x_j * weight_j + bias_j + identity_j` with
  `weight_j, bias_j in R^48` — a **vector** per feature, not a scalar.
  The scalar version was a real bug: a constant offset across the width is annihilated by
  LayerNorm (Pre-LN plus final LayerNorm), and measurement showed perturbing all 56 inputs
  moved predictions by 7.45e-9. With vector weights the same probe moves them by 4.2e-3 and
  hidden representations by 9e-2 to 1.4e-1.
- **4 categorical tokens**, one vocabulary each, unknown = code 0. Code 0 in the incoming
  data is a legal value meaning "not recorded" and maps to the unknown slot; it never takes
  a positive index.
- **Three poolings**: `mean`, `cls` (learned prediction token), `horizon_query` (that token
  conditioned on the 11 horizon/calendar columns). Both token poolings pool position 0 —
  they differ only in conditioning, so the comparison isolates conditioning.
- **No rent-sequence branch**: the legacy 48-step recurrence has a confirmed leading-padding
  defect, so importing it would inherit a known fault. This is **not** free: the 24 history
  *summaries* are coarser than per-month history, so the omission is a real difference in
  information and every comparison against the old LSTM-fusion runs must say so.
- Attention key-padding mask and pooling mask are separate arguments; padding positions are
  excluded from the pooled mean.
- The query token has its own type id after the feature tokens (an earlier version reused
  `n_types - 1`, which collided with the last feature token).

Units and labels follow `neural_fusion_v2.preprocessing.ScalerState` rather than a local
reimplementation, because three things there are easy to get wrong and all three were wrong
in a first draft: `log1p` on the eight `hist_n__*` / `hist_months__*` count columns, the
anchor kept in SGD and divided by 1000 only in tensors, and categorical codes passed through
as indices. Each unit is asserted against the contract separately, because the identity
`target + anchor == actual` holds even under a consistent 1000x unit error.

## Two runs, two backends — not one comparison

| run | backend | purpose | state |
|---|---|---|---|
| `runs/v4-search-s42` | CPU (2 threads) | reference: trial 1 complete, trial 2 accepted as it landed | **queue stopped** |
| `runs/v4-search-mps-s42` | Apple MPS | the primary search, all eight candidates on one backend | **8/8 done**, top two evaluated on B |

CPU and MPS are not bitwise identical: the measured evaluation gap at batch 512 / 60 tokens
is **3.7e-8 kSGD (3.7e-5 SGD)**, and the MPS smoke's CPU reload reproduced the device
predictions to exactly that delta. A CPU number and an MPS number are therefore never
treated as the same setting and no ledger merges them.

`runs/v4-search-s42` (CPU): trial 1 is the attested legacy run (inner RMSE 587.442704296612,
epoch 23/33). Trial 2 was allowed to finish so its artefacts are complete; the remaining CPU
queue (trials 3-8) was then stopped, because the MPS run supersedes it and two batches were
contending for the same cores. The stop is recorded in
`runs/v4-search-s42/queue_stop.json` with the reason and an explicit note that nothing was
deleted.

`runs/v4-search-mps-s42` (MPS, primary): all eight candidates. The device is recorded in the
run manifest, in every trial summary and in every checkpoint. Checkpoints are written with
CPU tensors, so they reload on a CPU-only host without refitting.

### Inner results so far (MPS, seed 42, A/holdout only — no B read yet)

| trial | change | pooling | inner RMSE |
|---|---|---|---|
| 1 | base | mean | 586.5235 |
| 2 | base | cls | 581.4621 |
| 3 | base | horizon_query | 633.1862 |
| 4 | lr 3e-4 | cls | 622.7982 |
| 5 | d_model 96 | cls | 585.2680 |
| 6 | layers 3 | cls | **581.0164** |
| 7 | dropout .15, wd 1e-3 | cls | 585.0238 |
| 8 | batch 1024 | cls | 597.2145 |

Readings:

- `cls` beats `mean` by 5.06 and `horizon_query` by 51.7 at the same budget. The
  horizon-conditioned token is clearly *not* helping here — it is the worst of the three by a
  wide margin.
- Narrower learning rate (3e-4) loses 41.3 to the base; the 60-epoch budget cannot recover
  from the slower start.
- Extra width (+2 SGD) and extra depth (−0.45 vs trial 2) are both inside the run-to-run
  spread, so neither is a demonstrated gain yet.

### Registered top two, and the first outer results

Frozen from inner RMSE **before** B was read (`registered_candidates.json`,
`registered_before_loading_B: true`):

| rank | trial | change | pooling | inner RMSE | B/validation RMSE | MAE | bias |
|---|---|---|---|---|---|---|---|
| 1 | 6 | layers 3 | cls | **581.0164** | **468.9730** | 336.3942 | 43.0836 |
| 2 | 2 | base (L2) | cls | 581.4621 | 469.4544 | 336.7487 | 43.0539 |

These are the first outer numbers this package produces, on the complete 35,158-row
validation population, each evaluated once. For context against read-only references:
block LSTM-fusion probe 475.598687709, macro-both 475.400694003, per-feature fixed-7
474.491228, LightGBM spatial 477.2036.

**What this does not establish.** This validation split has been viewed many times by
several experiments. 468.97 is a lower number than the references above, but it is not
independent evidence of generalisation: the architecture was chosen on A/holdout, yet the
whole workspace has repeatedly looked at B/validation. Treat it as "this configuration is
worth a submission attempt", not as an unbiased estimate.

The gap between rank 1 and rank 2 is 0.48 SGD — the same order as the seed-to-seed spread
already measured elsewhere in this project (1.2 to 3.9 SGD on a comparable model), so the
ordering of the two is not meaningful. What *is* supported is that both land near 469,
well below the ~475 references.

### Checkpoint parity, judged on three separate questions

The earlier single "reload delta, ok=False" reading was misleading: its 1e-4 SGD threshold
sits below the normal MPS-vs-CPU divergence, so a healthy checkpoint was being reported as
broken. `parity.py` now answers three questions with tolerances declared in advance:

| question | trial 6 | trial 2 | tolerance |
|---|---|---|---|
| same backend: reload on MPS, forward on MPS | 0.0 SGD (bit-identical) | 0.0 SGD | 1e-5 kSGD |
| cross backend: same weights, CPU vs MPS | max 5.96e-4 SGD, rms 1.04e-4, **0/35,158 rows over tolerance** | max 3.58e-4, rms 1.35e-4, 0/35,158 | 1e-6 kSGD per row |
| metric effect: RMSE/MAE gap between backends | 8.2e-6 SGD | 1.1e-5 SGD | 1e-3 SGD |

Every checkpoint holds CPU tensors and reloads without fitting anything; the parity check
patches the fitting functions to raise, and `tests/test_reload_parity.py` asserts that.

**Measured speed, stated plainly:** the MPS smoke (2,000 rows, batch 512, 3 epochs) finished
in 7.6 s, and real trial epochs on full A/fit run at **~19 s per epoch** versus ~32-55 s on
CPU, i.e. roughly 1.7-2.9x *in this pipeline*. The isolated benchmark's 4.63x (batch 512,
60 tokens, one step) does **not** transfer as an end-to-end figure: a trial also spends time
on per-epoch evaluation over 35,882 holdout rows. No claim of a "4.6x faster search" is made.

## Search discipline

Seed 42, `orig`, A/fit train + A/holdout monitor only. Max 60 epochs, patience 10, epochs
1–4 recorded but ineligible for selection; a better ineligible early epoch is reported as
`ineligible_best` rather than used. The cosine `T_max` is always the trial's max-epoch
budget, in A and in the B refit, so both stages share one schedule.

`orchestrate.py` replaces the earlier in-`main` loop after three orchestration defects were
found:

1. It froze the pooling for trials 4–8 from whatever subset was requested, so
   `--trials 1` alone decided "best of trials 1–3" from one trial. The decision now refuses
   until all three have a completed summary.
2. An existing run directory could only be refused. `--resume` now reuses a completed trial
   after verifying its config, contract and data fingerprint agree; a mismatch raises
   instead of quietly retraining. The original manifest is untouched and each resumed
   invocation writes `resume_manifest_<ts>.json`.
3. `freeze_top_two` returned dicts keyed `id` while callers read `trial_id`, crashing the log
   line after a successful freeze. One candidate schema is now defined once and asserted on
   registration.

## Tests

`env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest experiments/feature_transformer_v4/tests -q`
→ **42 passed**.

`test_numeric_flow.py` asserts on the full-model prediction and hidden representation,
because a per-token unit test on the embedding alone passed while the values were being
annihilated downstream.

## Macro ablation on the primary candidate

`runs/v4-macro-primary-s42`: rank-1 architecture (trial 6, cls, d48, L3), one arm with no
macro branch and one with DBS+COE memory tokens. Same protocol in both arms: select the
epoch on A/holdout, refit on B/fit for exactly that count with the stage-A cosine schedule,
score all 35,158 validation rows once.

| arm | macro tokens | params (B) | inner epoch | B RMSE | MAE | bias |
|---|---|---|---|---|---|---|
| none | 0 | 73,153 | 23 | 468.9730 | 336.3942 | 43.0837 |
| both | 2 | running | — | — | — | — |

The `none` arm is a genuine independent check on the registered path: it was produced by
`macro_ablation.py`'s own loop, not by `search.run_trial`, and it reproduced the registered
`final_trial6` result to every printed digit (inner 581.01643, B 468.9730, epoch 23,
73,153 params).

Timing, stated precisely: the `none` arm took 2,351 s for A (up to 60 epochs, 33 run) **plus**
B (23 epochs) plus a full validation pass at every A epoch. It is not comparable to a number
measured for A alone or B alone.

### Self-sufficiency gap in the macro-arm checkpoints — recorded, not papered over

`macro_ablation.py` saves the two FT `StageState`s but **not** the macro preprocessing state:
no `MacroScalers`, no contract id/policy, no row mapping. Its inline reload rebuilt those
from the live artifacts, so that reload proved the weights reload — not that the saved state
is sufficient to reproduce the predictions.

Closed with a sidecar rather than by rewriting the checkpoint
(`recovery.py`, per-arm `recovery_state.json` + `recovery_verification.json`):

| recorded | verified |
|---|---|
| `MacroScalers` for stages A and B (fit-only) | reload rebuilds them, `fit_macro_scalers` patched to raise |
| contract id/version, `policy: origin_asof`, source sha | contract id matches the live manifest |
| row -> window mapping actually used | reproduces the mapping bit-for-bit |
| checkpoint sha256 before/after | unchanged |

Result on `arm_none`: delta 4.77e-04 SGD over 35,158 rows, rows match, contract matches,
`fitted_anything: false`. Seed 43 will save all of this at training time instead of
recovering it afterwards.

## Reference results (read-only, not mine to change)

| run | RMSE | meaning |
|---|---|---|
| block probe orig/s42 | 475.598687709 | legacy rent LSTM, block tokenisation |
| macro both orig/s42 | 475.400694003 | same + DBS/COE |
| per_feature fixed-7 | 474.491228121 | `reports/spatial_token_ablation_v3/refit7/`, fixed-epoch diagnostic |
| per_feature inner-selected | 543.999111320 | same pipeline with epoch selection |

These four are not interchangeable: they differ in architecture and in how epochs were
chosen.

## Next, in order

1. Let trial 1 finish, then continue 2–8 through `orchestrate.py --resume` (trial 1 is
   reused, not retrained).
2. Register the top two by inner RMSE, then read B once per final run.
3. Macro `none` vs `both` on the primary candidate; seed 43 for that pair if verification
   passes.

No result is claimed in this file.
