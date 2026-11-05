# neural_fusion_v3 — STATUS

Last updated: 2026-10-25. Read this before citing any number.

**Open defect, affects interpretation:** `_masked_lstm_pool` does not ignore leading
padding (measured `max|Δ| = 0.0281` with real `both-s42` weights). It affects **training
rows only** — A/holdout and B/validation have zero masked steps in all three splits — so
the arm-to-arm comparisons stay internally valid, but no trained weight is free of the
artefact. Reproduction, the full per-split census, and what a fix must decide (padding vs.
a genuine no-trade month) are in `DEFECT_masked_lstm_pool.md`. Not fixed yet; v2 stays
frozen so `macro=none` keeps reproducing it bit-for-bit.

## Answer first: can I run `--macro both` today?

**Yes, and it has been run.** In this checkout, on this machine:

```
env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  -m experiments.neural_fusion_v3.main --split orig --controls lstm_fusion \
  --token-modes block --seeds 42 --mode probe --macro both \
  --run-id <new-id> --verify-checkpoints
```

exits 0, trains, saves a checkpoint, and reloads it to 4.5e-13 over all 35,158 validation
rows. The same holds for `--macro none`, `--macro dbs`, `--macro coe`. Four real probe runs
exist on disk (`runs/macro-probe-block-{none,dbs,coe,both}-s42`); their numbers and their
limits are in `RESULTS_macro_block.md`.

What `--macro` still refuses is only the genuinely-missing case: no v3 contract, a policy
mismatch, or absent stage arrays → typed `exit 4`, nothing written.

## Where the macro branch actually stands

**Wired and executed end to end.** `main.py` no longer refuses `--macro`; the old
`MACRO NOT READY` early exit is gone.

Real CLI runs, `orig`, seed 42, `lstm_fusion`, `block` tokens, `--mode macro-smoke`
(2 epochs, capped fit rows, **full 35,158-row validation**), each a separate process:

| macro | tokens | params | reload Δ | run |
|---|---|---|---|---|
| none | 0 | 51,249 | 4.5e-13 | `runs/macro-smoke-fusion-none-r2` |
| dbs | 1 | 65,345 | 4.5e-13 | `runs/macro-smoke-fusion-dbs-r2` |
| coe | 1 | 65,345 | 4.5e-13 | `runs/macro-smoke-fusion-coe-r2` |
| both | 2 | 65,393 | 4.5e-13 | `runs/macro-smoke-fusion-both-r2` |

Those 2-epoch RMSE values (683–685) are **smoke numbers**: they prove the path trains,
steps an optimiser, saves, and reloads. The probe-budget comparison (475.60 / 477.44 /
482.39 / 475.40) is in `RESULTS_macro_block.md`; read its "what these do not support"
section before quoting them.

## What was fixed to get here (each found by a real CLI run, not by a unit test)

1. **`--help` crashed.** `argparse` default read `macro.Policy.ORIGIN_ASOF`; the rewritten
   reader exposes `POLICIES`. Fixed, and `test_cli_blockers.py` now runs `--help` in a
   subprocess.
2. **`self.modules` shadowing.** `_Source` assigned the encoder set to `self.modules`.
   In torch 2.8 `modules` is a class-level method, so the attribute still returned the
   *bound method* while the set was parked in `_modules["modules"]` under a `modules.`
   state-dict prefix. `encode_batch` would have received a method. Renamed to
   `self.encoders`; a test builds the real `_Source` and asserts `encoders.dbs` /
   `encoders.coe` are registered and that an optimiser built from the fused model covers
   every encoder parameter.
3. **Reload refitted the scalers.** `verify_checkpoint` rebuilt stage B with a fresh
   `fit_macro_scalers`, so it verified against a different scaling than training used.
   `build_stage_macro(..., scalers=...)` now accepts the saved state, and the checkpoint
   carries `macro.scalers_a` / `macro.scalers_b`; a test patches `fit_macro_scalers` to
   raise and asserts a reload never calls it. Scaler round-trip (means, scales, channel
   names, fitted_on) is asserted separately.
4. **`row_cap` truncated the evaluation set**, so smoke exports held 512 rows and reload
   verification failed its row-count check. B/validation is no longer capped.

## Checks

- `env -u PYTHONPATH .venv/bin/python -m pytest experiments/neural_fusion_v3/tests -q`
  → **78 passed**.
- Real CLI smoke, four macro modes: all exit 0, all reload within 4.5e-13.
- Real CLI probe, four macro modes: all exit 0, all reload within 4.5e-13.

`tests/test_macro_real_training.py` is the one that would catch a disconnected branch. It
runs the real containers: real v2.1 rows (a 192-row slice per population), real v3 macro
windows, the real `macro_encoders_v3` encoders, and the contract's own residual target with
MSE. What it pins:

* every parameter of an **enabled** branch receives non-zero gradient under the rental loss;
* the **disabled** branch receives none (`build_encoders` always builds both so state-dict
  keys line up, so this is a real risk, not a hypothetical);
* `optimizer.step()` moves real encoder weights;
* `none` and `both` see identical targets and identical tabular/sequence inputs;
* a v2 `state_dict` loaded into the `block`/`none` model reproduces at `atol=0` on contract
  rows;
* a checkpoint save/load round trip reproduces predictions with `fit_macro_scalers`
  replaced by a raiser.

## Known limitations, stated so they are not mistaken for results

- **The macro embeddings are not an economic indicator.** They are learned through the
  rental MSE only; no external label supervises them. `describe()` writes this sentence
  into every manifest and checkpoint.
- **`macro_bridge` expands windows to rows.** `(123,350, 63, 7)` DBS plus
  `(123,350, 12, 24)` COE is roughly 218 MB per stage. Correct, but a copy of the
  deduplicated window table; the unique-window training path is the intended optimisation
  and is **not** done yet.
- **`per_feature` token mode** has a real CLI only for `block`+macro so far; the
  per-feature four-arm probe still needs a clean run
  (`tok-probe-orig-s42` is `MISMATCHED_ARCHITECTURE` — see `runs/README.md`).
- `macro=None` runs are literally v2's classes; `tests/test_macro_fusion.py` proves it by
  loading a v2 `state_dict` into a v3 `block` model and comparing predictions at
  `atol=0`.

## Block macro arms — done (orig, seed 42, probe)

Full write-up and artifact map: `RESULTS_macro_block.md`.

| macro | RMSE | Δ vs none | epochs |
|---|---|---|---|
| none | 475.60 | — | 7 |
| dbs | 477.44 | +1.84 | 4 |
| coe | 482.39 | +6.79 | 4 |
| both | 475.40 | −0.20 | 7 |

All four reload at 4.5e-13 over 35,158 rows. `both`'s 0.20 SGD edge is a tie, not a gain;
`dbs`/`coe` stopped at 4 epochs, so their scores mostly reflect early stopping before the
branch could be used. No claim is made either way about macro usefulness yet.

## Seed 43 (`RESULTS_macro_seed43.md`)

| seed | none | both | both − none |
|---|---|---|---|
| 42 | 475.60 | 475.40 | −0.198 |
| 43 | 479.54 | 476.60 | −2.935 |

Direction reproduces, magnitude does not. Between-seed spread of a single arm is 3.94
(`none`) and 1.20 (`both`) — larger than the seed-42 effect — so 0.198 cannot be quoted as
the size of the gain. Two seeds only; `dbs`/`coe` were not re-run at seed 43 and no third
seed was added, per instruction.

## Report-layer changes (no numerics touched)

`fusion_macro.describe()` now embeds `models.model_description()["encoder"]`, and
`install_per_feature` stores `feature_indices`. Every new run therefore records
`numeric.mode`, `numeric.n_tokens`, `numeric.token_width`, `numeric.bottleneck`,
`feature_names` and `feature_indices`. `tests/test_description_records_layout.py` pins the
distinguishability **and** that a fixed state dict predicts identically before and after
(`atol=0`).

Older runs never recorded this. Their layout was recovered from their own checkpoints and
is documented in `SUPPLEMENT_legacy_run_layouts.md` / `legacy_run_layouts.json`; no old
manifest was rewritten:

| run | layout | numeric tokens | width |
|---|---|---|---|
| all `macro-probe-block-*`, `macro-smoke-*` | block | 1 | 8 |
| `spatial-probe-perfeature-none-s42-h1` | per_feature | 21 | 48 |

## Next, in order

1. Same four arms at `--mode full` — separates "no signal" from "stopped early".
2. 2–3 seeds per arm before any difference is treated as real.
3. Rent-sequence on/off (`lstm_fusion` vs `static_transformer`), separate ablation.
4. `per_feature` tokenisation, kept separate from the macro question. A parallel worker
   owns one `per_feature`/`none` probe run (`spatial-probe-perfeature-none-s42-h1`); it
   does not touch this package or re-run the block baseline.
