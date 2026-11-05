# Macro encoders v3 — consumer API

**Status: stable.** Producer: Hermes. Consumer: Hermes2, read-only.
Package: `experiments/macro_encoders_v3/`. Tests: 36 passing (incl. 12 blocker
regressions). Do not edit this package; request changes through Codex.

This module turns the **already exported** macro tensors from
`reports/fusion_contract_v3/<contract_id>/` into two 48-dimensional tokens per
supervised row: one for DBS (`D05.SI`, 63 trading days) and one for COE
(4 categories × 12 months). It does not read `data/auxiliary/*` and does not
re-parse anything — the v3 contract is the only source.

---

## 1. What you get

| output | shape | meaning |
|---|---|---|
| `dbs_token` | `(N, 48)` | DBS financial-state embedding for each supervised row |
| `dbs_missing` | `(N, 1)` | 1.0 = this row's window had **no** DBS observation |
| `coe_token` | `(N, 48)` | COE embedding for each supervised row |
| `coe_missing` | `(N, 1)` | 1.0 = this row's window had **no** COE observation |

`N` is the number of supervised rows you asked for (a whole stage, or one batch).
Tokens are always **finite**. Missingness is carried by the `*_missing` flag, never
by `NaN` and never by a silently-valid zero vector — a window with no history is
projected from a learned "empty" state and flagged so your fusion layer can drop it.

---

## 2. Architecture (as actually built)

| branch | input | recurrent | hidden | projection | params |
|---|---|---|---|---|---|
| DBS | `(B, 63, 7)` | `LSTM`, 1 layer | 32 | `Linear(32→48)` | 6,864 |
| COE | `(B, 12, 24)` | `GRU`, 1 layer | 32 | `Linear(32→48)` | 7,184 |

Total 14,048 parameters — verified by `encoders.parameter_count()`.

**Why COE is `(12, 24)` and not `(48, 6)`.** The month is the recurrent axis.
The four categories are *channels of the same month* (4 × 6 = 24), flattened
category-major: `a.premium, a.quota, …, e.auction_count, e.observed`. Laying the
categories end to end along time would turn a 12-month history into 48 pseudo-months
and silently claim four times the temporal reach. `data._flatten_coe()` is the only
place that reshape happens, and it transposes `(W, cat, month, chan) →
(W, month, cat*chan)`.

---

## 3. Padding and masked steps

Real exported windows are **leading-padded**: `orig/A` window 0 has 43 zeros then 20
real trading days for DBS, and 11 zeros then 1 month for COE. Windows can also
contain interior holes. The encoders therefore **gather the valid steps in their
original order** and run the recurrence over that compacted sequence — they do not
assume the valid steps form a prefix. A leading-padded window and the same
observations right-padded produce **bit-identical** tokens (asserted in
`test_dbs_token_is_invariant_to_padding_position`).

Practical consequence for you: pass the mask, always. Never re-sort or trim a window
yourself; the encoder handles the layout.

---

## 4. API

```python
from experiments.macro_encoders_v3 import contract, data, scaling, encoders, api

manifest = contract.load_manifest()                 # newest exported v3 contract
windows  = data.load_windows(manifest, "orig", "B") # unique windows: W = 39
rows     = data.load_rows(manifest, "orig", "B")    # per-row mapping: N = 123,350

scalers  = scaling.fit_macro_scalers(windows, rows) # fit-only; see §5
# or, to avoid refitting:  scalers = scaling.MacroScalers.load(scaler_dir)

tensors  = api.build_window_tensors(windows, scalers)
modules  = encoders.build_encoders()                # 14,048 params
```

### 4.1 Whole-stage encoding (all windows at once)

```python
per_window = api.encode_windows(modules, tensors)
# per_window["dbs_token"] : (39, 48)
# per_window["coe_token"] : (39, 48)
# per_window["dbs_missing"], per_window["coe_missing"] : (39, 1)

per_row = api.gather_to_rows(per_window, rows.row_window_index, rows.row_id)
# per_row["dbs_token"] : (123350, 48)   <- live gather, graph preserved
```

### 4.2 Per-batch encoding (recommended inside a training loop)

Encodes only the windows the batch actually references, then gathers. This is what
makes a deduplicated batch cheap; gradients still flow to every window the batch
touches.

```python
batch = api.encode_batch(modules, tensors, rows.row_window_index, row_index)
# -> {"dbs_token": (len(row_index), 48), "coe_token": (...), "dbs_missing": (...), ...}
```

`api.batch_indices(row_window_index, row_index, n_windows)` is exposed if you want
the `(unique_window_ids, inverse)` mapping yourself.

### 4.3 Ablation switches

```python
modules = encoders.build_encoders(use_coe=False)   # DBS-only arm
# or freeze one branch:
for parameter in modules.coe.parameters():
    parameter.requires_grad_(False)
```

Both branches are always constructed so `state_dict` keys line up across arms.

---

## 5. Scaling: fit-only, and what "fit-only" means here

`scaling.fit_macro_scalers(windows, rows)` uses **only the windows referenced by
`population == "fit"` rows** of that stage. On `orig/B` that is **38 of 39** windows;
on `orig/A`, **26 of 27**. Holdout and validation windows never contribute to a mean
or a standard deviation. (The single excluded window in each stage is the one
referenced only by frozen rows: the inner holdout's window in stage A, the outer
validation's window in stage B.)

Four states are produced and saved together:

```python
scalers = scaling.MacroScalers.load(scaler_dir)   # exact round-trip
scalers.dbs          # (7,)  channels
scalers.dbs_scalar   # (12,) scalar controls
scalers.coe          # (24,) month channels
scalers.coe_scalar   # (36,) = 4 categories x 9
```

Each `ScalerState` records `mean`, `scale`, `degenerate`, `n_used`, `fitted_on`,
`split_id`, `stage`, `policy`. A channel with fewer than two usable observations or
zero variance gets `scale = 1.0` and `degenerate = True` rather than a divide-by-zero.

`ScalerState.transform(values, mask)` accepts **either** a per-position mask
`(...,)` **or** a per-element mask `(..., C)`; both are honoured, and masked
elements come out exactly `0.0` with the mask carrying the truth.

`build_window_tensors` scales only the *sequence* blocks. The scalar blocks are not
part of the window tensors; use the scaler states directly if you want them as flat
features.

---

## 6. Files on disk

```
reports/macro_encoders_v3/<contract_id>/
  smoke.json                                  shapes, params, gradient sums, timings
  scalers/<split>/stage<A|B>/index.json       + dbs_channels.json, dbs_scalars.json,
                                              coe_channels.json, coe_scalars.json
  checkpoints/<split>_stage<A|B>_smoke.pt     state_dict only, load with weights_only=True
```

Reload without fitting:

```python
scalers = scaling.MacroScalers.load(out_dir / "scalers" / "orig" / "stageB")
modules = encoders.build_encoders()
modules.load_state_dict(torch.load(checkpoint, weights_only=True))
modules.eval()
```

`verify_checkpoint.py` proves this path reproduces the recorded tokens; it does not
and cannot fit anything.

---

## 7. Verified on real data

`orig`, policy `origin_asof`, contract
`fusion-contract-v3.0.0__43867d0ba111__c52fef9f33cf__e32fccd39da3`:

| stage | rows | windows | dbs in → token | coe in → token |
|---|---:|---:|---|---|
| A | 88,192 | 27 | `(27, 63, 7)` → `(27, 48)` | `(27, 12, 24)` → `(27, 48)` |
| B | 123,350 | 39 | `(39, 63, 7)` → `(39, 48)` | `(39, 12, 24)` → `(39, 48)` |

Confirmed on both stages: forward runs; **both** branches receive non-zero
gradients; a real `SGD` step changes a weight; a saved checkpoint reproduces tokens
exactly; no window is fully missing on real data.

**The smoke loss is a surrogate** (`mean(token²)` per branch) whose only purpose is
to produce gradients. `smoke.json` contains **no rental-prediction score** and
nothing in it says whether macro inputs help. Do not cite it as a result.

---

## 8. Thread and process limits

`torch` runs at its default thread count unless you set it. If you are sharing the
machine, pin it:

```python
torch.set_num_threads(2)
```

Every command in this package was run with `OMP_NUM_THREADS=2`, one process.

---

## 9. Commands

```bash
cd <project root>

# tests (36)
env -u PYTHONPATH -u PYTHONSTARTUP OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
    experiments/macro_encoders_v3/tests -q

# real-data smoke: forward, backward, checkpoint, scalers (orig A and B)
env -u PYTHONPATH -u PYTHONSTARTUP OMP_NUM_THREADS=2 .venv/bin/python \
    -m experiments.macro_encoders_v3.smoke --split orig

# checkpoint reconstruction without fitting
env -u PYTHONPATH -u PYTHONSTARTUP OMP_NUM_THREADS=2 .venv/bin/python \
    -m experiments.macro_encoders_v3.verify_checkpoint --split orig
```

---

## 10. Gotchas

- **Policy is per contract.** These arrays are `origin_asof`. A
  `target_previous_month` contract directory would carry different windows; assert
  `windows.policy == "origin_asof"` if your experiment requires strict
  fixed-cutoff forecasting, and never mix tokens from two policies in one model.
- **`window_index` is stage-specific.** A row's `macro_window_index` from stage A is
  meaningless against stage B's window table. Always pair `windows` and `rows` from
  the same `(split, stage)`.
- **`coe_mask` is per-category; `WindowTensors.coe_mask` is per-month.** The encoder
  takes the per-month "any category observed" mask, and
  `WindowTensors.coe_channel_mask` keeps the finer `(month, 24)` detail if you need
  it. A genuine `$0` COE premium is a real value with `observed = 1`, not a hole.
- **Category E has no 2026-08 row.** Irrelevant for `orig` (its windows end well
  before that), but do not be surprised by an all-zero E slice in a future window.
- **Do not cache `encode_windows` output across optimiser steps.** The encoder is
  meant to learn through the rental loss; a detached cached token would freeze it.
