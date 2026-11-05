# Macro encoders v3 — status

Owner: Hermes. Consumer: Hermes2 (read-only). Reviewer: Codex.
Package: `experiments/macro_encoders_v3/`. API doc: `API.md` (same directory).

**API is stable and consumable.** No further interface changes are planned for this
package; the three blockers found in independent review are fixed and pinned by
regression tests.

---

## Current state

| item | state |
|---|---|
| `API.md` (consumer interface) | **written** |
| DBS encoder (LSTM 32→48, 63×7) | **done** |
| COE encoder (GRU 32→48, 12×24) | **done** |
| Fit-only `ScalerState` (4 blocks, save/reload) | **done** |
| Unique-window encode → live row gather | **done** |
| Tests | **36 passed** (24 base + 12 blocker regressions) |
| Real `orig` A/B smoke (forward/backward/optimizer) | **done**, post-fix |
| Checkpoint reconstruction without fitting | **done** — 22 checks passed |
| Codex independent run | 60 passed (this package + neural fusion test) |
| Tree-model grids, rental scoring | **deliberately not attempted** |

Contract consumed:
`reports/fusion_contract_v3/fusion-contract-v3.0.0__43867d0ba111__c52fef9f33cf__e32fccd39da3`
(policy `origin_asof`).

Nothing under `experiments/neural_fusion_v1/`, `experiments/neural_fusion_v2/`,
`experiments/neural_fusion_v3/`, `experiments/fusion_contract_v2/` or
`experiments/fusion_contract_v3/` was modified. The v3 contract's exported arrays
are read-only inputs here.

---

## 1. The three blockers, fixed

All three were reported by independent review. Each has a test that fails on the
pre-fix code.

### 1.1 `build_window_tensors` raised on the DBS scalar block

`ScalerState.transform` accepted only a per-position mask, while `dbs_scalar_mask`
is per-element `(W, 12)`. Two things were wrong: the transform was too narrow, and
the builder computed a block it never used.

Fix: `transform` now accepts **either** `(...,)` or `(..., C)` and masks
element-wise; the dead scalar scaling was removed from the builder (scalar blocks
belong to the flat-feature path, not to the window tensors). Masked elements come
out exactly `0.0` with the mask carrying the truth — a masked scalar is *absent*,
not "zero data".

Tests: `test_dbs_scalar_block_transform_accepts_per_element_mask`,
`test_masked_scalar_is_absent_not_zero_valued_data`,
`test_build_window_tensors_succeeds_and_drops_dead_scalar_scaling`.

### 1.2 Padding position changed the token (the serious one)

Both encoders ran `pack_padded_sequence(lengths = mask.sum(dim=1))`, which assumes
the valid steps are a **prefix**. The real exported windows are **leading-padded**:
`orig/A` window 0 is 43 zeros then 20 ones (DBS) and 11 zeros then 1 one (COE). The
recurrence therefore consumed padding and dropped the real tail. Measured effect
before the fix: left- vs right-padded identical observations differed by up to
**0.219** (DBS) and **0.464** (COE) in token space.

Fix: `encoders.compact_masked()` gathers the valid steps in their **original order**
into a dense sequence (keeping true temporal order, so an interior hole does not
reorder anything), and the recurrence runs over that compacted sequence via
`pack_padded_sequence`. Leading padding, right padding and interior holes now give
identical tokens.

Tests: `test_dbs_token_is_invariant_to_padding_position`,
`test_coe_token_is_invariant_to_padding_position`,
`test_dbs_uses_the_real_tail_of_a_leading_padded_window`,
`test_coe_uses_the_real_tail_of_a_leading_padded_window`,
`test_mid_gap_window_keeps_true_order`,
`test_real_early_window_is_leading_padded` (asserts the real 43/20 and 11/1 layout).

A secondary bug surfaced while fixing this: `_recurrent_last_state` initially took
`state[-1]` on the recurrent output, which for an `LSTM` is the **cell** state
tuple element, producing a token with a leading layer dimension. LSTM and GRU return
shapes now handled separately.

### 1.3 `MacroEncoderSet` as a `@dataclass`

Reported as a risk to `parameters()` enumeration. Measured: **enumeration works**,
14,048 parameters, unique ids, and a real `SGD` step updates weights. The
`@dataclass` decorator was nevertheless removed during the padding work (a
dataclass-generated `__eq__`/`__hash__` breaks `nn.Module` bookkeeping in general),
so this is now a regression guard rather than a documented failure:

`test_encoder_parameters_enumerate_and_optimize`,
`test_encoder_set_is_hashable_and_not_a_dataclass`.

---

## 2. Real-data results (post-fix)

`orig`, policy `origin_asof`, 2 threads, one process.

| stage | rows | windows | DBS | COE | params |
|---|---:|---:|---|---|---:|
| A | 88,192 | 27 | `(27,63,7)` → `(27,48)` | `(27,12,24)` → `(27,48)` | 14,048 |
| B | 123,350 | 39 | `(39,63,7)` → `(39,48)` | `(39,12,24)` → `(39,48)` | 14,048 |

Verified on both stages:

- forward runs on the actual exported windows;
- **both** branches receive non-zero gradients
  (DBS `lstm.weight_ih_l0` ≈ 0.20, COE `gru.weight_ih_l0` ≈ 2.6–3.3);
- a real `torch.optim.SGD` step changes a projection weight;
- a saved `state_dict` reproduces tokens **exactly** after reload;
- no window is fully missing on real data;
- window 0 of both stages really is leading-padded and its tail is used.

Regenerate/verify:

```bash
.venv/bin/python -m experiments.macro_encoders_v3.smoke --split orig
.venv/bin/python -m experiments.macro_encoders_v3.verify_checkpoint --split orig
```

**The smoke loss is a surrogate** (`mean(token²)` per branch) whose only job is to
produce gradients and exercise the optimizer. `smoke.json` contains **no
rental-prediction score**. Nothing here indicates whether macro inputs improve rent
prediction; that question belongs to a downstream fusion experiment.

---

## 3. Scaler isolation

`fit_macro_scalers` uses only windows referenced by `population == "fit"` rows:
**38 of 39** on `orig/B`, **26 of 27** on `orig/A` (the excluded window in each
stage is the frozen one: inner holdout in A, outer validation in B). Verified by
`test_scaler_state_ignores_validation_values`, which tampers with a
validation-only window and asserts the fitted mean/scale/counts are unchanged bit
for bit, and by `test_scaler_masks_are_respected_in_the_mean`, which recomputes a
channel mean by hand over masked positions.

State round-trips through JSON exactly (`test_scaler_state_round_trips_exactly`)
and produces identical tensors after reload
(`test_reloaded_scaler_produces_identical_tensors`).

---

## 4. Not done

1. **No rental-prediction score.** These are encoders and a smoke harness, not a
   model comparison. Any claim about macro usefulness needs a fusion run.
2. **No `early` / `long` smoke.** The code path is identical and takes under a
   second per stage, but only `orig` was exercised. Running them would create new
   artifact directories.
3. **No sequence-length or capacity search.** 63 rows / 12 months, hidden 32,
   one layer each — the values requested. No ablation of those.
4. **No banking basket** (OCBC/UOB).
5. **No `target_previous_month` use.** The contract supports it; nothing here has
   been run against such a contract directory.
6. **No Kaggle artifact, no full-data retrain, no commit, no branch change.**

---

## 5. Known limitations worth stating

- **DBS `log_close` is a level, not a return.** It is standardised on fit windows, so
  across a long horizon its standardised value drifts far from the training range.
  That is a property of the input, not a bug, but it is the channel most likely to
  behave oddly in extrapolation.
- **COE `premium` is in SGD and standardised per category-month channel.** A single
  scalar per channel means a category's premium level is divided by its own SD; the
  cross-category level ordering is therefore not preserved in token space.
- **Only one `observed` flag per block.** The per-category detail survives in
  `WindowTensors.coe_channel_mask`, but the encoder does not consume it. A future
  variant could, at the cost of a wider input.
- **`coe_mask` is per-month "any category".** A month where only category A had an
  auction is `observed = 1`; the three other categories are zeros in that month's
  channels. This is faithful to the export, but a model cannot tell "no auction" from
  "premium happened to standardise to 0" without `coe_channel_mask`.
