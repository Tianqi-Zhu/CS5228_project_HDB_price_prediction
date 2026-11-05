# temporal_fusion_v4 — API

Minimal surface needed to build a v4 temporal arm. Everything imports v2/v3 read-only.

## Window representations (`adapter.py`)

The contract exports `sequence` as `(n, n_levels=4, n_months=12, n_channels=6)` with a
level-major `sequence_order` of `"level::channel"` names.

### `legacy_48_step(sequence, mask, sequence_order) -> WindowRep`

v2's reading: month-major flatten of `(month, level)` into **48 successive steps**,
6 channels each. Four hierarchy levels are treated as four timesteps of the same month.

### `monthly_12_step(sequence, mask, sequence_order, origin_month, panel_start=0, panel_end=None) -> WindowRep`

The corrected reading: **12 calendar steps × 24 channels** (4 levels × 6 channels).
Levels are channels of one month, not timesteps.

Calendar month of slot `s` is `origin_month - (n_months - 1) + s`.

Raises `AdapterError` if an observed cell falls outside `[panel_start, panel_end]`, or if
`mask == 0` disagrees with "calendar outside the panel". Both would mean the source
contradicts its own recorded origin.

### `WindowRep`

| field | shape | meaning |
|---|---|---|
| `values` | `(n, T, C)` float32 | repacked channels |
| `mask` | `(n, T)` float32 | 1 = real timestep, enters the recurrence |
| `observed` | `(n, T)` float32 | 1 = at least one level actually saw a transaction |
| `kept` | `(n, T)` bool | identical to `mask > 0`, kept for clarity |
| `channel_names` | `list[str]` | `level::channel`, length `levels*channels` |
| `step_names` | `list[str]` | per-step label (`m2/l3` for legacy, `slot3` for monthly) |
| `calendar` | `(n, T)` int32 or None | absolute month index per step (monthly only) |

### Mask semantics — important

The producer (`fusion_contract_v2/panel.py:275,294`) sets

```
valid_level = (code >= 0) & in_range
mask[:, level] = valid_level
```

so `sequence_mask` means **"this level is valid and its calendar month is inside the
panel window"**. It says nothing about transactions. The transaction flag is channel
`PANEL_CHANNELS[5]`, written as `observed = (counts > 0) & valid_level` (panel.py:283).

Therefore:

- `mask == 0` → genuine padding (outside the panel, or invalid level). Dropped.
- `mask == 1, observed == 0` → a **real calendar month with no transactions**. Retained as
  a timestep. Calling it "observed" would be wrong.

`month_presence_report(rep)` returns `padded_slots`, `kept_slots`,
`kept_unobserved_months` (real months with no transaction), `observed_months`,
`rows_with_no_kept_month`, `calendar_min`, `calendar_max`. The unobserved count is read
from `observed`, never from the keep-mask.

### Helpers

`level_names(sequence_order, n_levels, n_channels)` → `['global', 'flat_type', 'town_ft', 'block_ft']`

`channel_names(...)` → `['log_mean_rent', 'log_mean_psm', 'log_count', 'month_sin', 'month_cos', 'observed']`

## Encoder (`encoders.py`)

### `PaddingSafeTemporalEncoder(channels, hidden=32, token_dim=48, kind="lstm", project=True)`

Order-preserving compaction before the recurrence: only real steps enter the LSTM, dense
right-padded. Pooling is a masked mean over the compacted steps, so the token is
**invariant to where the padding sits**.

`forward(sequence, mask) -> TemporalOutput(token, present, steps_used)`

| field | shape | meaning |
|---|---|---|
| `token` | `(B, token_dim)` | pooled state; zero for absent rows |
| `present` | `(B, 1)` | 1.0 where the row had ≥1 real step |
| `steps_used` | `(B,)` | number of real steps fed to the recurrence |

Zero history is explicit: `present = 0`, `steps_used = 0`, token finite — never NaN.

### `compact_by_mask(sequence, mask) -> (dense, dense_mask, steps_used)`

The primitive both encoders use. Real steps keep their relative order; output is
right-padded so it batches.

### `legacy_masked_pool(lstm, sequence, mask)`

v2's `_masked_lstm_pool`, reproduced including its defect, so the repair can be measured
against it. `reproduce_legacy_padding_defect(sequence, mask, hidden, seed)` returns the
max absolute difference between left- and right-padded pooling of the *same observations*
(0 for a correct encoder; 0.121 measured on a real early window).

## Models (`models.py`)

### `build_arm(arm, shapes, numeric_columns, temporal_channels, seed=42, capacity=None)`

`arm ∈ {"padding_fixed_48", "monthly_12"}`.

Both arms:

- build via `set_seed(42)` then `build_control("lstm_fusion", ..., token_mode="per_feature")`
- **reuse** `base.block.type_embedding` (temporal token uses the rent slot
  `n_types - 1`), `base.temporal_proj`, `base.gate`
- use an LSTM at `hidden=32`
- add no presence parameter: an absent row yields a zero hidden state which
  `temporal_proj` maps like any other

| arm | recurrent module | parameters (stage A) |
|---|---|---|
| `padding_fixed_48` | v2's own `nn.LSTM`, wrapped, 6-channel input | **56,011** — identical to the fixed-7 diagnostic |
| `monthly_12` | new `nn.LSTM`, 24-channel input | 58,315 (+2,304; only the input width differs) |

`padding_fixed_48` differs from the verified diagnostic in **one behaviour only**: the
pooling call. `assert_init_comparable(...)` rebuilds the reference and compares every
tensor after mapping `lstm.*` ↔ `temporal.rnn.*`; expect
`shared=53 differing=0 only_in_arm=[] only_in_reference=[]`.

`monthly_12`'s new LSTM is an unavoidable re-initialisation, recorded as a residual
difference: a score gap for that arm is not purely a time-axis effect.

### Inputs the model expects

A dict with `static`, `history`, `context`, `categorical` (from
`adapter.build_tensors(..., use_sequence=False)`) plus `sequence` `(B, T, C)` and
`sequence_mask` `(B, T)` from a `WindowRep`. `experiments.temporal_fusion_v4.main._Repacked`
wraps a v2 `Tensors` with a `WindowRep` and reuses v2's batching; its `batches()` keeps
`with_target=False` as the default so shared callers like `training.predict` work.

## Training (`main.py`)

```
.venv/bin/python -m experiments.temporal_fusion_v4.main --arm padding_fixed_48 \
    --run-id temporal-paddingfix48-none-s42-h1
```

- A: train A/fit, monitor A/holdout, cosine `T_max=20`, ≤20 epochs, patience 5
- B: **fixed 7 refit epochs, predeclared** (the block reference's recorded inner
  selection). No outer score is used to choose an epoch.
- B scaler is the **saved fit-only** state from the fixed-7 diagnostic, verified equal to
  a fresh `B/fit` fit before use.
- Saves predictions, checkpoint (with both curves, separator states, init parity),
  preprocessing, `run_summary.json`, `run_manifest.json`, and a load-only reload check.

One process, `OMP_NUM_THREADS=2`.
