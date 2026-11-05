# temporal_fusion_v4 — STATUS

**Live state, updated as runs complete.** Task 2 (Hermes). Owner of
`experiments/temporal_fusion_v4/` and `reports/temporal_fusion_v4/` only.

## Reference numbers this work is measured against

| run | RMSE (SGD) | note |
|---|---|---|
| block / 7 B epochs (`macro-probe-block-none-s42`) | 475.598687709 | immutable reference |
| per_feature / 1 B epoch (probe) | 543.999111320 | inner selection chose 1 |
| **per_feature / 7 B epochs** (`spatial_token_ablation_v3/refit7`) | **474.491228121** | predeclared fixed-7 diagnostic |

## What the fixed-7 protocol does and does not mean

The two ran with the same declared hyperparameters and each selected its B refit length
by its own inner early-stopping rule; that is a legitimate pipeline comparison. This
package's arms instead use a **predeclared fixed 7**, because 7 is the block reference's
recorded inner selection and a controlled comparison needs one fixed number.

**A stage's inner-selected epoch count must not be reported as a B training schedule.**
Concretely: if an arm's A stage early-stops at 4 epochs, its B stage still runs the
predeclared 7. The B curve in the output is therefore always 7 entries regardless of
what A chose, and `run_summary.json` records `stage_a.best_epoch` and
`stage_b.curve` separately so the two numbers cannot be confused. `fixed_b_epochs: 7`
plus `epochs_predeclared: true` are written into both the summary and the checkpoint,
with an `epochs_provenance` string.

No outer-validation score is used to pick any epoch.

## Bug found and fixed during the first real attempt (kept, not deleted)

`reports/temporal_fusion_v4/failed/temporal-paddingfix48-none-s42-h1.attempt1.log`

The first attempt completed stage A (9 epochs, best epoch 4, holdout 595.2284) and then
crashed entering stage B with `IndexError: index out of range in self` from an embedding
lookup. Cause: B/fit has its own categorical vocabulary — the third categorical column's
cardinality is **20 at B/fit but 19 at A/fit** — so building the B model from A's
`shapes` under-sized its embedding tables.

This is the same +8-parameter asymmetry noted earlier: the run summaries quote stage-A
parameter counts while the saved checkpoint is stage-B. Fix: the B model is now built
from `shapes_of(b_fit, True, encoding_b)`, exactly as v3's runner does, and the
checkpoint records `stage_a_shapes` and `stage_b_shapes` separately. A/B shape agreement
on numeric/history/context/sequence width is asserted before building.

The failed attempt's log is retained. Its run directory was removed because it had
written no artifacts; the log is the evidence.

## Arm 1 — `padding_fixed_48`

Design: the verified fixed-7 diagnostic's model, with only the pooling call changed.
Padded steps are compacted away before the recurrence instead of being fed in as zero
vectors, so the pooled token no longer depends on pad position.

Held identical to the reference, verified rather than asserted
(`experiments/temporal_fusion_v4/tests_temporal.py`, §7):

```
shared tensors 53, differing 0, only in arm [], only in reference []
PASS  padding_fixed_48 inherits the reference's backbone weights bit-for-bit
```

- `type_embedding` is `base.block.type_embedding`, untouched; the temporal token uses the
  rent slot `n_types - 1` that v2 already reserved. No extra row, so no existing type
  vector changes.
- `temporal_proj` and `gate` are `base.temporal_proj` / `base.gate`, reused verbatim.
- No presence parameter is added; a zero-history row emits a raw zero hidden state which
  `temporal_proj` maps like any other.
- v2's own `nn.LSTM` is wrapped, not rebuilt, so it holds the reference's weights.
- Parameters: **56,011 (stage A)** — identical to the reference. A pure repair adds no
  parameters.

**Complete.** Run `temporal-paddingfix48-none-s42-h1`, launched after the shapes fix.

| | value |
|---|---|
| RMSE | **478.487179** |
| MAE | 339.141880 |
| bias | +89.983092 |
| n | 35,158 |
| parameters A / B | 56,011 / 56,019 |
| stage A | 9 epochs, best epoch **4** (holdout 595.2284) |
| stage B | **7 epochs** (predeclared; not A's 4) |
| init parity | comparable, 53 shared tensors, 0 differing |
| wall clock | A 158.9 s, B 284.8 s, total 451.0 s |
| reload | `max |delta| = 0.0` SGD |

Stage A curve (train MSE kSGD², holdout RMSE SGD): 0.309155/598.43, 0.248685/604.20,
0.245290/601.58, **0.242844/595.23**, 0.242426/611.12, 0.240902/599.42, 0.241088/609.07,
0.239776/602.61, 0.239446/624.25. The monitor stopped after 9 epochs (patience 5 from
epoch 4) and restored epoch 4's weights.

Stage B curve (7 epochs, train MSE kSGD²): 0.34446, 0.28675, 0.28252, 0.27950, 0.27818,
0.27718, 0.27703.

Note the two epoch counts are different on purpose and must not be conflated: **A's inner
selection was 4, B trained the predeclared 7.** The B curve has 7 entries because the
schedule was fixed in advance, not because A chose 7.

### Effect vs repair — read this before comparing

**The repair is a correctness fix, not a score improvement.** This arm scores
**478.487179**, which is **+3.995951 worse** than the legacy fixed-7 diagnostic
(474.491228121) and +2.888 worse than the block/7 reference (475.598688).

That is a legitimate outcome and it is reported as such. The legacy pooling reads a state
contaminated by pad count; that contamination happens to act like a form of noise
injection here, and removing it did not help this seed's validation score. What the
result supports:

- the defect is real and measurable (0.121302 token difference on a real early window),
- the repair removes it (0.0 at every pad position),
- on this one seed and this fixed budget, removing it **did not improve** RMSE.

What it does **not** support: any claim that padding-safe pooling is worse, or better, as
a modelling choice. One seed, one fixed budget, a deterministic training path. The
legacy branch is not being proposed as a new main model and is not being reverted to;
both are reported side by side with the defect documented.

Bias is where this arm sits worst (+89.98 vs +57.16 for the legacy fixed-7), so the change
moved the error level rather than its spread, consistent with the earlier finding that
these arms differ mainly in bias.

## Arm 2 — `monthly_12`

**Complete.** Run `temporal-monthly12-none-s42-h1`.

| | value |
|---|---|
| RMSE | **478.497466** |
| MAE | 339.252564 |
| bias | +92.149300 |
| n | 35,158 |
| parameters A / B | 58,315 / 58,323 (+2,304 / +2,304 over arm 1) |
| stage A | 6 epochs, best epoch **1** (holdout 595.7343) |
| stage B | **7 epochs** (predeclared; not A's 1) |
| wall clock | A 146.1 s, B 556.6 s, total 711.8 s |
| reload | `max |delta| = 0.0` SGD |

Same construction, same reused `temporal_proj`, `gate`, `type_embedding`, same LSTM family
and hidden width. **One unavoidable difference**: the 12-step × 24-channel reading needs a
24-channel recurrent input, which a 6-input v2 LSTM cannot accept, so this arm's
`nn.LSTM` is newly initialised. That re-initialisation is recorded as a residual
difference — the gap to arm 1 is **not** purely a time-axis effect.

Again, A chose 1 and B trained the predeclared 7; the 7 B entries are the fixed schedule,
not A's selection.

## All arms side by side

Recomputed independently from the prediction CSVs, 35,158 rows each:

| arm | RMSE | MAE | bias |
|---|---|---|---|
| block / 7 (`macro-probe-block-none-s42`) | 475.598688 | 341.273758 | +56.4537 |
| per_feature / 1 (`spatial-probe-perfeature-none-s42-h1`) | 543.999111 | 390.280821 | +265.9103 |
| per_feature / 7 (`spatial_token_ablation_v3/refit7`) | 474.491228 | 340.080944 | +57.1592 |
| **padding_fixed_48 / 7** | **478.487179** | 339.141880 | **+89.9831** |
| **monthly_12 / 7** | **478.497466** | 339.252564 | **+92.1493** |

Row agreement verified for both new runs against the refit7 reference: identical
`SOURCE_ROW_ID` order, identical labels (max abs delta 0.0), identical `horizon_months`,
identical `target_month`.

The two v4 arms land within 0.010 SGD of each other (478.487 vs 478.497) on identical rows
and identical budgets, while differing in time axis, input width and initialisation. That
near-equality is itself the notable result: changing the time-axis reading moved the score
by 0.01 SGD, far below what the token-mode change moved it (68.4 SGD between the 1-epoch
and 7-epoch variants), so the temporal representation is not where this model's error
lives.

Both v4 arms are ~4 SGD worse than the legacy fixed-7 diagnostic. Since the padding repair
is a correctness fix rather than a tuning change, that is reported as-is and neither arm is
proposed as a replacement main model; no defective branch is being reinstated either.

## Mask semantics — what "observed" means

The producer (`fusion_contract_v2/panel.py:275,294`) sets
`valid_level = (code >= 0) & in_range` and `mask[:, level] = valid_level`, so
`sequence_mask` means **"valid level, calendar month inside the panel window"**. It says
nothing about transactions.

The transaction flag is a **separate channel**, `PANEL_CHANNELS[5]`, written as
`observed = (counts > 0) & valid_level` (panel.py:283).

So in this export:

- `mask == 0` → genuine padding (outside the panel). Dropped.
- `mask == 1, observed == 0` → a real calendar month with **no transactions**. Retained as
  a timestep.

On the real export, every in-panel month happens to have at least one observed level, so
`kept_unobserved_months` is 0 today; the adapter counts it from the channel anyway so the
distinction survives data changes. Nothing in this package calls `mask == 1`
"has transactions".

## The defect being repaired, measured on real data

`PaddingSafeTemporalEncoder` exists because v2's `_masked_lstm_pool` zeroes masked
*inputs* and masks *outputs* but still advances the recurrence through them. For trailing
padding that is harmless; for leading padding the first real step reads a state built
from zeros.

This export is mostly leading padding: 31,807 of 52,310 A/fit rows have an empty first
window slot, because `mask` is `suffix(origin)` — a row with origin `t < 11` fills only
the last `t + 1` slots of a 12-month window.

Measured on a real early window (row 626, origin 2, 3 valid months):

```
legacy pooling, same observations left- vs right-padded: max |diff| = 0.121302
```

and 0.0 for a fully-observed window. The repaired encoder gives exactly 0.0 at every pad
position on the same observations.

## Test coverage

`experiments/temporal_fusion_v4/tests_temporal.py` — **28 passed, 0 failed**.

Covers: defect reproduced on real windows (0.121302) and absent on a full window; pad
invariance of the repair; zero history finite with `present=0`; legacy 48 steps / 24
channels; monthly 12 steps / 24 channels; level and channel names preserved; keep-mask is
"calendar in panel" and `observed` is read from channel 5; an observation outside the
panel calendar raises instead of being retained; nonzero gradients through both encoders;
init parity of arm 1 against the reference; our copy of v2's pooling equals v2's.
