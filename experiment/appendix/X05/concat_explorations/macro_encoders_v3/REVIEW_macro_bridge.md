# macro_bridge — read-only review

Scope: `experiments/neural_fusion_v3/macro_bridge.py` (209 lines) and the parts of
`experiments/neural_fusion_v3/main.py` that consume it, reviewed **read-only**.

**No consumer source was modified.** I did not write to, revert, or reformat
anything under `experiments/neural_fusion_v3/`. Hermes2 is actively editing those
files; this review is a snapshot at the time of reading and its findings are
reproducible from the audit script below.

Evidence: `experiments/macro_encoders_v3/bridge_audit.py` — 54 checks, exit 0.

```bash
env -u PYTHONPATH -u PYTHONSTARTUP OMP_NUM_THREADS=2 .venv/bin/python \
    -m experiments.macro_encoders_v3.bridge_audit
```

Contracts used: macro `fusion-contract-v3.0.0__43867d0ba111__c52fef9f33cf__e32fccd39da3`
(policy `origin_asof`), v2 rent `fusion-contract-v2.1.0__43867d0ba111`.

---

## 1. What the bridge gets right (verified, not asserted)

| property | result |
|---|---|
| `position` map covers every stage row exactly once, values `0..n-1` | pass |
| lookup round-trips (`row_id[position[id]] == id`) | pass |
| `row_id` strictly ascending in both stages | pass |
| per-row `dbs_seq` / `coe_seq` equal an independent gather of the producer's windows | pass |
| `window_index`, `macro_cutoff_month`, `row_id` identical to the producer's | pass |
| `scaler_fitted_on` names the fit population only | pass |
| under `origin_asof`, per-row cutoff equals the row's `origin_month_index` | pass |
| population row sets identical to the producer's sample tables (both stages, all four populations) | pass |
| `align_population` returns **stage-absolute positions**, and `row_id[positions]` reproduces the population ids exactly | pass |
| `MacroTensors` subsets keep row ids, targets and macro arrays on the same index | pass |
| unshuffled batches: macro arrays track `macro_row`; targets track the batch offset | pass |
| shuffled batches: permutation covers the population exactly once and replays from the seed | pass |

Population sizes reconcile with the contract: A/fit 52,310 (26 windows), A/holdout
35,882 (1), B/fit 88,192 (38), B/validation 35,158 (1).

The frozen-population single window is correct under `origin_asof` (all held-out rows
share the cutoff), and it is why the fit-only scaler excludes exactly one window per
stage — consistent with the 26/27 and 38/39 figures in the encoder docs.

---

## 2. Finding — `align_population`'s return value is a trap (medium severity)

`align_population` looks up each `SOURCE_ROW_ID` in `position` and returns the
resulting **stage-absolute positions**. `StageMacroTensors.select()` expects
**`SOURCE_ROW_ID`s** and does the same lookup internally.

At call time in `main.py` the two are equivalent — the row ids are looked up and the
positions come out — so nothing breaks. But the two functions return *the same
numeric type* describing *different things*, and on this data the values are
indistinguishable as scalars:

```
first 50 positions that are also valid row ids: 50/50
```

That is: every position index `0..n-1` is also a legal `SOURCE_ROW_ID` for some row.
So if a future edit ever passed `align_population`'s result into `select()`, it would
not raise — it would silently map rows to the wrong stock/COE windows. The audit
demonstrates the mis-map is real, not hypothetical:

```
PASS  select mis-maps if fed align_population's return value
```

Currently harmless because:

- every call site in `main.py` uses the return value **only** as a validation side
  effect (the results are discarded — verified by reading all four call sites);
- batching uses `MacroTensors.row_ids`, not `align_population`'s output.

**Suggested fix (Hermes2's call):** either `align_population` returns nothing and is
renamed `assert_population_aligned`, or it returns a distinct wrapper type. The
cheapest safe change is the rename plus dropping the return, since no caller needs
the value. I am not making this edit.

---

## 3. Finding — `MacroTensors.batches` rebuilds a sub-batch via `next(iter(...))` (low severity, correctness OK)

`batches()` calls `self.base.subset(index).batches(len(index), shuffle=False)` and
takes `next(...)`. The audit confirms this is **currently correct**: the yielded
feature dict and targets correspond to the same rows, unshuffled and shuffled, and
the permutation replays from the seed.

Two things worth knowing rather than fixing:

- It allocates a `Tensors` subset per batch, which copies the numeric/history/context
  arrays. For B/fit (88,192 rows, batch 4,096) that is a few extra copies per epoch.
  A zero-copy path would slice the arrays directly, but it would also duplicate v2's
  feature-assembly logic — the current design deliberately trades efficiency for
  "cannot drift from v2". I would keep it unless epoch time becomes a problem, and
  say so in the runtime report if it does.
- `next(iter(generator))` silently takes the first batch. If `len(index)` were ever
  0 the `next()` would raise `StopIteration` instead of yielding nothing. `batches`
  cannot currently produce an empty slice (`range(0, len(self), batch_size)` with
  `batch_size >= 1`), so this is latent, not live.

---

## 4. Finding — target arrays are passed through by reference (informational)

`MacroTensors.actual_sgd` / `anchor_ksgd` return `self.base`'s arrays directly, not
copies. `subset()` slices `row_ids` but delegates target slicing to
`base.subset(index)`, so alignment holds (verified). The only hazard is a caller
mutating the exposed array in place, which would corrupt the base tensors too. No
current caller does this.

---

## 5. Consumer-side issues I did **not** review

Reported independently and out of scope here, listed so this file does not read as a
clean bill of health for `main.py`:

- `--help` / import-time failure on a non-existent `macro.Policy`;
- `_Source.self.modules` colliding with `nn.Module.modules`;
- `verify_checkpoint` calling `build_stage_macro` again, which **re-fits** the
  scalers instead of loading the saved `scalers_B` — that defeats the "reconstruct
  without fitting" requirement even though the resulting numbers may match.

The last one matters for the gate: a checkpoint reconstruction that re-derives its
scalers has not proven the saved state is sufficient. If Hermes2 wants, the encoder
package already provides `scaling.MacroScalers.load(dir)` and
`verify_checkpoint.py` as a working reference for the load-only path.

---

## 6. What I verified about optimizer registration (read-only, partial)

I did not execute a training run. From reading `main.py`: the macro encoders are
constructed inside `_macro_source` and returned as a `torch.nn.Module` subclass, so
they are reachable from the model's parameter tree **if** `build_fusion` registers
that module as a submodule. That is the one thing I could not confirm without either
running a fit or trusting the `nn.Module.modules` attribute that is reported broken.
Once the `self.modules` shadowing is fixed, the check to run is:

```python
names = [n for n, _ in model.named_parameters()]
assert any(n.startswith("<macro source prefix>.") for n in names)
```

and confirm the DBS `lstm.weight_ih_l0` and COE `gru.weight_ih_l0` appear in it.
The audit script does not do this because it would require constructing the consumer
model, which touches files Hermes2 is editing.

---

## 7. Summary

| # | item | severity | state |
|---|---|---|---|
| 1 | mapping, population membership, subset/batch ordering, target alignment | — | **54 checks pass** |
| 2 | `align_population` returns positions while `select` expects row ids, and the two are scalar-indistinguishable on this data | medium (latent) | reported; no caller currently misuses it |
| 3 | `batches` copies per batch via `next(iter(subset.batches(...)))` | low | correct; efficiency note only |
| 4 | target arrays exposed by reference | informational | no live issue |
| 5 | `main.py` `--help`, `self.modules`, checkpoint refit | — | out of scope, reported elsewhere |

The bridge's row/population/target mapping is sound as used. The one thing I would
change before this is called done is item 2, because it is the kind of latent
confusion that survives review and then mis-maps a batch six weeks later.
