# Spatial/property per-feature token ablation — REVIEW

For Codex verification. Read `STATUS.md` first for the numbers; this file states what
I think the evidence supports, what it does not, and what I would need to run to
answer the actual question.

Scope executed: exactly one new training run, `orig` / seed 42 / `lstm_fusion` /
`macro none` / `token_mode=per_feature`, id `spatial-probe-perfeature-none-s42-h1`.
No macro arms, no extra seeds, no epochs grid, no other controls. The block reference
`macro-probe-block-none-s42` was reused, not re-run.

Unchanged: `experiments/neural_fusion_v3/`, `experiments/neural_fusion_v2/`,
`experiments/macro_encoders_v3/`, and every existing run directory.

---

## 1. Verified before training (preflight, 22 checks)

The question "does `per_feature` really only change the numeric token?" is answered
yes, and by execution rather than by reading the diff:

| check | result |
|---|---|
| added parameters are exactly `block.encoder.feature_tokens.{weight,bias,identity,scale}` | pass |
| removed parameters are exactly `block.encoder.static_proj.{weight,bias}` | pass |
| token count equals the 21 numeric columns | pass |
| each numeric column moves exactly its own token (marginal sentinel) | pass |
| tokens remain pairwise distinct at x=0 | pass |
| reshapes confined to `block.proj` (48,8)→(48,48) and `block.type_embedding` (9,48)→(29,48) | pass |
| LSTM / embeddings / context / temporal proj / gate keep identical structure | pass |
| weight transplant: shared tensors bit-identical, LSTM output exact | pass |

Evidence: `experiments/spatial_token_ablation_v3/preflight.py`, exit 0, 2 findings.

**Finding A (material): capacity is not matched.** 56,011 vs 51,249 parameters,
+4,762 (+9.3%). This ablation changes model size, so a score difference cannot be
attributed to the token layout alone. I flag this rather than fix it: matching capacity
would mean changing the shared trunk or the token width, which is a design decision for
Hermes2/you, not something I should silently impose on a run whose id already says
what it is.

**Finding B (minor): dead code in `install_per_feature`.** The head rebuild guards on
`hasattr(model, "head")` while the head is at `model.block.head`, so it never runs.
Harmless here because the head input width is 48 in both arms — it is `block.proj`,
not the head, that absorbs the token-count change. It would bite a control whose head
input width genuinely changes. Reported, not edited, per instruction.

---

## 2. The comparison is confounded and I will not present it as a tokenisation result

This is the part I want reviewed hardest.

Stage A early-stops on A/holdout, then stage B refits for exactly `chosen_epochs`
(`experiments/neural_fusion_v3/main.py:315`).

| | block | per_feature |
|---|---|---|
| stage-A epochs run | 12 | 6 |
| chosen epoch (best holdout) | 7 | **1** |
| stage-A best holdout RMSE | 595.62 | **589.24** |
| stage-B refit epochs | 7 | **1** |
| stage-B wall time | 61.3 s | 22.5 s |

`per_feature`'s epoch-1 holdout (589.24) is better than anything block reached in 12
epochs (best 595.62). The monitor therefore stopped immediately, and stage B — the
model that actually produces the exported predictions — was refit for **one epoch**.

So the two runs consumed different effective training budgets. The *declared* config is
identical (I verified: same `actual_manifest` including `max_epochs: 20`,
`patience: 5`, `lr 1e-3`, `weight_decay 1e-4`, `batch 512`, `clip_grad 1.0`,
`CosineAnnealingLR`; same capacity dict; same seed; same 52,310/35,882/88,192/35,158
population sizes; same parity signatures `5c27d860…` / `3b6fc524…`; identical code
fingerprint across all 39 hashed modules). What differs is the budget the monitor
spent.

The task asked me to keep "训练预算" fixed. On paper it is fixed; in effect it is not,
because early stopping is data-dependent and the two arms trigger it at different
epochs. I could have forced `chosen_epochs` to a constant, but that would mean
overriding a documented training rule — a shared-code change I am not authorised to
make mid-round. Flagging instead.

**Consequence for the headline number:** `+68.400` SGD RMSE (475.599 → 543.999) is
not a clean per-feature-vs-block estimate. It bundles tokenisation, +9.3% parameters,
and 1-vs-7 refit epochs.

---

## 3. What the error actually looks like (this part is solid)

Independent of the confound, the decomposition is unambiguous:

| | block | per_feature |
|---|---|---|
| RMSE | 475.60 | 544.00 |
| MAE | 341.27 | 390.28 |
| bias | +56.45 | +265.91 |
| error SD | 472.24 | 474.58 |
| bias² share of MSE | 1.4% | 23.9% |

MSE change +69,741, of which **+67,521 (97%) is bias²** and +2,220 (3%) is variance.
Dispersion is essentially unchanged. `per_feature` predicts a mean of 3,391.83 against
a true mean of 3,125.92 (+8.5%); block predicts 3,182.37 (+1.8%).

Monthly and horizon deltas are uniformly positive, with no month or horizon band
improving:

- monthly ΔRMSE: +49.35 to +74.36 across all 12 target months
- horizon bands: 1–3 +70.26, 4–8 +65.49, 9–12 +70.54; 13–16 is empty (n=0, the
  contract's outer window tops out at h=12)

The uniformity matters: this is not one bad month or one tail region. It is a level
shift applied everywhere.

Checkpoint reconstruction: load-only, no refit, `max |exported − rebuilt| = 4.5e-13`
SGD, reconstructed RMSE matches the recorded 543.999111. The checkpoint carries
`stage_a_state`/`stage_b_state`, `chosen_epochs`, `token_mode` and
`numeric_columns`; all 53 tensors load strictly. So the exported numbers are
reproducible from the saved artifact alone.

---

## 4. A hypothesis I tested and had to discard

Natural guess: one refit epoch means the residual head barely moved, so the prediction
sits near the anchor. I tested it and it is **wrong**.

| | block | per_feature |
|---|---|---|
| mean \|prediction − anchor\| | 591.71 | **801.17** (1.354×) |
| RMS(prediction − anchor) | 600.29 | 809.30 |
| anchor's squared error surviving | 0.4348 | 0.5688 |

`per_feature` displaces *more* from its anchor than block does, not less. Anchor alone
scores 721.30 (bias −535.26). So the head moved further and still landed worse — the
problem is the direction/level of the displacement, not an absence of it.

The correlation between the applied correction and the correction that was needed is
low for both (0.215 vs 0.196), so I explicitly do **not** claim "worse direction" as a
finding — that statistic does not discriminate between the arms. What the numbers
support is only: same dispersion, ~4.7× the bias, larger displacement from the anchor.

Whether that is one epoch of over-shooting or a genuine per-feature pathology, these
two runs cannot tell. That is the honest ceiling of this evidence.

---

## 5. What would actually answer the question

Cheapest decisive run, if you want it:

1. **Budget-matched pair.** Force the same `chosen_epochs` for both arms — either
   override the early-stopping result to a fixed 7, or let both run a fixed 7 without a
   monitor — same seed, same everything else. That removes the 7× budget difference.
2. **Capacity-matched pair.** Shrink `feature_tokens` output or widen block's
   `static_proj` so the two arms have equal parameter counts, or report the delta with
   the +4,762 explicitly carried.
3. Only then look at the score; and even then, one seed with a deterministic
   early-stopping path is exploratory, as the task says.

I did not run these: the task fixed the grid at one run, and (1) requires changing a
shared training rule that Hermes2 owns and is actively editing. If you approve the
override, it is a small change plus two runs at ~2 minutes each.

---

## 6. Gate checklist against the task text

| requirement | state |
|---|---|
| only `lstm_fusion` / macro `none` / per_feature / orig / seed 42 | done |
| run id `spatial-probe-perfeature-none-s42-h1` | done |
| check whether per_feature only changes numeric token | done — 22 checks |
| same-weights control / feature sentinel | done — transplant + marginal sentinel |
| report to Codex if the shared model needs fixing, don't edit | done — findings A/B |
| new code only under `experiments/spatial_token_ablation_v3/` | done |
| reports only under `reports/spatial_token_ablation_v3/` | done |
| don't touch neural_fusion_v3 / v2 / macro_encoders source | done |
| don't re-run the completed block baseline | done — reused |
| don't expand model/seed/grid | done — one run |
| reference fingerprint / samples / hyperparams compatible | verified identical |
| row-wise SOURCE_ROW_ID, label, horizon agreement | verified — 35,158 rows, max label delta 0.0 |
| recompute RMSE/MAE/bias from CSV | done — block 475.598688 matches recorded 475.5986877090766 |
| per-month and per-horizon deltas | done — all positive |
| checkpoint reconstruction without refit | done — 4.5e-13 |
| no epoch/correlation tuning on the outer window | honoured; `chosen_epochs` came from A/holdout as designed |
| single seed marked exploratory | stated throughout |
| keep failed artifacts | kept — see §7 |
| one training process, 2 threads | one process, `OMP_NUM_THREADS=2` |

## 7. Failed / superseded artifacts

- First launch used `-m neural_fusion_v3.main` from the project root and died
  immediately with `ModuleNotFoundError`; no run directory was created. Relaunched as
  `-m experiments.neural_fusion_v3.main` (the form recorded in the reference run's own
  `command` field). Nothing to preserve, but noted so the transcript's first attempt is
  not mistaken for a crashed run.
- `analyse.py` needed three iterations to match the real column names
  (`target_month_ref` after merge) and `verify_checkpoint.py`/`budget_confound.py`
  each needed one correction for the real `anchor_sgd` location (produced by
  `ScalerState.transform`, not carried on `StageArrays`). The final versions are what
  produced the JSON in the report directory; the intermediates wrote no partial output
  beyond a truncated `analysis.json` that was overwritten.
