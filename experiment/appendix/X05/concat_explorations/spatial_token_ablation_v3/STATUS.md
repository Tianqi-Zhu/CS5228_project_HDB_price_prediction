# Spatial/property per-feature token ablation — STATUS

**State: run complete, analysis complete, result is a confounded comparison and must
not be read as an answer about tokenisation.**

| item | value |
|---|---|
| run id | `spatial-probe-perfeature-none-s42-h1` |
| control / tokens / macro | `lstm_fusion` / `per_feature` / `none` |
| split / seed | `orig` / 42 |
| v2 contract | `fusion-contract-v2.1.0__43867d0ba111` (same as the block reference) |
| parameters | 56,011 (block reference: 51,249; **+4,762**) |
| token count | 28 (8 block + 21 numeric per-feature replacings 1 pooled − 1) |
| RMSE | **543.999** SGD |
| reference RMSE | **475.599** SGD (`macro-probe-block-none-s42`) |
| delta | **+68.400** SGD (worse) |
| chosen epochs | **1** (block: 7) |
| stage-A epochs run | **6** (block: 12) |
| wall clock | 123 s |

## What was confirmed before spending the run

`experiments/spatial_token_ablation_v3/preflight.py`, 22 checks + 2 findings, exit 0:

- `per_feature` adds **only** `block.encoder.feature_tokens.*` and removes **only**
  `block.encoder.static_proj.*`. Reshapes are confined to `block.proj`,
  `block.type_embedding` and nothing in the head.
- Each of the 21 numeric columns moves **exactly one** token and no other
  (marginal-response sentinel test), so the "21 values → 21 tokens" claim is real.
- Identical inputs still yield pairwise-distinct tokens at x=0 (min distance > 0).
- Weight transplant: every shared tensor that keeps its shape is bit-identical, and
  the LSTM output matches exactly after transplanting from `block` into
  `per_feature`. The non-numeric pathway is genuinely unchanged.
- LSTM / embeddings / context projection / temporal projection / gate all keep
  identical parameter structure.

So the mechanism the user asked about is working as intended. The score is the
problem, and it is not the mechanism.

## Two findings from preflight (reported, not fixed)

1. **Capacity is not matched.** +4,762 parameters (+9.3%). A score difference between
   these arms therefore cannot be attributed to tokenisation alone — it is a
   different model size as well. This is the single most important caveat for
   interpreting the result.
2. **`install_per_feature`'s head rebuild is dead code.** It guards on
   `hasattr(model, "head")`, but for these controls the head lives at
   `model.block.head`, so the branch never executes. Harmless for `lstm_fusion`
   because the head input width is 48 in both arms and is set upstream by `block.proj`
   — which *is* rebuilt, going `(48, 8)` → `(48, 48)`. The head itself keeps
   `(48, 48)` unchanged, which is why the dead branch does not matter here. It would
   not be harmless for a control whose head input width actually changes. Reported to
   Codex; not edited.

## Training-budget confound (the decisive issue)

Stage A early-stops on the A/holdout RMSE, then stage B refits for exactly
`chosen_epochs` (`experiments/neural_fusion_v3/main.py:315`,
`fit_fixed_epochs(model_b, tensors_b_fit, config, chosen)`).

| | block | per_feature |
|---|---|---|
| stage-A epochs run | 12 | 6 |
| best holdout epoch | 7 (595.62) | **1 (589.24)** |
| chosen epochs | 7 | **1** |
| stage-B refit epochs | 7 | **1** |
| stage-B seconds | 61.3 | 22.5 |

`per_feature` had a *better* holdout RMSE at epoch 1 than block ever reached (589.24
vs 595.62), so the monitor fired immediately and stage B got **one** epoch. The stage-B
model is therefore barely trained, and the two runs did not receive the same effective
budget. Any RMSE gap mixes the token mode with a 7× difference in B refit length.

This is why the task's "same budget" requirement is not satisfied by these two runs,
even though the *declared* hyperparameters are byte-identical (verified: same
`actual_manifest`, same `max_epochs: 20`, same `patience: 5`, same capacity, same
optimiser, same batch size, same seed, same 88,192 B/fit rows, same parity
signatures `5c27d860…` / `3b6fc524…`). The budget that was *actually spent* differs
because the early-stopping monitor chose differently.

## Evidence about the direction of the error

`budget_confound.py` decomposed the gap:

| | block | per_feature |
|---|---|---|
| RMSE | 475.60 | 544.00 |
| bias | +56.45 | **+265.91** |
| error SD | 472.24 | 474.58 |
| bias share of MSE | 1.4% | **23.9%** |

`per_feature` has essentially the **same error dispersion** (474.58 vs 472.24) but a
**4.7× larger bias**. 97% of the MSE increase comes from bias², only 3% from variance.
Consistent with that, its mean prediction is +8.5% over the actual mean versus +1.8%
for block (3391.83 vs 3182.37 on a mean actual of 3125.92).

I tested the natural "under-trained head barely moves" hypothesis and **it was
refuted**: `per_feature` applies a *larger* mean displacement from its anchor (801.17
vs 591.71, ratio 1.354). So the head did move; it moved further, in a direction that
lands above the label. The 0.20 correlation between applied and needed correction is
low for *both* runs, so that statistic cannot support a claim about direction quality.

Net: the run is consistent with "one epoch of stage-B refit produced an
over-corrected level", but this is a hypothesis, not something these two runs can
separate from the capacity change.

## What this does and does not establish

- **Does**: `per_feature` genuinely turns 21 spatial/property values into 21 tokens,
  changing only the numeric token block; and at these settings it scores 68.4 SGD
  worse, almost entirely through level bias, not dispersion.
- **Does not**: say whether the tokenisation is better or worse. The comparison is
  confounded by (a) +9.3% parameters and (b) 1 vs 7 stage-B epochs. Neither is
  controlled.

## Files

- `experiments/spatial_token_ablation_v3/preflight.py` — 22 checks, 2 findings
- `experiments/spatial_token_ablation_v3/analyse.py` — provenance, row agreement,
  metric recomputation, monthly/horizon deltas
- `experiments/spatial_token_ablation_v3/verify_checkpoint.py` — load-only
  reconstruction (max |exported − rebuilt| = 4.5e-13 SGD) + bias/variance split
- `experiments/spatial_token_ablation_v3/budget_confound.py` — anchor-decomposition
  test of the under-training hypothesis
- `reports/spatial_token_ablation_v3/analysis.json`, `monthly_deltas.csv`,
  `horizon_band_deltas.csv`, `checkpoint_reconstruction.json`, `budget_confound.json`

Nothing under `experiments/neural_fusion_v3/`, `neural_fusion_v2/`,
`macro_encoders_v3/` or any existing run directory was modified.
