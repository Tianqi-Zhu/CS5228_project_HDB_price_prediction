# Architecture changes motivated by the latest diagnostics

The highest-priority change is a property-conditioned temporal correction with an exact T0 fallback. Increasing depth or adding more economic features is not supported by the current results. These are new hypotheses to test, not established improvements. The repeatedly examined development period cannot provide untouched confirmation of them.

## Evidence guiding the design

- T0 has the strongest mean primary transformer RMSE: 461.16 SGD, compared with 465.31 for T1, 466.78 for T2, and 465.09 for T3.
- On the weighted 2,000-row diagnostic sample, disabling the temporal branch improves RMSE for every T1 and T2 seed. Average improvements are 3.51 and 2.98 SGD. This post-hoc removal is diagnostic, not a validated deployment choice.
- The T1 branch adds 32–75 SGD on average across seeds. The global sigmoid gates remain close to 0.10 even when their contributions differ materially: gate magnitude alone does not constrain output magnitude.
- In the separate 11–16-month stress slice, T1/T2 biases are +133/+102 SGD, versus T0 at -28 SGD. Stronger temporal extrapolation is therefore not automatically better.
- Geography and rental histories are useful collectively. Grouped shuffles increase sampled pipeline RMSE by approximately 20–21 SGD for geography/amenities and 84 SGD for the history bundle. The latter includes the inherited forecast baseline; these numbers do not isolate the neural encoder or Fourier embedding.
- Persistent town biases remain. The existing macro sequence is identical across rows in the fixed-origin holdout; economic effects cannot be identified from row permutations there.

## First experiment: replace independent addition with conditioned temporal attention

Keep the current T0 property/history encoder as the starting model. Form a small forecast query from its property representation, continuous horizon, known calendar, history count and staleness. Let that query cross-attend to the available monthly history tokens. Start with one cross-attention block, dimension 64 and two or four heads.

This lets the history response depend directly on the property/location representation. The existing temporal branch sees local aggregate histories but never the property encoder's hidden state. A static-conditioned forecasting architecture is motivated by [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363); the specific proposed block is an experiment for this dataset.

Predict a correction around T0 rather than a second free rent level. Initialize the final correction projection's weights and bias to zero so the initial prediction exactly equals T0. Do not simultaneously zero both that output projection and an upstream multiplicative gate, which would block useful initial gradients. Penalize the actual standardized correction's squared magnitude; a gate-only penalty can be defeated by increasing the head's output scale.

Initially freeze T0 while evaluating the new head, then separately test limited joint fine-tuning. Choose correction settings on earlier rolling-origin forecasts, and include the disabled correction as an explicit candidate. Corrections trained on out-of-time base-model errors require matching origin-specific base models; do not treat in-sample residuals as equivalent evidence.

## Second experiment: replace redundant rent trajectories with innovations

Add a compact sequence of unexpected rent movements: observed adjusted monthly rent minus a prediction made using strictly preceding information. Pool national, flat-type and area/type innovations with explicit counts and missingness. Keep recent levels as a small control rather than duplicating many absolute histories.

Use a shared horizon-conditioned head, direct correction shrinkage, and an earlier-selected damping rule. Train and select on mature targets through horizons 1–16, with a separate long-horizon comparison. Do not assume future growth must continue or force every long-horizon correction to zero. This specifically tests the upward-drift failure observed in the current sequence branch.

## Third experiment: add comparable-transaction memory

Retrieve 32–64 strictly available earlier rentals, mixing same-building, nearby similar-property and broader same-type evidence. A historical rental becomes a token with property attributes, date and observed rent or a causally computed market-adjusted residual.

The target-property query attends to these tokens with a small per-head bias based on relative east/west and north/south displacement, distance, time gap, area difference, same type and same building. This encodes which *transactions* are useful rather than attending only between feature columns. [TabR](https://arxiv.org/abs/2307.14338) motivates retrieval-based tabular learning; it does not establish that the proposed spatial extension will improve this project.

Use origin-filtered candidate banks, exclude target/future labels, deduplicate candidates, and provide support/staleness flags plus a T0 fallback. Compare against the same model with simple weighted comparable averages. The earlier fixed comparable graph did not produce stable gains, so richer attention needs to earn its complexity.

## Economic and geographic complexity should remain controlled

COE A/C is an optional small context, tested after rent-only conditioned attention. Stocks and company/worksite interactions remain secondary. The latest masking results show learned dependence on COE, not consistent incremental benefit from adding it. Only 51 distinct labeled months constrain macro learning.

Retain the present coordinate and identity encodings initially. If sparse-building errors persist, test hierarchical town/area/block embeddings with count-based shrinkage or building-ID dropout. First isolate existing Fourier features in an ablation; grouped geography importance and a smooth rental map do not prove that Fourier encoding itself helps.

## Minimal comparison order

1. Unchanged T0.
2. T0 plus zero-initialized, property-conditioned rent-history attention.
3. Same architecture with causal innovation inputs.
4. Same model plus optional COE.
5. Best earlier-selected design plus comparable-transaction memory and relative spatial/time bias.

Keep identical forecast cases, information cutoffs, three declared seeds and comparable training budgets. Report individual-rent SGD RMSE, bias, town/month residuals, and horizons 11–16. Use several origins and distinguish overlapping windows. Add complexity only when it improves the matched predecessor consistently; no automatic submission replacement.
