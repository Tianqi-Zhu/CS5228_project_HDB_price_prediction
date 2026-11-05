# Rent, stock and COE temporal processing ablation

Prespecified before running the new models. Existing reports have already examined
June 2024–March 2025, so this is development evidence, not an untouched final test.

- Responses: Singapore arithmetic mean monthly HDB rent and a fixed 2021-only
  property-mix-adjusted arithmetic mean. Use all 150,000 supplied rental rows;
  there are only 51 distinct monthly observations, not 150,000 time steps.
- Primary stock input: the existing historical-cap-seeded 22-stock Singapore
  return proxy. It is not official STI. Reconstruct it from raw prices and the
  existing December 2020 holdings table. Never use current weights or test rents.
- COE inputs: category A/C median premium log levels and monthly log changes.
  The category choice follows the earlier economic study, not new correlations.
- Four information sets: rent only; rent + stocks; rent + COE; rent + both.
  Model returns/changes rather than correlating trending raw price levels.
- Processing: persistence, three-month momentum, EWMA, Ridge on latest/3/6/12
  month summaries, Ridge on ordered lags, matched flattened-input MLP, and LSTM.
  Ordered-lag/MLP/LSTM windows are 3, 6 and 12 months. Calendar sine/cosine are
  common to fitted models. Small fixed networks: hidden size 8, 100 epochs,
  AdamW lr .01, weight decay .05, gradient norm 1; no outer-period tuning.
- Predict 1–10 month changes directly, using ten outputs for neural models and
  ten Ridge heads. Targets are average monthly log growth over each horizon.
  Mask every training label whose endpoint exceeds the current forecast origin.
  No recursive insertion of realized future rents/stocks/COE. Fit imputers and
  scales from mature training examples only. All variants share training-origin
  start January 2022, so long windows are not scored on a different sample.
- Monthly-refreshed evaluation: rolling horizons 1/3/6, target months July
  2023–May 2024 for selection and June 2024–March 2025 for later comparison.
  Each forecast only sees observations through its own origin. Monthly rent
  availability at month end is an assumption; run an extra-month-delay check.
- Fixed-origin evaluation: ten-month paths from January/April/July 2023 choose
  models (all endpoints through May 2024); May 2024 predicts June 2024–March
  2025. This evaluates deployment without future realized macro observations.
  Overlapping earlier paths are not independent replicates.
- Select by development MSE normalized to persistence separately by target,
  scenario, input set and processing family; rolling horizons have equal weight.
  Report all candidates and the forced LSTM result even if simpler methods win.
  Sensitivities of development-selected candidates: seeds 43/44, equal-weight
  matched-stock index, and an extra month of information delay. No reselection.
- Individual-rental transfer: freeze the existing 29-feature HistGradientBoosting
  control at June 2023, December 2023 and May 2024 cutoffs. Add a national-index
  movement correction. Fit a shrinkage coefficient in [0,1] on earlier row errors
  only, then freeze it for June 2024–March 2025. Compare persistence, the selected
  rent-only processor, the selected joint processor and joint LSTM. Report this
  as a controlled tree transfer, not a measured improvement to TabPFN.
- Record source/code hashes, training-label cutoffs, predictions, all scores,
  source coverage, model sizes and timings. Paired three-month block intervals
  are conditional on fitted/selected models and omit selection uncertainty.
  Add tests for future-data invariance and maturity masks before full execution.
- Keep existing submissions and experiments unchanged. No Kaggle test labels
  or locally unobservable test scores enter selection.

Method references: [PyTorch LSTM](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html),
[rolling-origin evaluation](https://otexts.com/fpp3/tscv.html).
