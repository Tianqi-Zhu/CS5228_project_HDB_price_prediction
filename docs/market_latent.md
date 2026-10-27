# Rental-market latent vectors

`analysis/experiments/market_latent.py` implements small RNN and LSTM encoders.
It does not modify the presentation or existing models.

## Run

Install `requirements-market-latent.txt` in the project's virtual environment.
On this Mac, the existing LightGBM dependency needs the Miniconda OpenMP library:

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib MPLBACKEND=Agg \
  .venv/bin/python -m analysis.experiments.market_latent \
  --output-dir reports/experiments/market_latent/new-run
```

Use a new/empty output directory. Defaults: seeds 42, 43, 44, six-month lookback,
8 recurrent hidden units, 3 latent coordinates, maximum 300 epochs, Adam learning
rate 0.003, weight decay 0.01, gradient norm clipping at 1 and patience 25.

## Inputs and training task

Monthly inputs contain nine COE/SG-stock indicators, median rent, rent IQR,
mean area, six flat-type shares, and log median-rent growth. Rent aggregates are
computed independently within each calendar month. Shares and area provide
composition context; the median itself is **not** a mix-adjusted rent index.
Record counts are not interpreted as market transaction volume.

For prediction month t, a window contains only t-6 through t-1. A single-layer
RNN or LSTM produces an 8-dimensional final hidden state, projected to three
latent coordinates. A linear head predicts log median-rent growth in t.
The training loss is squared error on growth standardized using fitting targets.
Missing economic values are median-imputed from encoder fitting windows only.

## Time protocol

1. Initial encoder fit targets: Jul 2021–Sep 2022; tune targets: Oct–Dec 2022.
2. Select epochs on those three tuning months, then refit through Dec 2022.
3. Freeze encoder, imputer, scaler and target scaling. Export vectors only from
   Jan 2023 onward, so all downstream rows occur after encoder training.
4. Fit the rental regressor on Jan 2023–Mar 2024 and score Apr 2024–Mar 2025.
5. Every rental variant uses exactly the same rows and HistGradientBoosting
   settings. Baseline = property/time/town/block/amenities. Compare adding
   prior-month raw market features, three PCA components of the entire six-month
   window, and RNN/LSTM vectors for each seed. PCA is fitted before Jan 2023 too.

This is a **monthly-updated** experiment. It assumes actual rents and economic
summaries through the previous month are available at each new prediction date.
Actual prior validation-month rents may enter later-month inputs, but encoder
and rental model weights remain frozen. It is not fixed-origin batch forecasting;
these metrics must not be ranked against the earlier fixed-origin scores.
Publication lags are not provided and are not established by the month labels.

There are only 18 pre-cutoff encoder windows (15 fitting, 3 tuning), with overlapping
histories. This is an exploratory small-data experiment. Repeated seeds are not
independent time backtests. Do not assign economic meanings to individual latent
coordinates without evidence, or compare coordinates across different seeds.

## Outputs and reuse

- `monthly_inputs.csv`: monthly audit table, including evaluation-period observations.
- `rnn_42_latent.csv`, etc.: prediction month, 3 coordinates, last input month.
- `*_epochs.csv`: fitting/tuning losses and epoch selection trace.
- `*.pt`, `*_preprocessing.joblib`: frozen encoder and preprocessing metadata.
- `*_rent.joblib`: fitted rental model, feature order, categorical dtypes.
- `pca.joblib`: fitted full-window PCA baseline.
- `rental_metrics.csv`, `monthly_rental_metrics.csv`, `validation_predictions.csv`.
- `monthly_growth_predictions.csv`: auxiliary market forecast and zero-growth baseline.
- `results.json`: protocol, parameters, source hashes, selected epochs, limitations.

To reuse a trusted checkpoint (never load joblib from an untrusted source):

```python
import joblib
import torch
from analysis.experiments.market_latent import MarketEncoder, encode

meta = joblib.load(output / 'lstm_42_preprocessing.joblib')
net = MarketEncoder(len(meta['features']), meta['kind'], meta['hidden'], meta['latent'])
net.load_state_dict(torch.load(output / 'lstm_42.pt', weights_only=True))
net.eval()
# history: numpy array [number_of_predictions, 6, number_of_features]
# Columns must follow meta['features']; each window must end BEFORE its prediction month.
growth, latent = encode(net, meta['preprocessor'], history,
                        meta['target_mean'], meta['target_std'])
```

No fixed-origin test vectors or submission are created. For those, specify how to
forecast all missing future rents, economics and composition features first.

## Checks

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib MPLBACKEND=Agg \
  .venv/bin/python -m unittest tests.test_market_latent
```

Tests perturb same/future observations to verify strict prior-month windows,
perturb post-cutoff inputs and targets to verify encoder isolation, and check
prediction/vector equality after checkpoint reload for both architectures.

## Compare lookback lengths

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib MPLBACKEND=Agg \
  .venv/bin/python -m analysis.experiments.market_window_search \
  --output-dir reports/experiments/market_window_search/new-run
```

This compares 3/6/9/12 months with the same 19 aggregated inputs, hidden size,
latent dimension, and seeds 42/43/44. Stock features remain aggregate indicators,
not individual-stock inputs. All encoders share Jan–Dec 2022 target months (9
initial fitting and 3 epoch-selection months; 12 refitting windows), so longer
histories do not reduce the number of target samples in the comparison.

Rental regressors fit Jan–Sep 2023 and choose each architecture's lookback on
Oct 2023–Mar 2024 mean RMSE across seeds. Only after writing selected_windows.json,
selected lengths are refit downstream through Mar 2024 and evaluated Apr 2024–Mar
2025. The outer period was previously examined elsewhere in this project; it is
not a pristine test set. It does not choose the window in this search.

The selected feature CSVs use fixed seed 42, not a winning validation seed. Latent
coordinates cannot be averaged across seeds. Existing six-month CSVs are not
overwritten. A selected length means best among this small grid on the selection
period, not a proven global optimum or proof of improvement over no latent vector.

## Earlier encoder cutoff / wider feature coverage

```sh
DYLD_FALLBACK_LIBRARY_PATH=/opt/miniconda3/lib MPLBACKEND=Agg \
  .venv/bin/python -m analysis.experiments.market_early_cutoff \
  --output-dir reports/experiments/market_early_cutoff/new-run
```

Keep the selected three-month lookback, and compare feature availability from
Jan 2022 versus Jul 2022. For Jan 2022, initial encoder target months are Apr–Sep
2021 (6 windows); tune on Oct–Dec 2021 (3 windows), refit on all 9, then freeze.
For Jul 2022, initial fit has 12 windows, tuning Apr–Jun 2022 adds 3, and refitting
uses 15. The rental regressor can now use the extra earlier rows. Scores use the
same Apr 2024–Mar 2025 rental rows and include a separate baseline for each fit
period. This changes two factors (encoder history and rental training length),
so cross-cutoff changes cannot be attributed solely to encoder quality.

Both cutoffs are exported with seed 42, not selected using the reused holdout.
Files `*_features_from_YYYY-MM.csv` contain only the monthly join key and three
latent features. Companion `*_windows_from_YYYY-MM.csv` records input and encoder
time boundaries. Never join vectors from different encoders into one feature
column over time: their latent coordinate systems need not align.
