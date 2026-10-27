"""Frozen RNN/LSTM rental-market encoders; monthly-updated, NOT batch forecasting.

Run: python -m analysis.experiments.market_latent --output-dir reports/experiments/market_latent
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import torch
import sklearn
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analysis.pipeline.config import AUX, DATA, EXPERIMENT_REPORTS
from analysis.pipeline.data import prepare_main, sha256, check_output_path
from analysis.pipeline.evaluation import metrics
from analysis.studies.economic_signals import monthly_market, SETTINGS
from analysis.experiments.auxiliary_ablation import BASE, BLOCK, AMENITIES, model_inputs
from analysis.pipeline.auxiliary import build_auxiliary_features

ECONOMIC = ['COE_A_PREMIUM', 'COE_C_PREMIUM', 'COE_A_GROWTH', 'COE_C_GROWTH',
            'COE_A_BID_PRESSURE', 'COE_C_BID_PRESSURE', 'SG_STOCK_RETURN',
            'SG_STOCK_RETURN_3M', 'SG_STOCK_VOL_3M']
TYPES = ['1-room', '2-room', '3-room', '4-room', '5-room', 'executive']


def monthly_panel(rent, market):
    """Month-local summaries; no backward filling or full-sample target adjustment."""
    f = prepare_main(rent)
    f['month'] = pd.PeriodIndex(f.RENT_APPROVAL_DATE, freq='M')
    g = f.groupby('month')
    p = g.MONTHLY_RENT.agg(rent_median='median', rent_q25=lambda x: x.quantile(.25),
                         rent_q75=lambda x: x.quantile(.75))
    p['rent_iqr'] = p.pop('rent_q75') - p.pop('rent_q25')
    p['area_mean'] = g.FLOOR_AREA_SQM.mean()
    shares = pd.crosstab(f['month'], f.FLAT_TYPE_CLEAN, normalize='index')
    for name in TYPES:
        p['share_' + name] = shares[name] if name in shares else 0.0
    p = p.reindex(pd.period_range(p.index.min(), p.index.max(), freq='M'))
    if p.rent_median.isna().any() or (p.rent_median <= 0).any():
        raise ValueError('Rental months must be contiguous with positive medians')
    p['rent_growth'] = np.log(p.rent_median).diff()
    return p.join(market[ECONOMIC]).replace([np.inf, -np.inf], np.nan)


def windows(panel, lookback):
    """The vector for month t sees only t-lookback,...,t-1."""
    if lookback < 1 or len(panel) <= lookback:
        raise ValueError('Insufficient months for positive lookback')
    x = np.stack([panel.iloc[i-lookback:i].to_numpy(float) for i in range(lookback, len(panel))])
    return x, panel.rent_growth.iloc[lookback:].to_numpy(float), panel.index[lookback:]


class MarketEncoder(nn.Module):
    def __init__(self, n_features, kind='lstm', hidden=8, latent=3):
        super().__init__()
        if kind not in ('rnn', 'lstm'):
            raise ValueError('kind must be rnn or lstm')
        self.recurrent = (nn.RNN if kind == 'rnn' else nn.LSTM)(n_features, hidden, batch_first=True)
        self.project = nn.Linear(hidden, latent)
        self.head = nn.Linear(latent, 1)

    def forward(self, x):
        output, _ = self.recurrent(x)
        z = self.project(output[:, -1])
        return self.head(z).squeeze(-1), z


def train_encoder(x, y, months, kind, seed, cutoff, tune_start, epochs=300):
    """Select epochs before cutoff, refit on pre-cutoff data, then freeze."""
    fit = months < pd.Period(tune_start, 'M')
    tune = (months >= pd.Period(tune_start, 'M')) & (months < pd.Period(cutoff, 'M'))
    pre = months < pd.Period(cutoff, 'M')
    if fit.sum() < 6 or tune.sum() < 2:
        raise ValueError('Need >=6 encoder fitting and >=2 tuning windows')

    def initialize(mask):
        torch.manual_seed(seed)
        preprocessor = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler())
        preprocessor.fit(x[mask].reshape(-1, x.shape[-1]))
        values = preprocessor.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        target_mean, target_std = float(y[mask].mean()), max(float(y[mask].std()), 1e-8)
        net = MarketEncoder(x.shape[-1], kind)
        return net, preprocessor, torch.tensor(values, dtype=torch.float32), target_mean, target_std

    net, _, xx, mu, sd = initialize(fit)
    yy = torch.tensor((y-mu)/sd, dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=.003, weight_decay=.01)
    best, stale, selected, history = float('inf'), 0, 1, []
    for epoch in range(1, epochs+1):
        net.train(); optimizer.zero_grad()
        loss = nn.functional.mse_loss(net(xx[fit])[0], yy[fit]); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
        net.eval()
        with torch.no_grad():
            score = float(nn.functional.mse_loss(net(xx[tune])[0], yy[tune]))
        history.append({'epoch': epoch, 'train_mse_scaled': float(loss.detach()), 'tune_mse_scaled': score})
        if score < best - 1e-5:
            best, stale, selected = score, 0, epoch
        else:
            stale += 1
        if stale >= 25:
            break
    net, prep, xx, mu, sd = initialize(pre)
    yy = torch.tensor((y-mu)/sd, dtype=torch.float32)
    optimizer = torch.optim.Adam(net.parameters(), lr=.003, weight_decay=.01)
    for _ in range(selected):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(net(xx[pre])[0], yy[pre]); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0); optimizer.step()
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net, prep, mu, sd, selected, history


def encode(net, prep, x, mu, sd):
    xx = prep.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    with torch.no_grad():
        pred, z = net(torch.tensor(xx, dtype=torch.float32))
    return pred.numpy()*sd+mu, z.numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=EXPERIMENT_REPORTS/'market_latent')
    parser.add_argument('--lookback', type=int, default=6)
    parser.add_argument('--encoder-cutoff', default='2023-01')
    parser.add_argument('--encoder-tune-start', default='2022-10')
    parser.add_argument('--validation-start', default='2024-04')
    parser.add_argument('--data-end', help='Exclusive last rental month, e.g. 2024-04 for window selection')
    parser.add_argument('--encoder-target-start', help='Common first target month when comparing lookbacks')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args(argv)
    if args.epochs < 1 or not args.seeds:
        parser.error('Positive epochs and at least one seed required')
    if not args.encoder_tune_start < args.encoder_cutoff < args.validation_start:
        parser.error('Require tuning < encoder cutoff < rental validation start')
    out = check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory to preserve previous experiments')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    sources = [DATA/'train.csv', *sorted(AUX.glob('*.csv'))]
    hashes = {str(p): sha256(p) for p in sources}
    raw = pd.read_csv(DATA/'train.csv')
    if args.data_end:
        raw = raw[raw.RENT_APPROVAL_DATE < args.data_end].copy()
    market = monthly_market(pd.read_csv(AUX/'sg-coe-prices.csv'), pd.read_csv(AUX/'sg-stock-prices.csv'))
    panel = monthly_panel(raw, market)
    panel.to_csv(out/'monthly_inputs.csv', index_label='month')
    x, y, months = windows(panel, args.lookback)
    if args.encoder_target_start:
        keep = months >= pd.Period(args.encoder_target_start, 'M')
        x, y, months = x[keep], y[keep], months[keep]
    pre = months < pd.Period(args.encoder_cutoff, 'M')
    downstream = months >= pd.Period(args.encoder_cutoff, 'M')
    validation = months >= pd.Period(args.validation_start, 'M')
    if not validation.any() or not (downstream & ~validation).any():
        parser.error('Empty downstream fitting or validation period')
    frame = build_auxiliary_features(raw)
    frame['month'] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq='M')
    frame = frame[frame.month.isin(months[downstream])].copy()
    core = BASE + BLOCK + AMENITIES
    variants = {'spatial_baseline': pd.DataFrame(index=months)}
    variants['raw_market_lag1'] = pd.DataFrame(x[:, -1, :], index=months,
                                            columns=['market_'+c for c in panel.columns])
    # PCA gets exactly the same flattened history as the recurrent encoders.
    pca = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), PCA(n_components=3))
    flat = x.reshape(len(x), -1)
    pca.fit(flat[pre])
    variants['pca_history'] = pd.DataFrame(pca.transform(flat), index=months, columns=['pca_1','pca_2','pca_3'])
    joblib.dump(pca, out/'pca.joblib')
    forecast = pd.DataFrame({'month': months.astype(str), 'actual_growth': y,
                             'persistence_growth': np.zeros(len(y))})
    encoder_meta = []
    for kind in ('rnn', 'lstm'):
        for seed in args.seeds:
            name = f'{kind}_{seed}'
            print('Training', name, flush=True)
            net, prep, mu, sd, selected, history = train_encoder(
                x, y, months, kind, seed, args.encoder_cutoff, args.encoder_tune_start, args.epochs)
            pred, z = encode(net, prep, x, mu, sd)
            # Export only vectors after the encoder's training/selection period.
            latent = pd.DataFrame(z, index=months, columns=[f'{name}_z{i+1}' for i in range(3)])
            variants[name] = latent
            exported = latent.loc[downstream].copy()
            exported['input_end_month'] = (exported.index-1).astype(str)
            exported.to_csv(out/f'{name}_latent.csv', index_label='prediction_month')
            pd.DataFrame(history).to_csv(out/f'{name}_epochs.csv', index=False)
            torch.save(net.state_dict(), out/f'{name}.pt')
            joblib.dump({'preprocessor': prep, 'target_mean': mu, 'target_std': sd,
                         'features': list(panel.columns), 'lookback': args.lookback,
                         'kind': kind, 'hidden': 8, 'latent': 3,
                         'encoder_cutoff': args.encoder_cutoff}, out/f'{name}_preprocessing.joblib')
            forecast[name] = pred
            encoder_meta.append({'model': name, 'selected_epochs': selected,
                                 'fit_windows': int(pre.sum()),
                                 'outer_growth_rmse': float(np.sqrt(np.mean((pred[validation]-y[validation])**2)))})
    forecast.loc[downstream].to_csv(out/'monthly_growth_predictions.csv', index=False)
    scores, monthly_scores = [], []
    predictions = frame[['RENT_APPROVAL_DATE','MONTHLY_RENT']].copy()
    predictions['source_row'] = frame.index
    for name, extra in variants.items():
        print('Rental comparison', name, flush=True)
        enriched = frame.join(extra, on='month')
        fit = enriched[enriched.RENT_APPROVAL_DATE < args.validation_start]
        val = enriched[enriched.RENT_APPROVAL_DATE >= args.validation_start]
        features = core + list(extra.columns)
        xf, xv, _ = model_inputs(fit, val, features)
        model = HistGradientBoostingRegressor(**SETTINGS)
        with threadpool_limits(limits=2):
            model.fit(xf, fit.MONTHLY_RENT)
            pred = model.predict(xv)
        scores.append({'variant': name, 'n_fit': len(fit), 'n_validation': len(val),
                       **metrics(val.MONTHLY_RENT.to_numpy(), pred)})
        predictions.loc[val.index, name] = pred
        e = pd.DataFrame({'month': val.RENT_APPROVAL_DATE.to_numpy(), 'actual': val.MONTHLY_RENT.to_numpy(), 'pred': pred})
        for month, g in e.groupby('month'):
            monthly_scores.append({'variant': name, 'month': month, **metrics(g.actual.to_numpy(),g.pred.to_numpy())})
        joblib.dump({'model': model, 'features': features,
                     'category_dtypes': {c: xf[c].dtype for c in xf if isinstance(xf[c].dtype,pd.CategoricalDtype)}}, out/f'{name}_rent.joblib')
    scored = pd.DataFrame(scores)
    scored['delta_rmse_vs_baseline'] = scored.rmse_sgd - scored.iloc[0].rmse_sgd
    scored.to_csv(out/'rental_metrics.csv', index=False)
    pd.DataFrame(monthly_scores).to_csv(out/'monthly_rental_metrics.csv', index=False)
    predictions[predictions.RENT_APPROVAL_DATE >= args.validation_start].to_csv(out/'validation_predictions.csv', index=False)
    if hashes != {str(p):sha256(p) for p in sources}:
        raise AssertionError('Source data changed')
    result = {'scenario': 'monthly-updated: actual prior-month rental summaries and economics are assumed available',
              'not_valid_for': 'fixed-origin multi-month test submission',
              'configuration': vars(args) | {'output_dir': str(out)}, 'monthly_features': list(panel.columns),
              'encoder_results': encoder_meta,
              'persistence_outer_growth_rmse': float(np.sqrt(np.mean(y[validation]**2))),
              'encoder_frozen_before_downstream_fit': True,
              'downstream_fit_start': args.encoder_cutoff, 'raw_hashes': hashes,
              'rental_model_settings': SETTINGS,
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
              'limitations': ['Very few overlapping monthly windows; exploratory results.',
                              'Monthly medians are not mix-adjusted; composition features only supply context.',
                              'All variants use reduced fitting period; not comparable to earlier full-training scores.',
                              'Prior-month availability is assumed; actual publication delays are not supplied.',
                              'No unseen-future test latent vectors exported; future actual rents are unavailable.']}
    (out/'results.json').write_text(json.dumps(result, indent=2))
    print(scored.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
