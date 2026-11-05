"""Small-sample, origin-safe temporal processing ablation; no submission writes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.experiments.stock_index_regions import construct_indices, regional_rent_panel

TARGETS = ('mean_rent', 'adjusted_mean_rent')
GROUPS = ('rent', 'rent_stock', 'rent_coe', 'rent_stock_coe')
HORIZONS = 10
TRAIN_START = 12  # January 2022, common even for the 3-month window.
OUTER_START = '2024-06'
SEED = 42
EPOCHS = 100
OUT = REPORTS / 'temporal_sequence_ablation'


@dataclass(frozen=True)
class Spec:
    method: str
    group: str = 'rent'
    window: int = 12

    @property
    def name(self):
        return f'{self.method}__{self.group}__w{self.window}'


def specifications():
    result = [Spec(m) for m in ('persistence', 'momentum3', 'ewma')]
    for group in GROUPS:
        result.append(Spec('ridge_summary', group))
        result.extend(Spec(method, group, width) for method in ('ridge_lags', 'mlp', 'lstm')
                      for width in (3, 6, 12))
    return result


def build_panel(raw, stocks, coe, weights, blocks):
    regional = regional_rent_panel(raw, blocks)
    panel = regional[regional.region.eq('Singapore')].set_index('month')[list(TARGETS) + ['n']]
    panel.index = pd.PeriodIndex(panel.index, freq='M')
    levels, _, _, diagnostics = construct_indices(stocks, weights, str(panel.index.max()))
    panel = panel.join(levels)
    c = coe.copy()
    c['month'] = pd.to_datetime(c.YEAR.astype(str) + '-' + c.MONTH, format='%Y-%b').dt.to_period('M')
    c['premium'] = pd.to_numeric(c.QUOTA_PREMIUM.astype(str).str.replace(r'[$,]', '', regex=True))
    wide = c.groupby(['month', 'CATEGORY']).premium.median().unstack()
    for category in ('a', 'c'):
        panel[f'coe_{category}'] = wide[category].reindex(panel.index)
    if panel.isna().any().any() or (panel.drop(columns='n') <= 0).any().any():
        raise ValueError('Incomplete/nonpositive monthly source panel')
    return panel, diagnostics


def channels(panel, target, group, index='cap_seeded_22'):
    values = {'rent_growth': 100 * np.log(panel[target]).diff()}
    if 'stock' in group:
        values['stock_return'] = 100 * np.log(panel[index]).diff()
    if 'coe' in group:
        for category in ('a', 'c'):
            values[f'coe_{category}_log_level'] = np.log(panel[f'coe_{category}'])
            values[f'coe_{category}_growth'] = 100 * np.log(panel[f'coe_{category}']).diff()
    return pd.DataFrame(values, index=panel.index)


def examples(panel, target, spec, origin, *, index='cap_seeded_22', delay=0):
    """Targets after origin are masked, including the query's entire target row."""
    # Truncation happens before transformations, including price differencing.
    past = panel.iloc[:origin + 1].copy()
    data = channels(past, target, spec.group, index).to_numpy(float)
    level = np.log(past[target].to_numpy(float))
    source_origins = np.arange(TRAIN_START + delay, origin + 1)
    sequences, calendar, labels, masks = [], [], [], []
    for s in source_origins:
        cutoff = s - delay
        sequences.append(data[cutoff - spec.window + 1:cutoff + 1])
        hs = np.arange(1, HORIZONS + 1)
        phase = (past.index[s].month + hs) * 2 * np.pi / 12
        calendar.append(np.stack([np.sin(phase), np.cos(phase)], axis=1))
        y, mask = np.zeros(HORIZONS), np.zeros(HORIZONS, dtype=bool)
        # Anchor remains the last AVAILABLE rent; target distance includes delay.
        for h in hs:
            if s + h <= origin - delay:
                y[h - 1] = 100 * (level[s + h] - level[cutoff]) / (h + delay)
                mask[h - 1] = True
        labels.append(y)
        masks.append(mask)
    return (np.asarray(sequences), np.asarray(calendar), np.asarray(labels),
            np.asarray(masks), source_origins)


def standardize(sequence, mask):
    """Only mature training examples determine missing-value fills and scales."""
    train = sequence[mask.any(axis=1)]
    mean = np.nanmean(train, axis=(0, 1))
    sd = np.nanstd(train, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1)
    clean = np.where(np.isfinite(sequence), sequence, mean)
    return (clean - mean) / sd


def summary_inputs(sequence):
    pieces = [sequence[:, -1]]
    for width in (3, 6, 12):
        tail = sequence[:, -min(width, sequence.shape[1]):]
        pieces += [tail.mean(axis=1), tail.std(axis=1)]
    return np.concatenate(pieces, axis=1)


class SequenceNetwork(nn.Module):
    def __init__(self, method, width, features):
        super().__init__()
        self.method = method
        if method == 'lstm':
            self.encoder = nn.LSTM(features, 8, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(width * features, 8), nn.Tanh())
        self.head = nn.Linear(8 + 2 * HORIZONS, HORIZONS)

    def forward(self, x, calendar):
        if self.method == 'lstm':
            _, (hidden, _) = self.encoder(x)
            state = hidden[-1]
        else:
            state = self.encoder(x)
        return self.head(torch.cat([state, calendar.flatten(1)], dim=1))


def forecast(panel, target, spec, origin, *, seed=SEED, epochs=EPOCHS,
             index='cap_seeded_22', delay=0):
    past = panel.iloc[:origin + 1]
    cutoff = origin - delay
    rent_growth = 100 * np.log(past[target].iloc[:cutoff + 1]).diff().dropna()
    hs = np.arange(1, HORIZONS + 1)
    anchor = float(past[target].iloc[cutoff])
    audit = {'n_training_origins': 0, 'min_mature_labels_per_head': 0,
             'last_training_target': None, 'latest_input_month': str(past.index[cutoff]),
             'parameters': 0}
    if spec.method in ('persistence', 'momentum3', 'ewma'):
        growth = (0 if spec.method == 'persistence' else
                  rent_growth.iloc[-3:].mean() if spec.method == 'momentum3' else
                  rent_growth.ewm(span=6, adjust=False).mean().iloc[-1])
        avg = np.repeat(growth, HORIZONS)
    else:
        seq, cal, y, mask, source = examples(panel, target, spec, origin, index=index, delay=delay)
        if len(source) < 2 or not mask.any():
            raise ValueError('Insufficient mature training origins')
        x = standardize(seq, mask)
        counts = mask.sum(axis=0)
        audit.update(n_training_origins=int(mask.any(axis=1).sum()),
                     min_mature_labels_per_head=int(counts.min()),
                     mature_labels_by_horizon=counts.tolist(),
                     last_training_target=str(panel.index[cutoff]))
        if spec.method.startswith('ridge'):
            flat = summary_inputs(x) if spec.method == 'ridge_summary' else x.reshape(len(x), -1)
            avg = np.zeros(HORIZONS)
            for h in range(HORIZONS):
                eligible = mask[:, h]
                if eligible.sum() < 3:
                    continue
                design = np.concatenate([flat, cal[:, h]], axis=1)
                # Refit scales separately for each horizon's mature examples.
                mu, sd = design[eligible].mean(0), design[eligible].std(0)
                sd = np.where(sd > 1e-6, sd, 1)
                design = (design - mu) / sd
                model = Ridge(alpha=10.).fit(design[eligible], y[eligible, h])
                avg[h] = model.predict(design[-1:])[0]
        else:
            torch.manual_seed(seed)
            model = SequenceNetwork(spec.method, spec.window, x.shape[-1])
            audit['parameters'] = sum(p.numel() for p in model.parameters())
            mu = (y * mask).sum(0) / np.maximum(counts, 1)
            sd = np.sqrt((((y - mu) ** 2) * mask).sum(0) / np.maximum(counts, 1))
            sd = np.maximum(sd, .25)
            tx, tc = torch.tensor(x, dtype=torch.float32), torch.tensor(cal, dtype=torch.float32)
            ty = torch.tensor((y - mu) / sd, dtype=torch.float32)
            tm = torch.tensor(mask, dtype=torch.float32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.05)
            denominator = tm.sum(0).clamp_min(1)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = (((model(tx, tc) - ty) ** 2 * tm).sum(0) / denominator).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                avg = model(tx[-1:], tc[-1:]).numpy()[0] * sd + mu
    cumulative = avg * (hs + delay)
    prediction = anchor * np.exp(cumulative / 100)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite forecast')
    return prediction, cumulative, audit


def evaluation_cases(index):
    cases = []
    for phase, start, end in [('development', '2023-07', '2024-05'),
                               ('outer', '2024-06', '2025-03')]:
        for month in pd.period_range(start, end, freq='M'):
            for h in (1, 3, 6):
                cases.append((phase, 'rolling', month - h, month, h))
    for origin in ('2023-01', '2023-04', '2023-07', '2024-05'):
        o = pd.Period(origin, freq='M')
        for h in range(1, HORIZONS + 1):
            cases.append(('outer' if origin == '2024-05' else 'development',
                          'fixed_origin', o, o + h, h))
    return pd.DataFrame(cases, columns=['phase', 'scenario', 'origin', 'month', 'horizon'])


def evaluate_spec(panel, target, spec, cases, **kwargs):
    rows = []
    for origin, local in cases.groupby('origin', sort=True):
        position = panel.index.get_loc(origin)
        predictions, growth, audit = forecast(panel, target, spec, position, **kwargs)
        for row in local.itertuples(index=False):
            actual = float(panel.loc[row.month, target])
            anchor = float(panel.loc[origin - kwargs.get('delay', 0), target])
            rows.append(dict(target=target, model=spec.name, method=spec.method, group=spec.group,
                             window=spec.window, phase=row.phase, scenario=row.scenario,
                             origin=str(origin), month=str(row.month), horizon=row.horizon,
                             actual=actual, predicted=float(predictions[row.horizon - 1]),
                             actual_growth=100 * np.log(actual / anchor),
                             predicted_growth=float(growth[row.horizon - 1]), **audit))
    return pd.DataFrame(rows)


def score(predictions):
    rows = []
    columns = ['target', 'phase', 'scenario', 'model', 'method', 'group', 'window', 'horizon']
    for keys, part in predictions.groupby(columns, sort=False):
        rows.append(dict(zip(columns, keys), cases=len(part),
                         mse=float(np.mean((part.predicted - part.actual) ** 2)),
                         rmse_sgd=float(np.sqrt(np.mean((part.predicted - part.actual) ** 2))),
                         growth_rmse=float(np.sqrt(np.mean((part.predicted_growth - part.actual_growth) ** 2)))))
    scores = pd.DataFrame(rows)
    base = scores[scores.method.eq('persistence')][['target', 'phase', 'scenario', 'horizon', 'mse']]
    scores = scores.merge(base.rename(columns={'mse': 'persistence_mse'}),
                          on=['target', 'phase', 'scenario', 'horizon'], validate='m:1')
    scores['normalized_mse'] = scores.mse / scores.persistence_mse
    return scores


def rankings(scores):
    return scores.groupby(['target', 'phase', 'scenario', 'model', 'method', 'group', 'window'],
                          as_index=False).agg(normalized_mse=('normalized_mse', 'mean'),
                                              mean_horizon_rmse_sgd=('rmse_sgd', 'mean'))


def select(rank):
    """All choices use the earlier phase, including forced family/group choices."""
    chosen = []
    for (target, scenario), part in rank[rank.phase.eq('development')].groupby(['target', 'scenario']):
        pools = {'overall': part, 'rent_only': part[part.group.eq('rent')],
                 'joint': part[part.group.eq('rent_stock_coe')],
                 'joint_lstm': part[part.group.eq('rent_stock_coe') & part.method.eq('lstm')]}
        pools.update({f'method:{m}': g for m, g in part.groupby('method')})
        pools.update({f'group:{g}': v for g, v in part.groupby('group')})
        # Same-processor input ablations have their window selected earlier too.
        pools.update({f'cell:{m}:{g}': v for (m, g), v in part.groupby(['method', 'group'])})
        for label, candidates in pools.items():
            best = candidates.sort_values(['normalized_mse', 'model']).iloc[0]
            chosen.append(dict(target=target, scenario=scenario, selection=label,
                               model=best.model, development_normalized_mse=float(best.normalized_mse)))
    return pd.DataFrame(chosen)


def block_interval(actual, control, prediction, *, weights=None, repeats=2000):
    """Input is one paired error summary per target month, retaining row weights."""
    rng = np.random.default_rng(42)
    n = len(actual)
    w = np.ones(n) if weights is None else np.asarray(weights)
    deltas = []
    for _ in range(repeats):
        starts = rng.integers(0, n - 3 + 1, size=int(np.ceil(n / 3)))
        p = np.concatenate([np.arange(s, s + 3) for s in starts])[:n]
        deltas.append(np.sqrt(np.average(prediction[p], weights=w[p])) -
                      np.sqrt(np.average(control[p], weights=w[p])))
    return np.quantile(deltas, [.025, .975]).tolist()


def selected_comparison(pred, chosen):
    rows = []
    for choice in chosen.itertuples(index=False):
        if choice.selection.startswith('cell:'):
            continue
        subset = pred[(pred.target == choice.target) & (pred.scenario == choice.scenario) & pred.phase.eq('outer')]
        candidate = subset[subset.model.eq(choice.model)].copy()
        control = subset[subset.method.eq('persistence')][['origin', 'month', 'horizon', 'predicted']]
        paired = candidate.merge(control, on=['origin', 'month', 'horizon'], suffixes=('', '_control'), validate='1:1')
        # Average within target month before bootstrap for overlapping horizons.
        paired['candidate_mse'] = (paired.predicted - paired.actual) ** 2
        paired['control_mse'] = (paired.predicted_control - paired.actual) ** 2
        monthly = paired.groupby('month')[['candidate_mse', 'control_mse']].mean()
        low, high = block_interval(monthly.index.to_numpy(),
                                   monthly.control_mse.to_numpy(), monthly.candidate_mse.to_numpy())
        c, p = np.sqrt(monthly.control_mse.mean()), np.sqrt(monthly.candidate_mse.mean())
        rows.append(dict(target=choice.target, scenario=choice.scenario, selection=choice.selection,
                         model=choice.model, development_normalized_mse=choice.development_normalized_mse,
                         outer_persistence_rmse_sgd=float(c), outer_rmse_sgd=float(p), delta_rmse_sgd=float(p-c),
                         delta_ci_low=low, delta_ci_high=high,
                         improved_months=int((monthly.candidate_mse < monthly.control_mse).sum())))
    return pd.DataFrame(rows)


def temporal_candidates(chosen, scenario, target='adjusted_mean_rent'):
    part = chosen[(chosen.target == target) & (chosen.scenario == scenario)]
    return {label: part[part.selection.eq(label)].model.iloc[0]
            for label in ('rent_only', 'joint', 'joint_lstm')}


def row_transfer(panel, predictions, chosen, out):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from analysis.eda.economic_signals import SETTINGS
    from analysis.experiments.auxiliary_ablation import feature_frame, model_inputs, BASE, BLOCK, AMENITIES
    frame = feature_frame()
    forecasts = predictions[predictions.target.eq('adjusted_mean_rent')].copy()
    forecasts = forecasts.drop_duplicates(['model', 'origin', 'month', 'horizon']).set_index(['model', 'origin', 'month'])
    parts = []
    for start, end in [('2023-07', '2024-01'), ('2024-01', '2024-06'), ('2024-06', '2025-04')]:
        train = frame[frame.RENT_APPROVAL_DATE < start]
        query = frame[(frame.RENT_APPROVAL_DATE >= start) & (frame.RENT_APPROVAL_DATE < end)]
        x, q, _ = model_inputs(train, query, BASE + BLOCK + AMENITIES)
        tree = HistGradientBoostingRegressor(**SETTINGS).fit(x, train.MONTHLY_RENT)
        base = tree.predict(q)
        original = pd.Period(start, 'M') - 1
        anchor = panel.loc[original, 'adjusted_mean_rent']
        for scenario in ('rolling', 'fixed_origin'):
            candidates = {'persistence': Spec('persistence').name, **temporal_candidates(chosen, scenario)}
            for label, model in candidates.items():
                for month, group in query.groupby('RENT_APPROVAL_DATE'):
                    m = pd.Period(month, 'M')
                    origin = m - 1 if scenario == 'rolling' else original
                    key = (model, str(origin), month)
                    # Fixed-origin earlier row fold Dec2023 needs forecasts outside
                    # the aggregate case list; computed from the same locked model.
                    if key in forecasts.index:
                        mean_prediction = float(forecasts.loc[key, 'predicted'])
                    else:
                        spec = next(s for s in specifications() if s.name == model)
                        mean_prediction = forecast(panel, 'adjusted_mean_rent', spec, panel.index.get_loc(origin))[0][m.ordinal-origin.ordinal-1]
                    correction = mean_prediction - anchor
                    positions = query.index.get_indexer(group.index)
                    residual = group.MONTHLY_RENT.to_numpy(float) - base[positions]
                    parts.append(dict(phase='outer' if start == OUTER_START else 'development',
                                      scenario=scenario, candidate=label, model=model, month=month,
                                      n=len(group), baseline_mse=float(np.mean(residual ** 2)),
                                      mean_residual=float(residual.mean()), correction=correction))
    monthly = pd.DataFrame(parts)
    scores = []
    for (scenario, label), part in monthly.groupby(['scenario', 'candidate']):
        dev = part[part.phase.eq('development')]
        denominator = np.sum(dev.n * dev.correction ** 2)
        gamma = float(np.clip(np.sum(dev.n * dev.correction * dev.mean_residual) / denominator, 0, 1)) if denominator else 0.
        monthly.loc[part.index, 'gamma'] = gamma
        for phase, sub in part.groupby('phase'):
            correction = gamma * sub.correction
            mse = sub.baseline_mse - 2 * correction * sub.mean_residual + correction ** 2
            c = float(np.sqrt(np.average(sub.baseline_mse, weights=sub.n)))
            p = float(np.sqrt(np.average(mse, weights=sub.n)))
            low, high = block_interval(sub.month.to_numpy(), sub.baseline_mse.to_numpy(), mse.to_numpy(), weights=sub.n.to_numpy())
            scores.append(dict(scenario=scenario, candidate=label, model=sub.model.iloc[0], phase=phase,
                               gamma=gamma, baseline_rmse_sgd=c, corrected_rmse_sgd=p,
                               delta_rmse_sgd=p-c, delta_ci_low=low, delta_ci_high=high,
                               improved_months=int((mse < sub.baseline_mse).sum()), rows=int(sub.n.sum())))
    monthly.to_csv(out / 'row_transfer_monthly.csv', index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(out / 'row_transfer_scores.csv', index=False)
    return scores


def table(frame):
    if frame.empty:
        return '(No eligible cases.)'
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.3f}'
        return str(value).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                       '| ' + ' | '.join('---' for _ in frame.columns) + ' |'] +
                      ['| ' + ' | '.join(cell(v) for v in row) + ' |'
                       for row in frame.itertuples(index=False, name=None)])


def write_report(out, metadata, pred, rank, chosen, comparisons, row_scores, sensitivity):
    headline = comparisons[comparisons.selection.isin(['overall', 'rent_only', 'joint', 'joint_lstm'])]
    columns = ['target', 'scenario', 'selection', 'model', 'development_normalized_mse', 'outer_rmse_sgd',
               'delta_rmse_sgd', 'delta_ci_low', 'delta_ci_high']
    lines = ['# Rental, stock-index and COE temporal processing ablation', '',
             'The earlier national-index study did compare Singapore mean-rent changes with overall stock returns: '
             'same-month correlation 0.005; prior-month correlation -0.253 (95% month-block interval -0.517 to 0.067). '
             'The worksite study instead concerns local relative rents. Neither earlier study used LSTM.', '',
             '## Earlier-selected models and later monthly-mean results', '',
             table(headline[columns]), '',
             'RMSE and deltas are SGD errors of the monthly arithmetic mean, not individual rental errors. '
             'Deltas compare persistence on identical cases. Rolling results pool 1/3/6-month horizons with equal '
             'target-month weight; fixed-origin results cover ten target months from May 2024 information. '
             'Development ranking gives each horizon equal weight after normalizing MSE to persistence. '
             'Every model/window is in all_scores.csv and rankings.csv; selection uses only development errors.', '',
             '## Which information and processing help?', '']
    cell = chosen[chosen.selection.str.startswith('cell:')].merge(
        rank[rank.phase.eq('outer')][['target', 'scenario', 'model', 'normalized_mse']],
        on=['target', 'scenario', 'model'], validate='m:1')
    cell.to_csv(out / 'input_processing_ablation.csv', index=False)
    for target in TARGETS:
        for scenario in ('rolling', 'fixed_origin'):
            part = cell[(cell.target == target) & (cell.scenario == scenario)]
            pivot = part.assign(method=part.selection.str.split(':').str[1],
                               group=part.selection.str.split(':').str[2]).pivot(index='method', columns='group', values='normalized_mse')
            lines += [f'### {target}, {scenario}', '', table(pivot.reset_index()), '']
    lines += ['Values above are later MSE / persistence MSE (below 1 improves persistence), '
              'after choosing each cell\'s window on earlier data. Compare both within a processor and against the '
              'rent-only control; a forced best joint model need not beat a simple rent baseline.', '',
              '## Transfer to individual-rental prediction', '']
    if row_scores is not None:
        lines += [table(row_scores[row_scores.phase.eq('outer')][['scenario', 'candidate', 'model', 'gamma',
                           'baseline_rmse_sgd', 'corrected_rmse_sgd', 'delta_rmse_sgd', 'delta_ci_low', 'delta_ci_high']]), '',
                  'The same 29-feature HistGradientBoosting baseline is frozen in each row fold. '
                  'An additive correction is gamma times the predicted adjusted national-index movement since '
                  'the fold cutoff. Gamma is constrained to [0,1] and fitted on July 2023–May 2024 row errors only. '
                  'Two earlier row folds precede the June 2024–March 2025 comparison. The input processor is selected '
                  'by earlier aggregate scores. This is a matched tree transfer, not a measured TabPFN improvement. '
                  'Monthly updates may use rents observed after the tree fit; fixed-origin paths cannot.', '']
    lines += ['## Locked-model sensitivity checks', '',
              table(sensitivity) if sensitivity is not None else '(Not run.)', '',
              'Seeds 43/44, replacing cap-seeded returns with equal-weight matched returns, and one extra month '
              'of input delay are checks of the earlier-selected joint LSTM; no new window is selected. '
              'The delay changes training and query information cutoffs and anchors to the last available rent. '
              'It approximates a full-month publication lag rather than a precise release calendar.', '',
              '## Design, availability and limits', '',
              '- 51 labeled months (January 2021–March 2025), from all 150,000 rental rows. Raw arithmetic means '
              'and a 2021-only fixed-property-model adjustment are separate responses. The reference model sees '
              'no later rents and is held fixed. Changing property premiums can remain.',
              '- Primary stock series is the existing historical-cap-seeded 22-name Singapore return proxy, '
              'not official STI. Matched stocks represent 81.56% of the source ETF portfolio. Later adjusted-price '
              'snapshots and fixed constituents can contain survivorship/revision effects.',
              '- COE A/C premium log levels and log changes; stocks use monthly log returns; rent uses log changes. '
              'Common known calendar features and training-origin start January 2022. Information sets are rent, '
              'rent+stock, rent+COE and all three. Windows 3/6/12 apply to lag regression, MLP and LSTM.',
              '- All 43 specifications run on the same evaluation cases per target/scenario. Training labels are '
              'masked until their endpoint is observed. Neural targets are mean monthly log growth for ten direct '
              'horizons; loss averages masked squared standardized errors equally across horizons. Outputs share '
              'an encoder, not independent new market observations. Ridge uses a separate mature-label head per '
              'horizon. No realized future exogenous path is supplied to a multi-step forecast.',
              '- Small MLP/LSTM: hidden size 8, 100 fixed epochs, AdamW .01, weight decay .05, gradient norm 1, '
              'seed 42, CPU deterministic execution. MLP uses the same ordered values and calendar inputs. '
              'This is a bounded architecture ablation, not exhaustive neural architecture search.',
              '- Development: rolling target months July 2023–May 2024; fixed ten-month paths from Jan/Apr/Jul '
              '2023. Later comparison: June 2024–March 2025. Earlier fixed paths overlap. This later period '
              'has already been examined elsewhere; it is not an untouched test. Few independent months and '
              'many candidates mean rankings can be unstable. No result establishes causality or global optimality.',
              '- Paired three-month moving-block intervals retain target-month dependence, and row intervals retain '
              'rental counts. They are conditional on fitted/selected models, omit fitting/selection uncertainty, '
              'and are not adjusted for model searches. Multi-horizon rolling errors also overlap beyond three months.',
              '- Test rents remain unavailable. A ten-month backtest cannot validate every forecast through July 2026. '
              'Existing submissions are unchanged. No neural or macro feature is promoted automatically.', '',
              '## Reproduce', '',
              '`.venv/Scripts/python.exe analysis/compare_temporal_sequence_ablation.py`', '',
              'Use `--resume` for hash-checked candidate caches, `--skip-row-transfer` for aggregate analysis only, '
              'or `--smoke` with a different output directory for a quick execution check. '
              'results.json contains versions, settings, hashes, elapsed times, source coverage and cutoff checks. '
              'forecast_predictions.csv records every origin, endpoint and latest permissible input/label date.', '',
              'References: [previous national-index study](../stock_index_regions/report.md), '
              '[PyTorch LSTM](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LSTM.html), '
              '[rolling-origin evaluation](https://otexts.com/fpp3/tscv.html).', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    # Standard static artifacts suitable for exporting into the project report.
    part = headline[(headline.target == 'adjusted_mean_rent') & headline.scenario.eq('fixed_origin')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    axes[0].barh(part.selection, part.outer_rmse_sgd, color='#227D91')
    axes[0].set(xlabel='Monthly mean RMSE (SGD)', title='Fixed-origin, adjusted mean rent')
    actual = panel_actual = pred[(pred.target == 'adjusted_mean_rent') & pred.scenario.eq('fixed_origin') & pred.phase.eq('outer')]
    actual = actual.drop_duplicates('month').sort_values('month')
    axes[1].plot(pd.to_datetime(actual.month), actual.actual, color='black', label='Observed')
    for choice in chosen[(chosen.target == 'adjusted_mean_rent') & chosen.scenario.eq('fixed_origin') &
                         chosen.selection.isin(['rent_only', 'joint_lstm'])].itertuples():
        local = panel_actual[panel_actual.model.eq(choice.model)].sort_values('month')
        axes[1].plot(pd.to_datetime(local.month), local.predicted, label=choice.selection)
    axes[1].set(ylabel='Adjusted monthly mean (SGD)', title='All forecasts use May 2024 information')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out / '01_fixed_origin_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-row-transfer', action='store_true')
    args = parser.parse_args()
    if args.smoke and args.output == OUT:
        args.output = OUT / 'smoke'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    sources = [DATA / 'train.csv', AUX / 'sg-stock-prices.csv', AUX / 'sg-coe-prices.csv',
               AUX / 'sg-hdb-block.csv', AUX / 'sg-mrt-stations.csv', AUX / 'sg-shopping-malls.csv',
               AUX / 'sg-schools.csv', ROOT / 'analysis/configs/sti_etf_weights_2020_12.csv',
               ROOT / 'analysis/configs/stock_index_methodology.json']
    code = [Path(__file__), ROOT / 'analysis/experiments/stock_index_regions.py',
            ROOT / 'analysis/experiments/auxiliary_ablation.py', ROOT / 'analysis/data.py']
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources + code}
    settings = dict(hashes=hashes, epochs=EPOCHS, seed=SEED, smoke=args.smoke,
                    specs=[asdict(s) for s in specifications()], targets=TARGETS,
                    development_start='2023-07', outer_start=OUTER_START)
    settings_path = out / 'settings.json'
    if args.resume and settings_path.exists() and json.loads(settings_path.read_text()) != json.loads(json.dumps(settings)):
        raise ValueError('Resume settings/source/code hashes differ; use a new output directory')
    settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        panel, diagnostics = build_panel(pd.read_csv(DATA / 'train.csv'), pd.read_csv(AUX / 'sg-stock-prices.csv'),
                                        pd.read_csv(AUX / 'sg-coe-prices.csv'),
                                        pd.read_csv(ROOT / 'analysis/configs/sti_etf_weights_2020_12.csv'),
                                        pd.read_csv(AUX / 'sg-hdb-block.csv'))
        panel.to_csv(out / 'monthly_input_panel.csv', index_label='month')
        cases = evaluation_cases(panel.index)
        specs = specifications()
        if args.smoke:
            specs = [s for s in specs if s.method in ('persistence', 'ridge_summary') or
                     (s.method in ('lstm', 'mlp') and s.group == 'rent_stock_coe' and s.window == 3)]
        all_predictions, timings = [], []
        for target in TARGETS:
            for i, spec in enumerate(specs):
                cache = out / 'candidate_cache' / f'{target}__{spec.name}.csv'
                tick = time.perf_counter()
                if args.resume and cache.exists():
                    result = pd.read_csv(cache)
                else:
                    result = evaluate_spec(panel, target, spec, cases, epochs=3 if args.smoke else EPOCHS)
                    cache.parent.mkdir(exist_ok=True)
                    result.to_csv(cache, index=False)
                all_predictions.append(result)
                timings.append(dict(target=target, model=spec.name, seconds=time.perf_counter()-tick))
                print(f'{target} {i+1}/{len(specs)} {spec.name} {timings[-1]["seconds"]:.1f}s', flush=True)
        pred = pd.concat(all_predictions, ignore_index=True)
        pred.to_csv(out / 'forecast_predictions.csv', index=False)
        scores = score(pred)
        rank = rankings(scores)
        chosen = select(rank)
        comparisons = selected_comparison(pred, chosen)
        scores.to_csv(out / 'all_scores.csv', index=False)
        rank.to_csv(out / 'rankings.csv', index=False)
        chosen.to_csv(out / 'earlier_selections.csv', index=False)
        comparisons.to_csv(out / 'selected_comparisons.csv', index=False)
        sensitivity_rows = []
        for choice in chosen[chosen.selection.eq('joint_lstm')].itertuples(index=False):
            spec = next(s for s in specs if s.name == choice.model)
            for label, options in [('seed43', dict(seed=43)), ('seed44', dict(seed=44)),
                                   ('equal_weight_index', dict(index='equal_weight_matched')),
                                   ('extra_month_delay', dict(delay=1))]:
                local = cases[(cases.phase == 'outer') & cases.scenario.eq(choice.scenario)]
                result = evaluate_spec(panel, choice.target, spec, local, epochs=3 if args.smoke else EPOCHS, **options)
                result.to_csv(out / f'sensitivity_{choice.target}_{choice.scenario}_{label}.csv', index=False)
                error = (result.predicted - result.actual) ** 2
                sensitivity_rows.append(dict(target=choice.target, scenario=choice.scenario, check=label,
                                             model=choice.model, outer_rmse_sgd=float(np.sqrt(error.mean()))))
        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(out / 'sensitivity_scores.csv', index=False)
        row_scores = None
        if not args.skip_row_transfer and not args.smoke:
            print('Individual-rental transfer: three matched tree fits', flush=True)
            row_scores = row_transfer(panel, pred, chosen, out)
    if any(sha256(ROOT / k) != value for k, value in hashes.items()):
        raise AssertionError('An input/code file changed during the experiment')
    fitted = pred.last_training_target.dropna()
    if (fitted > pred.loc[fitted.index, 'origin']).any() or (pred.latest_input_month > pred.origin).any():
        raise AssertionError('Temporal cutoff violation')
    metadata = dict(settings=settings, versions={p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'torch')},
                    monthly_observations=len(panel), rental_rows=int(panel.n.sum()), index_diagnostics=diagnostics,
                    source_files_unchanged=True, training_targets_mature=True, test_labels_used=False,
                    elapsed_seconds=time.perf_counter()-started)
    (out / 'results.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
    write_report(out, metadata, pred, rank, chosen, comparisons, row_scores, sensitivity)
    print('COMPLETE', out, f'{metadata["elapsed_seconds"]:.1f}s', flush=True)
