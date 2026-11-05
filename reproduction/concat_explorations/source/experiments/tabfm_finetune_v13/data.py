"""Causal, immutable episode data from the existing orig/A frozen banks."""
from dataclasses import dataclass, fields
from pathlib import Path
import json
import numpy as np
from experiments.tabfm_v10.benchmark import load_frozen, sha256
from experiments.macro_encoders_v3 import contract, data as macro_data, scaling

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / 'reports/tabfm_v10/runs/v10-screen-orig-s42/selection.npz'

@dataclass(frozen=True)
class Population:
    numeric: np.ndarray
    categorical: np.ndarray
    target: np.ndarray
    anchor_ksgd: np.ndarray
    actual_sgd: np.ndarray
    row_id: np.ndarray
    origin: np.ndarray
    horizon: np.ndarray

    def __post_init__(self):
        for f in fields(self):
            a = np.array(getattr(self, f.name), copy=True)
            a.setflags(write=False)
            object.__setattr__(self, f.name, a)
        n = len(self.row_id)
        if any(len(getattr(self, f.name)) != n for f in fields(self)):
            raise ValueError('row count mismatch')
        if len(np.unique(self.row_id)) != n:
            raise ValueError('duplicate row IDs')
        if not np.allclose((self.target + self.anchor_ksgd) * 1000, self.actual_sgd, atol=.002, rtol=0):
            raise ValueError('residual/anchor unit mismatch')

    @property
    def target_month(self):
        return self.origin + self.horizon


def eligible_context_indices(fit, query_origin, query_row_id):
    return np.flatnonzero((fit.target_month <= query_origin) & (fit.row_id != query_row_id))


def episode(fit, query, query_indices, context_indices):
    qi = np.asarray(query_indices, dtype=int)
    ci = np.asarray(context_indices, dtype=int)
    if not len(qi) or not len(ci) or len(np.unique(ci)) != len(ci):
        raise ValueError('nonempty query and unique context required')
    if np.any(fit.target_month[ci, None] > query.origin[qi][None, :]) or np.intersect1d(fit.row_id[ci], query.row_id[qi]).size:
        raise ValueError('context violates origin or self-ID rule')
    if fit.numeric.shape[1] != query.numeric.shape[1] or fit.categorical.shape[1] != query.categorical.shape[1]:
        raise ValueError('feature schema mismatch')
    mu = float(np.mean(fit.target[ci], dtype=np.float64))
    sd = float(np.std(fit.target[ci], dtype=np.float64))
    if sd < 1e-8: sd = 1.0
    x = np.concatenate((np.concatenate((fit.numeric[ci], query.numeric[qi])),
                        np.concatenate((fit.categorical[ci], query.categorical[qi]))), axis=1).astype(np.float32)
    y = np.concatenate(((fit.target[ci] - mu) / sd, np.zeros(len(qi)))).astype(np.float32)
    return dict(x=x, y=y, train_size=len(ci), categorical_mask=np.array([False]*fit.numeric.shape[1]+[True]*fit.categorical.shape[1]),
                target_mean=mu, target_std=sd, query_targets=query.target[qi].copy(),
                anchor_ksgd=query.anchor_ksgd[qi].copy(), query_row_id=query.row_id[qi].copy(), context_row_id=fit.row_id[ci].copy())


def fit_scalar_state(values, masks, fit_windows, names):
    idx = np.unique(np.asarray(fit_windows, dtype=int))
    return scaling.fit_scalars(np.asarray(values)[idx], np.asarray(masks)[idx], names, 'macro_scalar',
        split_id='orig', stage='A', policy='origin_asof', fitted_on='unique windows referenced by fit rows')


def load_banks(expanded=False, stock=False):
    if stock and not expanded: raise ValueError('stock bank requires expanded=True')
    fit, hold, nums, cats = load_frozen()
    metadata = {'numeric_names': list(nums), 'categorical_names': list(cats), 'expanded': expanded}
    if expanded:
        manifest = contract.load_manifest()
        if manifest['policy'] != 'origin_asof': raise ValueError('origin_asof macro contract required')
        w = macro_data.load_windows(manifest, 'orig', 'A')
        rows = macro_data.load_rows(manifest, 'orig', 'A')
        vals = np.concatenate((w.dbs_scalars, w.coe_scalars.reshape(w.n_windows, -1)), axis=1)
        masks = np.concatenate((w.dbs_scalar_mask, w.coe_scalar_mask.reshape(w.n_windows, -1)), axis=1).astype(bool)
        names = list(contract.DBS_SCALARS) + [f'{cat}_{name}' for cat in contract.COE_CATEGORIES for name in contract.COE_SCALARS]
        state = fit_scalar_state(vals, masks, rows.row_window_index[rows.population == 'fit'], names)
        features = np.concatenate((state.transform(vals, masks), masks.astype(np.float32)), axis=1)
        for population, popname in ((fit, 'fit'), (hold, 'holdout')):
            take = np.flatnonzero(rows.population == popname)
            lookup = {int(rows.row_id[i]): i for i in take}
            if len(lookup) != len(take): raise ValueError('duplicate macro row IDs')
            indices = np.array([lookup[int(i)] for i in population['row_id']])
            win = rows.row_window_index[indices]
            if np.any(rows.macro_cutoff_month[indices] != w.window_cutoff_month[win]) or np.any(rows.macro_cutoff_month[indices] != population['origin']): raise ValueError('macro cutoff is not query origin')
            population['numeric'] = np.concatenate((population['numeric'], features[win]), axis=1)
        metadata.update(numeric_names=list(nums)+names+[n+'_observed' for n in names], macro_scaler=state.to_dict(), macro_contract=manifest['contract_id'], macro_policy=manifest['policy'], macro_artifacts=contract.stage_entry(manifest, 'orig', 'A')['artifacts'])
        for artifact in metadata['macro_artifacts'].values():
            if sha256(contract.resolve(artifact['path'])) != artifact['sha256']: raise ValueError('macro artifact hash mismatch')
    if stock:
        import pandas as pd
        origins = np.unique(np.concatenate((fit['origin'], hold['origin'])))
        values, observed, audit = stock_sector_windows(pd.read_csv(STOCK_CSV), origins)
        fit_windows = np.flatnonzero(np.isin(origins, np.unique(fit['origin'])))
        state = fit_scalar_state(values, observed, fit_windows, STOCK_NAMES)
        transformed = np.concatenate((state.transform(values, observed), observed.astype(np.float32)), axis=1)
        for population in (fit, hold):
            idx = np.searchsorted(origins, population['origin'])
            population['numeric'] = np.concatenate((population['numeric'], transformed[idx]), axis=1)
        metadata['numeric_names'] += STOCK_NAMES + [n+'_observed' for n in STOCK_NAMES]
        metadata['stock'] = {'universe': STOCK_UNIVERSE, 'input': str(STOCK_CSV.relative_to(ROOT)), 'input_sha256': sha256(STOCK_CSV), 'fit_unique_origins': origins[fit_windows].tolist(), 'origins': origins.tolist(), 'feature_observed_windows': dict(zip(STOCK_NAMES, observed.sum(axis=0).astype(int).tolist())), 'scaler':state.to_dict(), 'symbol_audit':audit, 'policy':'month-end origin-as-of; fixed 9-symbol local universe; no filling; sector summaries are proxies', 'note':'C38U.SI is CICT; CSV retains an older company name'}
    metadata['input_hashes'] = {str(p.relative_to(ROOT)): sha256(p) for p in [ROOT/'reports/diagnostics_phase2_c/frozen/afit.npz', ROOT/'reports/diagnostics_phase2_c/frozen/ahold.npz', ROOT/'reports/diagnostics_phase2_c/frozen/afit_columns.json', SELECTION]}
    with np.load(SELECTION) as a:
        metadata['fixed_validation_row_id'] = a['validation_row_id'].tolist()
        metadata['fixed_context_row_id'] = a['context_row_id'].tolist()
    return Population(**fit), Population(**hold), metadata

# Fixed universe, chosen before sector evaluation; C38U.SI is CICT (older CSV name differs).
STOCK_UNIVERSE = {'banks': ('D05.SI', 'O39.SI', 'U11.SI'),
                  'property': ('C38U.SI', 'A17U.SI', 'M44U.SI', 'ME8U.SI', 'C09.SI', 'U14.SI')}
STOCK_MEASURES = ('mean_return_21', 'mean_return_63', 'mean_return_126',
                  'mean_volatility_63', 'mean_volume_ratio_21_63', 'available_symbol_fraction',
                  'return_63_cross_section_dispersion', 'mean_observation_coverage_63')
STOCK_NAMES = [f'stock_{sector}_{name}' for sector in STOCK_UNIVERSE for name in STOCK_MEASURES]
STOCK_CSV = ROOT / 'data/auxiliary/sg-stock-prices.csv'


def stock_sector_windows(frame, origins):
    """Build sector summaries over the common observed trading-date grid, with no filling.

    Returns use endpoint adjusted closes and require a valid price on the latest
    session; volatility requires at least 42 of the latest 63 adjacent daily returns.
    Volume ratios require all 63 nonnegative volumes and a positive denominator.
    Coverage counts positive adjusted closes among the latest 63 grid sessions,
    divided by 63 (including before-file history as unavailable).
    """
    import pandas as pd
    df = frame.loc[frame.SYMBOL.isin(sum(STOCK_UNIVERSE.values(), ())),
                   ['SYMBOL', 'DATE', 'ADJUSTED_CLOSE', 'VOLUME']].copy()
    df['DATE'] = pd.to_datetime(df.DATE, errors='raise')
    if df.duplicated(['SYMBOL', 'DATE']).any(): raise ValueError('duplicate symbol/date')
    for col in ('ADJUSTED_CLOSE', 'VOLUME'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[~np.isfinite(df.ADJUSTED_CLOSE) | (df.ADJUSTED_CLOSE <= 0), 'ADJUSTED_CLOSE'] = np.nan
    df.loc[~np.isfinite(df.VOLUME) | (df.VOLUME < 0), 'VOLUME'] = np.nan
    values = np.zeros((len(origins), len(STOCK_NAMES)), np.float32)
    masks = np.zeros_like(values, dtype=bool)
    audit = {}
    for symbol in sum(STOCK_UNIVERSE.values(), ()):
        s = df[df.SYMBOL == symbol]
        audit[symbol] = {'rows':len(s), 'first_date':str(s.DATE.min().date()) if len(s) else None,
                         'last_date':str(s.DATE.max().date()) if len(s) else None,
                         'invalid_price_rows':int(s.ADJUSTED_CLOSE.isna().sum()),
                         'invalid_volume_rows':int(s.VOLUME.isna().sum()), 'origin_windows': {}}
    for wi, origin in enumerate(origins):
        cutoff = pd.Period('2021-01', freq='M') + int(origin)
        visible = df[df.DATE < cutoff.end_time + pd.Timedelta(nanoseconds=1)]
        dates = pd.DatetimeIndex(sorted(visible.DATE.unique()))
        if not len(dates): continue
        for si, symbols in enumerate(STOCK_UNIVERSE.values()):
            samples = [[] for _ in range(5)]
            ret63 = []
            avail = 0
            coverages = []
            for symbol in symbols:
                s = visible[visible.SYMBOL == symbol].set_index('DATE').reindex(dates)
                prices = s.ADJUSTED_CLOSE.to_numpy(float)
                volumes = s.VOLUME.to_numpy(float)
                valid = np.isfinite(prices)
                avail += int(valid[-1])
                audit[symbol]['origin_windows'][str(int(origin))] = {'latest_session': str(dates[-1].date()), 'latest_price_observed': bool(valid[-1]), 'price_observations_last_63': int(valid[-63:].sum()), 'volume_observations_last_63': int(np.isfinite(volumes[-63:]).sum())}
                coverages.append(float(valid[-63:].sum()) / 63.)
                for j, lag in enumerate((21, 63, 126)):
                    if len(prices) > lag and valid[-1] and valid[-lag-1]:
                        ret = prices[-1] / prices[-lag-1] - 1.
                        samples[j].append(ret)
                        if lag == 63: ret63.append(ret)
                if len(prices) >= 64 and valid[-1]:
                    last = prices[-64:]
                    daily = np.log(last[1:] / last[:-1])
                    daily = daily[np.isfinite(daily)]
                    if len(daily) >= 42: samples[3].append(float(np.std(daily)))
                if len(volumes) >= 63 and valid[-1] and np.isfinite(volumes[-63:]).all():
                    denom = np.mean(volumes[-63:])
                    if denom > 0: samples[4].append(float(np.mean(volumes[-21:]) / denom))
            offset = si * len(STOCK_MEASURES)
            for j, sample in enumerate(samples):
                if sample:
                    values[wi, offset+j] = np.mean(sample); masks[wi, offset+j] = True
            values[wi, offset+5] = avail / len(symbols); masks[wi, offset+5] = True
            if len(ret63) >= 2:
                values[wi, offset+6] = np.std(ret63); masks[wi, offset+6] = True
            values[wi, offset+7] = np.mean(coverages); masks[wi, offset+7] = True
    return values, masks, audit
