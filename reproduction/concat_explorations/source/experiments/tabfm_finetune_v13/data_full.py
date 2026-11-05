"""Raw 150k chronology; cold-start rows explicitly retained and zero-context capable."""
from dataclasses import replace
import numpy as np
import pandas as pd
from experiments.fusion_contract_v2 import contract as c2, export, panel
from experiments.fusion_contract_v3 import contract as c3, dbs, coe
from experiments.neural_fusion_v2.contract import StageArrays
from experiments.neural_fusion_v2.preprocessing import ScalerState
from experiments.tabfm_finetune_v13.data import (Population, eligible_context_indices, fit_scalar_state,
    stock_sector_windows, STOCK_NAMES, STOCK_CSV, sha256)


def make_samples(months, phase):
    if phase not in ('selection', 'final_refit'): raise ValueError('unknown phase')
    months = np.asarray(months, dtype=int)
    rng = np.random.default_rng(c2.SAMPLE_SEED)
    horizon = rng.integers(1, c2.MAX_HORIZON+1, size=len(months))
    invalid = (months > 0) & (horizon > months)
    repair = np.random.default_rng(c2.SAMPLE_SEED + 1)
    horizon[invalid] = np.array([repair.integers(1, min(c2.MAX_HORIZON, t)+1) for t in months[invalid]])
    horizon[months == 0] = 1
    origin = months - horizon
    population = np.full(len(months), 'fit', dtype=object)
    if phase == 'selection':
        val = months > 38
        population[val] = 'validation'; origin[val] = 38; horizon[val] = months[val] - 38
    return pd.DataFrame({'SOURCE_ROW_ID': np.arange(len(months)), 'population':population,
                         'target_month_index':months, 'origin_month_index':origin, 'horizon_months':horizon})


def episode(fit, query, query_indices, context_indices):
    qi, ci = np.asarray(query_indices, dtype=int), np.asarray(context_indices, dtype=int)
    if not len(qi) or len(np.unique(ci)) != len(ci): raise ValueError('nonempty query and unique context required')
    if np.any(fit.target_month[ci,None] > query.origin[qi][None,:]) or np.intersect1d(fit.row_id[ci], query.row_id[qi]).size:
        raise ValueError('context violates origin or self-ID rule')
    if fit.numeric.shape[1] != query.numeric.shape[1] or fit.categorical.shape[1] != query.categorical.shape[1]: raise ValueError('feature schema mismatch')
    mu = float(np.mean(fit.target[ci], dtype=np.float64)) if len(ci) else 0.
    sd = float(np.std(fit.target[ci], dtype=np.float64)) if len(ci) else 1.
    if sd < 1e-8: sd = 1.
    x = np.concatenate((np.concatenate((fit.numeric[ci], query.numeric[qi])), np.concatenate((fit.categorical[ci], query.categorical[qi]))), axis=1).astype(np.float32)
    y = np.concatenate(((fit.target[ci]-mu)/sd, np.zeros(len(qi)))).astype(np.float32)
    return dict(x=x,y=y,train_size=len(ci),categorical_mask=np.array([False]*fit.numeric.shape[1]+[True]*fit.categorical.shape[1]), target_mean=mu,target_std=sd,query_targets=query.target[qi].copy(), anchor_ksgd=query.anchor_ksgd[qi].copy(),query_row_id=query.row_id[qi].copy(),context_row_id=fit.row_id[ci].copy())


def _slice(arrays, idx):
    return replace(arrays, **{key:getattr(arrays,key)[idx] for key in ('static','context','history','sequence','sequence_mask','target_sgd','row_id','horizon','origin')})


def prepare_phase(phase, feature_bank='base', *, source_frame=None):
    if feature_bank not in ('base','macro','stock'): raise ValueError('unknown feature bank')
    frame = export.load_source() if source_frame is None else source_frame.copy()
    table = make_samples(frame.MONTH_INDEX.to_numpy(), phase)
    table['actual_rent_sgd'] = frame.MONTHLY_RENT.to_numpy(dtype=float)
    cutoff = 38 if phase == 'selection' else 50
    p = panel.Panel.build(frame, cutoff)
    built = export.build_features(frame, table, p)
    ar = built.arrays; order = built.feature_order
    # Missing cold-start anchor is represented as 0 only in the anchor itself;
    # anchor_missing and anchor_level retain their actual values.
    anchor_idx = order['context'].index('anchor_rent_sgd')
    ar['context'][:,anchor_idx] = np.nan_to_num(ar['context'][:,anchor_idx], nan=0.)
    arrays = StageArrays(split='raw',stage=phase, **ar, static_order=order['static'], context_order=order['context'], history_order=order['history'], sequence_order=order['sequence'])
    fit_idx = np.flatnonzero(table.population.to_numpy() == 'fit')
    val_idx = np.flatnonzero(table.population.to_numpy() == 'validation')
    scaler = ScalerState.fit(_slice(arrays, fit_idx), population=f'raw/{phase}/fit')
    transformed = scaler.transform(arrays)
    numeric = np.concatenate([transformed['static'], transformed['history'], transformed['context']], axis=1)
    cats = transformed['categorical']
    names = list(scaler.static_columns)+list(scaler.history_columns)+list(scaler.context_columns)
    state = {'scaler':scaler.to_dict(), 'encoding':built.encoding}
    metadata = {'phase':phase, 'feature_bank':feature_bank, 'fit_cutoff':cutoff, 'numeric_names':names, 'categorical_names':arrays.categorical_columns}
    if feature_bank in ('macro','stock'):
        origins = np.unique(arrays.origin)
        daily = dbs.build_dbs_frame(dbs.parse_stock_frame())
        monthly = coe.build_coe_monthly(coe.parse_coe_frame())
        val, mask = [], []
        for o in origins:
            dv,dm = dbs.scalar_arrays(daily, dbs.asof_position(daily,int(o)), int(o))
            cv,cm = coe.scalar_arrays(monthly,int(o))
            val.append(np.concatenate((dv,cv.ravel()))); mask.append(np.concatenate((dm,cm.ravel())))
        val, mask = np.asarray(val), np.asarray(mask,dtype=bool)
        macro_names = list(c3.DBS_SCALARS)+[f'{cat}_{name}' for cat in c3.COE_CATEGORIES for name in c3.COE_SCALARS]
        fwin = np.flatnonzero(np.isin(origins, np.unique(arrays.origin[fit_idx])))
        ms = fit_scalar_state(val,mask,fwin,macro_names)
        extras = np.concatenate((ms.transform(val,mask),mask.astype(np.float32)),axis=1)
        numeric = np.concatenate((numeric,extras[np.searchsorted(origins,arrays.origin)]),axis=1)
        metadata['numeric_names'] += macro_names + [n+'_observed' for n in macro_names]
        state['macro_scaler'] = ms.to_dict()
        state['macro_fit_origins'] = origins[fwin].tolist()
        if feature_bank == 'stock':
            sv,sm,audit = stock_sector_windows(pd.read_csv(STOCK_CSV),origins)
            ss = fit_scalar_state(sv,sm,fwin,STOCK_NAMES)
            extras = np.concatenate((ss.transform(sv,sm),sm.astype(np.float32)),axis=1)
            numeric = np.concatenate((numeric,extras[np.searchsorted(origins,arrays.origin)]),axis=1)
            metadata['numeric_names'] += STOCK_NAMES + [n+'_observed' for n in STOCK_NAMES]
            state.update(stock_scaler=ss.to_dict(),stock_audit=audit)
    actual = arrays.target_sgd.astype(float)
    anchor = transformed['anchor_ksgd'].astype(float)
    def population(idx):
        return Population(numeric=numeric[idx],categorical=cats[idx],target=actual[idx]/1000-anchor[idx],anchor_ksgd=anchor[idx],actual_sgd=actual[idx],row_id=arrays.row_id[idx],origin=arrays.origin[idx],horizon=arrays.horizon[idx])
    metadata.update(n_fit=len(fit_idx),n_validation=len(val_idx),cold_start=int((arrays.origin[fit_idx]<0).sum()), source_sha256=sha256(c2.SOURCE), stock_sha256=sha256(c3.STOCK_SOURCE), coe_sha256=sha256(c3.COE_SOURCE))
    return population(fit_idx), population(val_idx) if len(val_idx) else None, metadata, state
