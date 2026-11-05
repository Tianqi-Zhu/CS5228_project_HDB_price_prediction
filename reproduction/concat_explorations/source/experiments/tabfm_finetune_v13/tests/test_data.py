import numpy as np
import pytest
from experiments.tabfm_finetune_v13.data import Population, episode, eligible_context_indices, fit_scalar_state


def pop(target=(1., 2., 3.)):
    n = len(target)
    return Population(numeric=np.arange(n * 2, dtype=np.float32).reshape(n, 2), categorical=np.ones((n, 1), dtype=np.int32), target=np.array(target), anchor_ksgd=np.ones(n), actual_sgd=(np.array(target)+1)*1000, row_id=np.arange(n), origin=np.array([0, 1, 2])[:n], horizon=np.ones(n, dtype=int))


def test_context_causal_and_episode_masks_query():
    p = pop()
    assert eligible_context_indices(p, 0, 0).size == 0
    assert eligible_context_indices(p, 2, 1).tolist() == [0]
    e = episode(p, p, [2], [0, 1])
    assert e['y'].tolist() == [-1., 1., 0.]
    assert e['query_targets'].tolist() == [3.]
    assert e['categorical_mask'].tolist() == [False, False, True]
    assert (e['query_targets'] + e['anchor_ksgd']).tolist() == [4.]
    with pytest.raises(ValueError):
        episode(p, p, [1], [1])


def test_future_and_query_label_perturbation_does_not_change_model_inputs():
    a = pop()
    b = pop((1., 2., 999.))
    ea, eb = episode(a, a, [2], [0, 1]), episode(b, b, [2], [0, 1])
    for key in ('x', 'y', 'categorical_mask'):
        np.testing.assert_array_equal(ea[key], eb[key])
    assert ea['target_mean'] == eb['target_mean'] and ea['target_std'] == eb['target_std']


def test_scalar_scaler_fit_only():
    values = np.array([[1., 10.], [3., 20.], [999., -999.]])
    masks = np.ones_like(values, dtype=bool)
    a = fit_scalar_state(values, masks, [0, 1], ['a', 'b'])
    values[2] = [1e9, 1e9]
    b = fit_scalar_state(values, masks, [0, 1], ['a', 'b'])
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.scale, b.scale)


def test_real_banks_and_fixed_monitor():
    from experiments.tabfm_finetune_v13.data import load_banks
    fit, hold, meta = load_banks(True)
    assert fit.numeric.shape == (52310, 152)
    assert hold.numeric.shape == (35882, 152)
    assert len(meta['fixed_validation_row_id']) == 512
    assert set(meta['fixed_validation_row_id']) <= set(hold.row_id)
    assert set(meta['fixed_context_row_id']) <= set(fit.row_id)
    assert (fit.target_month <= hold.origin.min()).all()
    assert int((fit.origin >= fit.target_month.min()).sum()) == 49146
    assert not fit.numeric.flags.writeable


def test_stock_causality_missingness_and_short_history():
    import pandas as pd
    from experiments.tabfm_finetune_v13.data import stock_sector_windows, STOCK_UNIVERSE
    dates = pd.bdate_range('2021-01-01', periods=150)
    rows = [(s, d, 100.+i, 1000.+i) for group in STOCK_UNIVERSE.values() for s in group for i,d in enumerate(dates)]
    frame = pd.DataFrame(rows, columns=['SYMBOL','DATE','ADJUSTED_CLOSE','VOLUME'])
    v, mask, _ = stock_sector_windows(frame, [0, 5])
    # January has too few trading sessions for 21/63/126-day return.
    assert not mask[0, :3].any()
    assert mask[1, :].all()
    altered = frame.copy()
    altered.loc[altered.DATE > '2021-06-30', 'ADJUSTED_CLOSE'] *= 1000
    v2, m2, _ = stock_sector_windows(altered, [0, 5])
    np.testing.assert_array_equal(v, v2)
    np.testing.assert_array_equal(mask, m2)
    # A missing latest price is not backfilled; other bank members remain available.
    damaged = frame.copy()
    damaged.loc[(damaged.SYMBOL == 'D05.SI') & (damaged.DATE == '2021-06-30'), 'ADJUSTED_CLOSE'] = np.nan
    vd, md, _ = stock_sector_windows(damaged, [5])
    assert vd[0, 5] == pytest.approx(2/3)
    assert md[0, 5]
    with pytest.raises(ValueError, match='duplicate'):
        stock_sector_windows(pd.concat([frame, frame.iloc[:1]]), [5])


def test_stock_bank_real_origins_and_fit_only_scaler():
    from experiments.tabfm_finetune_v13.data import load_banks, STOCK_NAMES
    fit, hold, meta = load_banks(expanded=True, stock=True)
    assert fit.numeric.shape == (52310, 184)
    assert hold.numeric.shape == (35882, 184)
    assert len(STOCK_NAMES) == 16
    assert meta['stock']['fit_unique_origins'] == list(range(26))
    assert meta['stock']['origins'] == list(range(27))
    assert meta['stock']['scaler']['n_windows_used'] == 26
    assert np.isfinite(fit.numeric).all()
    for origin in np.unique(fit.origin):
        idx = np.flatnonzero(fit.origin == origin)
        np.testing.assert_array_equal(fit.numeric[idx[0], -32:], fit.numeric[idx[-1], -32:])
