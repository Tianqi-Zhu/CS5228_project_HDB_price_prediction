import numpy as np
from experiments.tabfm_finetune_v13.data_full import make_samples, episode
from experiments.tabfm_finetune_v13.data import Population


def test_sample_horizons_preserved_and_cold_start():
    months = np.repeat(np.arange(20), 4)
    t = make_samples(months, 'final_refit')
    original = np.random.default_rng(20261024).integers(1, 17, size=len(months))
    valid = months >= original
    np.testing.assert_array_equal(t.horizon_months.to_numpy()[valid], original[valid])
    assert (t.origin_month_index[months == 0] == -1).all()
    assert (t.horizon_months[months == 0] == 1).all()
    assert (t.origin_month_index[months > 0] >= 0).all()


def test_empty_context_episode_masks_query():
    p = Population(numeric=np.ones((1,2)), categorical=np.ones((1,1)), target=np.array([2.]), anchor_ksgd=np.array([0.]), actual_sgd=np.array([2000.]), row_id=np.array([1]), origin=np.array([-1]), horizon=np.array([1]))
    e = episode(p,p,[0],[])
    assert e['train_size'] == 0 and e['y'].tolist() == [0.]
    assert e['target_mean'] == 0 and e['target_std'] == 1


def test_real_selection_and_final_and_validation_label_invariance():
    from experiments.fusion_contract_v2.export import load_source
    from experiments.tabfm_finetune_v13.data_full import prepare_phase, eligible_context_indices
    frame = load_source()
    a, b, meta, state = prepare_phase('selection', 'base', source_frame=frame)
    assert (len(a.row_id), len(b.row_id)) == (114842, 35158)
    assert len(np.union1d(a.row_id, b.row_id)) == 150000
    assert meta['cold_start'] == 3129
    first = np.flatnonzero(a.origin == -1)
    assert len(first) == 3129 and (a.anchor_ksgd[first] == 0).all()
    assert not len(eligible_context_indices(a, -1, a.row_id[first[0]]))
    assert (a.target_month <= 38).all() and (b.origin == 38).all()
    changed = frame.copy()
    changed.loc[changed.MONTH_INDEX > 38, 'MONTHLY_RENT'] *= 3
    aa, bb, _, ss = prepare_phase('selection', 'base', source_frame=changed)
    np.testing.assert_array_equal(a.numeric, aa.numeric)
    np.testing.assert_array_equal(b.numeric, bb.numeric)
    np.testing.assert_array_equal(a.categorical, aa.categorical)
    assert state == ss
    final, val, meta_final, _ = prepare_phase('final_refit', 'base', source_frame=frame)
    assert len(final.row_id) == 150000 and val is None and meta_final['cold_start'] == 3129
    assert np.array_equal(np.sort(final.row_id), np.arange(150000))


def test_macro_future_observations_do_not_change_past():
    from experiments.fusion_contract_v3 import dbs, coe
    daily = dbs.parse_stock_frame()
    changed = daily.copy()
    changed.loc[changed.DATE > '2024-03-31', 'ADJUSTED_CLOSE'] *= 2
    a, b = dbs.build_dbs_frame(daily), dbs.build_dbs_frame(changed)
    for o in (-1, 10, 38):
        x = dbs.scalar_arrays(a, dbs.asof_position(a,o),o)
        y = dbs.scalar_arrays(b, dbs.asof_position(b,o),o)
        np.testing.assert_equal(x[0], y[0]); np.testing.assert_array_equal(x[1], y[1])
    monthly = coe.parse_coe_frame()
    changed = monthly.copy()
    changed.loc[changed.month_index > 38, 'premium'] *= 2
    a, b = coe.build_coe_monthly(monthly), coe.build_coe_monthly(changed)
    for o in (-1, 10, 38):
        x, y = coe.scalar_arrays(a,o), coe.scalar_arrays(b,o)
        np.testing.assert_equal(x[0],y[0]); np.testing.assert_array_equal(x[1],y[1])
