import numpy as np
import torch
from experiments.tabfm_finetune_v13.data_full import prepare_phase
from experiments.tabfm_finetune_v13.macro_raw import make_source


def test_real_raw_origin_mapping_and_cold_start():
    fit, val, meta, _ = prepare_phase('selection','base')
    source = make_source(fit, meta)
    assert len(fit.row_id) == 114842 and len(val.row_id) == 35158
    assert source.fit_origins == list(range(-1,38))
    assert source.origins == list(range(-1,39))
    assert source.n_tokens == 2 and source.d_model == 48
    assert source.dbs_mask[0].sum() == 0 and source.coe_mask[0].sum() == 0
    tokens = source({'origin':torch.tensor([-1,38,-1,0])})
    assert tokens.shape == (4,2,48) and torch.isfinite(tokens).all()
    torch.testing.assert_close(tokens[0],tokens[2])
    assert source.describe()['scaler_fit_origins'] == list(range(-1,38))
    assert source.dbs_seq.shape == (40,63,7) and source.coe_seq.shape == (40,12,24)


def test_validation_macro_changes_do_not_refit_training_scalers(monkeypatch):
    from experiments.tabfm_finetune_v13 import macro_raw
    fit, _, meta, _ = prepare_phase('selection','base')
    original = make_source(fit,meta)
    stock = macro_raw.dbs.parse_stock_frame()
    stock.loc[stock.DATE >= '2024-03-01', 'ADJUSTED_CLOSE'] *= 2
    monthly = macro_raw.coe.parse_coe_frame()
    monthly.loc[monthly.month_index >= 38, 'premium'] *= 2
    monkeypatch.setattr(macro_raw.dbs,'parse_stock_frame',lambda:stock)
    monkeypatch.setattr(macro_raw.coe,'parse_coe_frame',lambda:monthly)
    changed = make_source(fit,meta)
    assert original.dbs_scaler == changed.dbs_scaler
    assert original.coe_scaler == changed.coe_scaler
    torch.testing.assert_close(original.dbs_seq[:-1],changed.dbs_seq[:-1])
    torch.testing.assert_close(original.coe_seq[:-1],changed.coe_seq[:-1])
    assert not torch.equal(original.dbs_seq[-1],changed.dbs_seq[-1])
    assert not torch.equal(original.coe_seq[-1],changed.coe_seq[-1])
