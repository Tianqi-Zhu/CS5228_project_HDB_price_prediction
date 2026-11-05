import numpy as np
import torch
from experiments.concat_raw_v14 import pipeline as p


def test_epoch_order_resume_covers_each_row_once():
    order = p.epoch_order(103, 7)
    assert len(np.unique(order)) == 103
    assert np.array_equal(np.r_[order[:37], p.epoch_order(103, 7)[37:]], order)


def test_metrics_convert_residual_and_anchor_to_sgd():
    assert p.rmse_sgd(np.array([1000., 4000.]), np.array([0., 1.]), np.array([1., 2.])) == np.sqrt(500000.)


def test_build_model_runs_native_macro_and_backpropagates():
    model = p.build_model(['x', 'y'], (3, 4))
    batch = {'numeric': torch.randn(3, 2), 'categorical': torch.tensor([[0,1],[1,2],[2,3]]),
             'dbs_seq': torch.randn(2,63,7), 'dbs_mask': torch.ones(2,63),
             'coe_seq': torch.randn(2,12,24), 'coe_mask': torch.ones(2,12),
             'window_index': torch.tensor([0,1,0])}
    result = model(batch)
    assert result.shape == (3,) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert model.source.encoders.dbs.lstm.weight_ih_l0.grad is not None
