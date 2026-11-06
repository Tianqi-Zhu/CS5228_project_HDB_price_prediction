import unittest
import numpy as np
import torch
from analysis.experiments.recency_weighting import recency_weights,weight_summary,training_loss


class RecencyWeightTests(unittest.TestCase):
    def test_calendar_gaps_and_half_life_ratios(self):
        w=recency_weights(['2023-03','2024-03','2025-03'],12)
        np.testing.assert_allclose(w[1:]/w[:-1],[2,2])
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
    def test_uniform_default_and_order_independence(self):
        dates=['2025-03','2021-01','2024-03']
        np.testing.assert_array_equal(recency_weights(dates),np.ones(3))
        np.testing.assert_array_equal(recency_weights(dates,24),recency_weights(dates[::-1],24)[::-1])
        summary=weight_summary(dates,24)
        self.assertEqual(summary['anchor_month'],'2025-03')
        self.assertLess(summary['effective_sample_size'],3)
    def test_invalid_half_lives_and_dates_rejected(self):
        for h in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):recency_weights(['2024-01'],h)
        for dates in [[],['2024-01',None]]:
            with self.assertRaises(ValueError):recency_weights(dates,12)
    def test_recent_sample_has_larger_regression_gradient(self):
        p=torch.ones(2,requires_grad=True)
        w=torch.as_tensor(recency_weights(['2024-03','2025-03'],12))
        training_loss(p,torch.zeros(2),w).backward()
        self.assertAlmostEqual(float(p.grad[1]/p.grad[0]),2.)
    def test_recency_weights_also_apply_to_auxiliary_head(self):
        p=torch.zeros((2,4),requires_grad=True)
        w=torch.as_tensor(recency_weights(['2024-03','2025-03'],12))
        training_loss(p,torch.zeros(2),w,aux_weight=.1,
                      band_targets=torch.tensor([0,0]),weight_auxiliary=True).backward()
        self.assertAlmostEqual(float(p.grad[1,1]/p.grad[0,1]),2.)
    def test_uniform_loss_preserves_original_objective_exactly(self):
        p=torch.tensor([[1.,.3,-.2,.5],[2.,-.4,.8,.6]],requires_grad=True)
        y=torch.tensor([3.,4.]);bands=torch.tensor([1,2]);w=torch.ones(2)
        original=((p[:,0]-y).square()*w).mean()+.1*torch.nn.functional.cross_entropy(p[:,1:],bands)
        actual=training_loss(p,y,w,aux_weight=.1,band_targets=bands)
        torch.testing.assert_close(actual,original,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
