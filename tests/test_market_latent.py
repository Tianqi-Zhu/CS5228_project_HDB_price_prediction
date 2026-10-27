"""Causality, frozen fitting, and checkpoint round-trip checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis.experiments.market_latent import windows, train_encoder, encode, MarketEncoder


class MarketLatentTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame(np.random.default_rng(5).normal(size=(40, 3)),
                                  index=pd.period_range('2021-01', periods=40, freq='M'),
                                  columns=['rent_growth', 'economic', 'rent_level'])

    def test_same_and_future_month_do_not_enter_window(self):
        x, _, months = windows(self.panel, 6)
        changed = self.panel.copy()
        changed.loc['2023-01':] = 999
        xx, _, _ = windows(changed, 6)
        np.testing.assert_array_equal(x[months <= pd.Period('2023-01')], xx[months <= pd.Period('2023-01')])
        self.assertFalse(np.array_equal(x[-1], xx[-1]))

    def test_future_targets_cannot_change_encoder_and_checkpoint(self):
        torch.set_num_threads(1)
        x, y, months = windows(self.panel, 6)
        future = months >= pd.Period('2023-01')
        for kind in ('rnn','lstm'):
            args = (months, kind, 42, '2023-01', '2022-10')
            net, prep, mu, sd, epoch, _ = train_encoder(x,y,*args,epochs=4)
            xx, yy = x.copy(), y.copy()
            xx[future], yy[future] = 999, -999
            other, _, _, _, other_epoch, _ = train_encoder(xx,yy,*args,epochs=4)
            self.assertEqual(epoch,other_epoch)
            for key, value in net.state_dict().items():
                torch.testing.assert_close(value,other.state_dict()[key],rtol=0,atol=0)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.pt'
                torch.save(net.state_dict(),path)
                restored=MarketEncoder(x.shape[-1],kind)
                restored.load_state_dict(torch.load(path,weights_only=True))
                a,z=encode(net,prep,x,mu,sd)
                b,zz=encode(restored,prep,x,mu,sd)
                np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(z,zz)
                self.assertEqual(z.shape,(len(months),3))
                self.assertTrue(np.isfinite(z).all())


if __name__=='__main__':
    unittest.main()
