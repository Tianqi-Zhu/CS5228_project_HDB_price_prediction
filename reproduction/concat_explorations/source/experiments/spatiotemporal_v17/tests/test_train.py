import unittest
import numpy as np
from experiments.spatiotemporal_v17.train import validate_bank, metrics

def population(n, offset=0):
    return dict(x_num=np.ones((n,2)), x_cat=np.ones((n,1),dtype=int), y_sgd=np.ones(n)*3000, residual=np.ones(n), anchor_ksgd=np.ones(n)*2, row_id=np.arange(n)+offset, origin=np.ones(n)*38, horizon=np.ones(n), month=np.ones(n), town=np.array(['a']*n), flat_type=np.array(['b']*n))

class TrainTest(unittest.TestCase):
    def test_bank_rejects_target_and_overlap(self):
        b=dict(train=population(3),val=population(2,4),numeric_names=['a','b'],categorical_names=['c'],cardinalities=[2],manifest={})
        validate_bank(b, 3, 2)
        b['val']['row_id'][0]=0
        with self.assertRaises(ValueError): validate_bank(b,3,2)
        b['val']['row_id'][0]=4; b['train']['y_sgd'][0]=0
        with self.assertRaises(ValueError): validate_bank(b,3,2)
    def test_metrics_are_prediction_derived(self):
        result=metrics(population(2),np.array([3000., 3200.]))
        self.assertAlmostEqual(result['rmse_sgd'],np.sqrt(20000))
        self.assertEqual(result['by_town']['a']['n'],2)

class RunnerTest(unittest.TestCase):
    def test_tiny_train_resume_reload(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from experiments.spatiotemporal_v17.train import run
        b=dict(train=population(8),val=population(4,20),numeric_names=['a','b'],categorical_names=['c'],cardinalities=[2],manifest={'test':True})
        with TemporaryDirectory() as d, patch('experiments.spatiotemporal_v17.contract.load_bank',return_value=b):
            run('F1',max_epochs=1,expected=(8,4),output=d)
            self.assertTrue((Path(d)/'result.json').exists())
            self.assertEqual(len((Path(d)/'predictions.csv').read_text().splitlines()),5)
            run('F1',max_epochs=1,expected=(8,4),output=d)

class TemporalRunnerTest(unittest.TestCase):
    def test_temporal_checkpoint_resume_and_identity(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        from experiments.spatiotemporal_v17 import train
        b=dict(train=population(8),val=population(4,20),numeric_names=['a','b'],categorical_names=['c'],cardinalities=[2],sequence_names=['rent'],manifest={'test':True})
        for p in (b['train'],b['val']):
            p['rent_sequence']=np.ones((len(p['row_id']),12,1),np.float32)
            p['sequence_mask']=np.ones_like(p['rent_sequence'],dtype=bool)
        original=train.atomic_save; interrupted=[False]
        def stop_after_checkpoint(data,path):
            original(data,path)
            if str(path).endswith('last.pt') and not interrupted[0]:
                interrupted[0]=True; raise RuntimeError('simulated interruption')
        with TemporaryDirectory() as d, patch('experiments.spatiotemporal_v17.contract.load_bank',return_value=b):
            with patch.object(train,'atomic_save',side_effect=stop_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError,'simulated'): train.run('F0',max_epochs=1,expected=(8,4),output=d,architecture='concat')
            train.run('F0',max_epochs=1,expected=(8,4),output=d,architecture='concat')
            result=json.loads((Path(d)/'result.json').read_text()); self.assertTrue(result['reload_verified'])
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'): train.run('F0',max_epochs=1,expected=(8,4),output=d,architecture='cross')

class DiagnosticsTest(unittest.TestCase):
    def test_lowlr_reports_train_val_bias_gradient_and_schedule(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        from experiments.spatiotemporal_v17.train import run
        b=dict(train=population(8),val=population(4,20),numeric_names=['a','b'],categorical_names=['c'],cardinalities=[2],sequence_names=['rent'],manifest={'test':True})
        for p in (b['train'],b['val']):
            p['rent_sequence']=np.ones((len(p['row_id']),12,1),np.float32); p['sequence_mask']=np.ones_like(p['rent_sequence'],dtype=bool)
        with TemporaryDirectory() as d, patch('experiments.spatiotemporal_v17.contract.load_bank',return_value=b):
            run('F0',max_epochs=1,expected=(8,4),output=d,architecture='cross',lr=.0001,diagnostics=True)
            result=json.loads((Path(d)/'result.json').read_text()); h=result['history'][0]
            for key in ('train_eval_rmse_sgd','train_prediction_bias_sgd','val_prediction_bias_sgd','val_prediction_std_sgd','grad_clip_fraction','preclip_grad_norm_mean','learning_rate_min','learning_rate_max','train_residual_prediction_mean_sgd','train_residual_prediction_std_sgd','train_residual_correlation','val_residual_prediction_mean_sgd','val_residual_prediction_std_sgd','val_residual_correlation','by_month','by_horizon'): self.assertIn(key,h)
            config=json.loads((Path(d)/'config.json').read_text()); self.assertEqual(config['lr'],.0001); self.assertLessEqual(h['learning_rate_max'],.0001)
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'): run('F0',max_epochs=1,expected=(8,4),output=d,architecture='cross',lr=.00003,diagnostics=True)
    def test_schedule_matches_prior_3e4(self):
        import torch
        from experiments.training_schedule_v15 import WarmupCosine
        p=torch.nn.Parameter(torch.tensor(1.)); a=torch.optim.AdamW([p],lr=.0003); b=torch.optim.AdamW([p],lr=.0003)
        old=WarmupCosine(a,total_steps=2360,warmup_steps=59,min_lr_ratio=.01,start_ratio=.01)
        new=WarmupCosine(b,total_steps=59*40,warmup_steps=59,min_lr_ratio=.01,start_ratio=.01)
        for _ in range(100): self.assertEqual(a.param_groups[0]['lr'],b.param_groups[0]['lr']); a.step();b.step();old.step();new.step()

class DiagnosticsResumeTest(unittest.TestCase):
    def test_mid_epoch_checkpoint_keeps_clipping_telemetry(self):
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        import json
        from pathlib import Path
        from experiments.spatiotemporal_v17 import train
        b=dict(train=population(8),val=population(4,20),numeric_names=['a','b'],categorical_names=['c'],cardinalities=[2],manifest={'test':True})
        orig=train.atomic_save; flag=[False]
        def interrupted(obj,path):
            orig(obj,path)
            if str(path).endswith('last.pt') and not flag[0]: flag[0]=True; raise RuntimeError('pause')
        with TemporaryDirectory() as d,patch('experiments.spatiotemporal_v17.contract.load_bank',return_value=b):
            with patch.object(train,'atomic_save',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'pause'): train.run('F0',max_epochs=1,expected=(8,4),output=d,lr=.0001,diagnostics=True)
            train.run('F0',max_epochs=1,expected=(8,4),output=d,lr=.0001,diagnostics=True)
            h=json.loads((Path(d)/'result.json').read_text())['history'][0]; self.assertEqual(h['learning_rate_max'],.0001); self.assertGreater(h['preclip_grad_norm_max'],0)
