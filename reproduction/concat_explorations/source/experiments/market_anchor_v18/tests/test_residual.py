import unittest
import numpy as np
from experiments.market_anchor_v18.residual import build_inputs, reconstruct

def pop(ids):
 n=len(ids);return {'row_id':np.array(ids),'x_num':np.arange(n*2,dtype=float).reshape(n,2),'x_cat':np.ones((n,1),int),'horizon':np.ones(n),'y_sgd':np.ones(n)*3000}
def anchor(ids):
 n=len(ids);return {'row_id':np.array(ids),'anchor_sgd':np.ones(n)*2200,'count':np.ones(n),'recency':np.ones(n),'level':np.array(['town']*n),'missing':np.zeros(n),'uncertainty_sgd':np.ones(n)*50}
class ResidualTest(unittest.TestCase):
 def test_aligned_fit_only_and_exact_units(self):
  bank={'train':pop([0,1,2]),'val':pop([3,4])}; a={'train':anchor([0,1,2]),'val':anchor([3,4])}
  x,y,state=build_inputs(bank,a); target=(bank['train']['y_sgd']-a['train']['anchor_sgd'])/1000
  np.testing.assert_allclose(reconstruct(target,a['train']['anchor_sgd'],'residual'),bank['train']['y_sgd'])
  bank['val']['y_sgd'][:]=123456; x2,y2,state2=build_inputs(bank,a)
  np.testing.assert_array_equal(x['train'],x2['train']); np.testing.assert_array_equal(x['val'],x2['val'])
  self.assertEqual(state['fingerprint'],state2['fingerprint'])
  a['val']['row_id'][0]=999
  with self.assertRaises(ValueError):build_inputs(bank,a)
 def test_train_ridge_reconstructs_and_does_not_fit_validation(self):
  from experiments.market_anchor_v18.train import fit_predict
  bank={'train':pop([0,1,2]),'val':pop([3,4])}; a={'train':anchor([0,1,2]),'val':anchor([3,4])}
  bank['train']['y_sgd'][:]=[2000,3000,4000]
  first=fit_predict(bank,a,'ridge','residual'); bank['val']['y_sgd'][:]=999999
  second=fit_predict(bank,a,'ridge','residual')
  np.testing.assert_allclose(first['val_prediction'],second['val_prediction'])
  np.testing.assert_allclose(first['val_prediction'],a['val']['anchor_sgd']+1000*first['val_model_output'])
 def test_evaluate_groups_and_paired_error(self):
  from experiments.market_anchor_v18.evaluate import evaluate
  p=pop([1,2]); p.update(month=np.array([39,40]),town=np.array(['A','B']),flat_type=np.array(['x','x']))
  a=anchor([1,2]); m=evaluate(p,a,np.array([2800.,3200.]),baseline=np.array([2700.,3300.]))
  self.assertAlmostEqual(m['rmse_sgd'],200.); self.assertAlmostEqual(m['paired_mean_squared_error_delta'],-50000.)
  self.assertEqual(m['by_town']['A']['n'],1)
  p['block_known']=np.array([True,False]); self.assertEqual(evaluate(p,a,np.array([2800.,3200.]))['by_block_known']['unseen']['n'],1)
 def test_run_saves_aligned_contributions_and_resumes(self):
  from experiments.market_anchor_v18.train import run
  from tempfile import TemporaryDirectory
  from pathlib import Path
  import json
  bank={'train':pop([0,1,2]),'val':pop([3,4])}; a={'train':anchor([0,1,2]),'val':anchor([3,4])}
  for p in bank.values(): p.update(month=np.ones(len(p['row_id'])),town=np.array(['A']*len(p['row_id'])),flat_type=np.array(['x']*len(p['row_id'])))
  with TemporaryDirectory() as d:
   run(bank,a,'ridge','residual',Path(d)); data=json.loads((Path(d)/'result.json').read_text())
   self.assertTrue(data['complete']); self.assertEqual(len((Path(d)/'predictions.csv').read_text().splitlines()),3)
   run(bank,a,'ridge','residual',Path(d))
 def test_paired_comparison_checks_rows(self):
  from experiments.market_anchor_v18.evaluate import paired
  a={'row_id':np.array([1,2]),'y':np.array([3000.,3000.]),'pred':np.array([2800.,3200.])}; b={'row_id':np.array([1,2]),'y':np.array([3000.,3000.]),'pred':np.array([2700.,3300.])}
  self.assertAlmostEqual(paired(a,b),-50000.)
  b['row_id'][0]=9
  with self.assertRaises(ValueError): paired(a,b)
