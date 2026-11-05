"""Structural equations, trainable numeric tokens and label-free inference."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from analysis.experiments.sixth_wave_deep import dependencies
from analysis.experiments.tenth_wave_structures import CrossLayer,ExpertNetwork,build,predict_saved


class TenthWaveTests(unittest.TestCase):
    def setUp(self):dependencies();torch.set_num_threads(2)

    def test_cross_layer_matches_explicit_full_matrix_equation(self):
        layer=CrossLayer(3)
        x0=torch.tensor([[1.,2.,3.],[2.,1.,-1.]])
        x=torch.tensor([[.1,.2,.3],[.4,.5,.6]])
        expected=x+x0*(x@layer.linear.weight.T+layer.linear.bias)
        torch.testing.assert_close(layer(x0,x),expected)
        layer(x0,x).sum().backward()
        self.assertIsNotNone(layer.linear.weight.grad)

    def test_expert_prediction_is_soft_convex_combination(self):
        nums=np.random.default_rng(42).normal(size=(20,3)).astype(np.float32)
        spec={'embedding':'plr','experts':4}
        model=ExpertNetwork(nums,[3],spec).eval()
        values=torch.tensor(nums);cats=torch.ones(20,1,dtype=torch.long)
        gates,experts=model.components(values,cats);result=model(values,cats)
        torch.testing.assert_close(gates.sum(dim=1),torch.ones(20))
        self.assertTrue(torch.all(result>=experts.min(dim=1).values-1e-6))
        self.assertTrue(torch.all(result<=experts.max(dim=1).values+1e-6))
        result.square().mean().backward()
        self.assertIsNotNone(model.gate[-1].weight.grad)
        self.assertTrue(all(expert[-1].weight.grad is not None for expert in model.experts))

    def test_both_numeric_token_variants_train_through_author_attention(self):
        nums=np.random.default_rng(42).normal(size=(20,3)).astype(np.float32)
        for kind in ('ple','plr'):
            model=build({'family':'ft_'+kind,'embedding':kind,'width':16,'blocks':1},nums,[3])
            output=model(torch.tensor(nums),torch.ones(20,1,dtype=torch.long))
            self.assertEqual(tuple(output.shape),(20,1))
            output.square().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.cont_embeddings.parameters()))

    def test_saved_predict_does_not_consume_query_targets(self):
        class Encoder:
            def transform(self,x):return np.ones((len(x),1),dtype=np.int64),np.ones((len(x),2),dtype=np.float32)
        class Model:
            def to(self,device):return self
        payload={'encoder':Encoder(),'columns':['v'],'model':Model(),'mean':10.,'std':2.}
        query=pd.DataFrame({'v':[1.,2.]});bundle={'y_val':np.array([1.e9,-1.e9])}
        with patch('analysis.experiments.third_wave_training.inputs',return_value=(None,query,None,np.array([100.,200.]))), \
             patch('analysis.experiments.tenth_wave_structures.tensor_arrays',side_effect=lambda a,b:(a,b)), \
             patch('analysis.experiments.tenth_wave_structures.inference',return_value=np.array([1.,2.])):
            a=predict_saved(payload,bundle);bundle['y_val']*=1000.;b=predict_saved(payload,bundle)
        np.testing.assert_array_equal(a,[112.,214.]);np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
