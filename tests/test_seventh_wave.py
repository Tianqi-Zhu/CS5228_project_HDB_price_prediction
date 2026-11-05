"""Transformer adapters retain the label and preprocessing boundaries."""
import unittest
import importlib.util
from unittest.mock import patch
import numpy as np
import pandas as pd
import sys
import torch
from analysis.experiments.seventh_wave_transformers import context_indices,frame_inputs,numeric_arrays


class SeventhWaveTests(unittest.TestCase):
    def test_context_membership_never_depends_on_rents(self):
        spec={'family':'nori15','context':50}
        first=context_indices({'y_fit':np.arange(100)},spec)
        second=context_indices({'y_fit':np.arange(100)[::-1]*10000},spec)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(len(np.unique(first)),50)

    def test_tabdpt_retrieval_has_access_only_to_supplied_fitting_pool(self):
        rows=context_indices({'y_fit':np.zeros(80)}, {'family':'tabdpt13','context':20})
        np.testing.assert_array_equal(rows,np.arange(80))

    def test_mature_pool_requires_calendar_warmup_and_supported_past_cohort(self):
        bundle={'y_fit':np.zeros(300),'fit_month':np.r_[np.zeros(100),np.full(100,24),np.full(100,25)],
            'ann_fit':pd.DataFrame({'ANN_BLOCK_24_R1_N':np.r_[np.ones(200),np.zeros(100)]})}
        rows=context_indices(bundle,{'family':'tabdpt13','context':20,'mature':True})
        np.testing.assert_array_equal(rows,np.arange(100,200))

    def test_ordinal_encoder_does_not_fit_query_categories_or_fill_query_nan(self):
        fit=pd.DataFrame({'kind':['a','b'],'number':[1.,2.]})
        query=pd.DataFrame({'kind':['new'],'number':[np.nan]})
        encoder,numbers,values=numeric_arrays(fit,query)
        self.assertTrue(np.isnan(values[0,0]))
        self.assertEqual(values[0,-1],0.)
        self.assertEqual(encoder.cardinalities,[3])

    def test_schema_and_residual_target_are_source_only(self):
        x=pd.DataFrame({'a':[1.,2.,3.],'empty':[np.nan]*3,'constant':[1.]*3})
        q=pd.DataFrame({'a':[4.],'empty':[100.],'constant':[999.]})
        base=np.array([100.,200.,300.]);offset=np.array([400.])
        bundle={'y_fit':np.array([110.,220.,330.]),'y_val':np.array([99999.])}
        with patch('analysis.experiments.seventh_wave_transformers.inputs',return_value=(x,q,base,offset)):
            first=frame_inputs(bundle,{'target':'residual'},np.arange(3))
            bundle['y_val'][:]=-99999.
            second=frame_inputs(bundle,{'target':'residual'},np.arange(3))
        self.assertEqual(list(first[0]),['a'])
        pd.testing.assert_frame_equal(first[1],second[1])
        np.testing.assert_array_equal(first[2],[10.,20.,30.])
        np.testing.assert_array_equal(first[3],[400.])

    def test_ratio_target_uses_past_baseline_and_rejects_zero_baseline(self):
        x=pd.DataFrame({'a':[1.,2.]});q=pd.DataFrame({'a':[3.]})
        bundle={'y_fit':np.array([150.,500.])}
        with patch('analysis.experiments.seventh_wave_transformers.inputs',return_value=(x,q,np.array([100.,200.]),np.array([400.]))):
            _,_,target,offset=frame_inputs(bundle,{'target':'ratio'},np.arange(2))
        np.testing.assert_array_equal(target,[1.5,2.5])
        np.testing.assert_array_equal(offset,[400.])
        with patch('analysis.experiments.seventh_wave_transformers.inputs',return_value=(x,q,np.array([0.,200.]),np.array([400.]))):
            with self.assertRaises(ValueError):
                frame_inputs(bundle,{'target':'ratio'},np.arange(2))

    def test_supervised_adapter_backpropagates_only_into_selected_author_layers(self):
        from analysis.experiments.frontier_search import ROOT
        from analysis.experiments.seventh_wave_adaptation import set_trainable
        for folder in ('transformer-deps','foundation-deps'):
            sys.path.insert(0,str(ROOT/'.cache'/folder))
        if importlib.util.find_spec('tabdpt') is None:
            self.skipTest('Optional TabDPT package absent; prepare preflight --profile wave07.')
        from tabdpt.model import TabDPTModel
        torch.set_num_threads(2)
        model=TabDPTModel(dropout=0.,enc_cell_dim=8,n_out=3,regression_bin_count=8,
            regression_bin_min=-4.,regression_bin_max=4.,nhead=2,nhid=32,ninp=16,
            nlayers=3,num_features=4,base_len=4,max_len=100,y_encoder_dim=8,use_flash=False)
        model.eval();set_trainable(model)
        output=model(torch.randn(1,6,4),torch.randn(1,4),False)
        output.square().mean().backward()
        self.assertIsNone(model.encoder.weight.grad)
        self.assertIsNone(model.transformer_encoder[0].q_proj.weight.grad)
        self.assertIsNotNone(model.transformer_encoder[-1].q_proj.weight.grad)
        self.assertIsNotNone(model.head[-1].weight.grad)


if __name__=='__main__':
    unittest.main()
