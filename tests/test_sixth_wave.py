"""Author adapter, fitting-only vocabulary, and retrieval information tests."""
import unittest
import importlib.util
import numpy as np
import pandas as pd
import torch
from analysis.experiments.sixth_wave_deep import dependencies,load_realmlp,load_nca,NativeEncoder,nca_encode,sample_nca_context
from analysis.pipeline.config import ROOT


class SixthWaveTests(unittest.TestCase):
    def setUp(self):
        dependencies();torch.set_num_threads(2)

    def require_nca_source(self):
        base = ROOT / '.cache/deep-sources/TALENT'
        if not all((base / relative).is_file() for relative in ('model/models/modernNCA.py', 'model/lib/tabr/utils.py')):
            self.skipTest('Optional ModernNCA author source absent; prepare preflight --profile wave06.')

    def test_native_encoder_does_not_learn_query_categories_or_fill_query_values(self):
        fit=pd.DataFrame({'category':pd.Categorical(['a','b']),'number':[1.,np.nan]})
        encoder=NativeEncoder().fit(fit)
        query=pd.DataFrame({'category':['unseen'],'number':[np.nan]})
        cats,nums=encoder.transform(query)
        self.assertEqual(cats[0,0],0)
        self.assertTrue(np.isnan(nums[0,0]))
        self.assertEqual(encoder.cardinalities,[3])

    def test_official_tabm_uses_independent_member_losses(self):
        if importlib.util.find_spec('tabm') is None:
            self.skipTest('Optional TabM package absent; install requirements/optional_deep_models.txt.')
        import tabm
        model=tabm.TabM.make(n_num_features=2,cat_cardinalities=[3],d_out=1,
            k=4,n_blocks=2,d_block=16,dropout=0.)
        out=model(torch.randn(8,2),torch.ones(8,1,dtype=torch.long)).squeeze(-1)
        self.assertEqual(tuple(out.shape),(8,4))
        target=torch.randn(8)
        member_loss=(out-target[:,None]).square().mean()
        mean_loss=(out.mean(dim=1)-target).square().mean()
        self.assertGreaterEqual(member_loss.item()+1e-6,mean_loss.item())

    def test_realmlp_preprocessing_uses_training_partition_only(self):
        if not (ROOT / '.cache/deep-deps/pytabkit/models/realmlp/__init__.py').is_file():
            self.skipTest('Optional RealMLP author package absent; prepare preflight --profile wave06.')
        api=load_realmlp()
        torch.manual_seed(42)
        x=torch.arange(20,dtype=torch.float32)[:,None]
        cats=torch.zeros(20,0,dtype=torch.long)
        y=x*2
        config=api.RealMLPConfig.td_reg(n_epochs=2,batch_size=8,hidden_width=8,
            n_hidden_layers=1,num_emb_type='none',lr=.01,p_drop=0.)
        model,_=api.fit(config,x,cats,[],y,torch.arange(16)[None],torch.arange(16,20)[None],seed=42)
        median=model.x_median.clone()
        changed=x.clone();changed[16:]=100000.
        other,_=api.fit(config,changed,cats,[],y,torch.arange(16)[None],torch.arange(16,20)[None],seed=42)
        torch.testing.assert_close(median,other.x_median)
        self.assertEqual(float(median.flatten()[0]),7.5)

    def test_modernnca_own_training_target_is_masked(self):
        self.require_nca_source()
        cls=load_nca()
        model=cls(d_in=3,d_num=3,d_out=1,dim=8,dropout=0.,d_block=16,n_blocks=1,
            num_embeddings=None,sample_rate=1.)
        query=torch.randn(2,3);context=torch.randn(10,3)
        labels=torch.tensor([1.,2.]);context_y=torch.arange(10,dtype=torch.float32)
        torch.manual_seed(42);before=model(query,labels,context,context_y,True)
        labels[0]=100000.
        torch.manual_seed(42);after=model(query,labels,context,context_y,True)
        self.assertAlmostEqual(float(before[0].detach()),float(after[0].detach()),places=5)

    def test_context_pool_cannot_retrieve_current_training_queries(self):
        query=torch.tensor([2,4,8,10])
        for _ in range(10):
            context=sample_nca_context(100,query,20)
            self.assertFalse(torch.isin(context,query).any())
            self.assertEqual(len(torch.unique(context)),20)

    def test_modernnca_cached_encoding_matches_author_forward(self):
        self.require_nca_source()
        model=load_nca()(d_in=3,d_num=3,d_out=1,dim=8,dropout=0.,d_block=16,
            n_blocks=1,num_embeddings=None,sample_rate=1.)
        model.eval()
        query=torch.randn(4,3);context=torch.randn(10,3);labels=torch.randn(10)
        with torch.no_grad():
            expected=model(query,None,context,labels,False)
            weights=torch.softmax(-torch.cdist(nca_encode(model,query),nca_encode(model,context))/model.T,dim=1)
            actual=weights@labels
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


if __name__=='__main__':
    unittest.main()
