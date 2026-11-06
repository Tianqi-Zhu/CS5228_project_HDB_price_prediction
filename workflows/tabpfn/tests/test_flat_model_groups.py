import unittest
import pandas as pd
from analysis.flat_model_groups import model_group, apply_model_group


class FlatModelGroupTests(unittest.TestCase):
    def test_exact_proposed_merges_and_preserved_distinctions(self):
        values=pd.Series([' Type  S1 ','type s2','model a maisonette','premium maisonette','improved maisonette',
                          'maisonette','model a','model a2','apartment','premium apartment','premium apartment loft','dbss'])
        self.assertEqual(model_group(values).tolist(),['type s','type s','maisonette','maisonette','maisonette',
                          'maisonette','model a','model a2','apartment','premium apartment','premium apartment loft','dbss'])

    def test_only_direct_model_feature_changes_and_inputs_are_immutable(self):
        x=pd.DataFrame({'FLAT_MODEL_CLEAN':pd.Categorical(['type s1','model a maisonette','model a']),
                        'HIST_SIGNATURE_MEAN':[1000.,2000.,3000.]},index=[3,8,10])
        q=pd.DataFrame({'FLAT_MODEL_CLEAN':pd.Categorical(['type s2','new unseen']),
                        'HIST_SIGNATURE_MEAN':[1500.,2500.]},index=[11,12])
        before_x,before_q=x.copy(deep=True),q.copy(deep=True)
        a,b=apply_model_group(x,q,'grouped')
        self.assertEqual(a.FLAT_MODEL_CLEAN.tolist(),['type s','maisonette','model a'])
        self.assertEqual(b.loc[11,'FLAT_MODEL_CLEAN'],'type s')
        self.assertTrue(pd.isna(b.loc[12,'FLAT_MODEL_CLEAN']))
        pd.testing.assert_series_equal(a.HIST_SIGNATURE_MEAN,x.HIST_SIGNATURE_MEAN)
        self.assertNotIn('new unseen',a.FLAT_MODEL_CLEAN.cat.categories)
        pd.testing.assert_frame_equal(x,before_x)
        pd.testing.assert_frame_equal(q,before_q)

    def test_added_group_preserves_all_original_columns(self):
        x=pd.DataFrame({'FLAT_MODEL_CLEAN':pd.Categorical(['type s1','type s2','maisonette']), 'area':[90,100,110]})
        q=x.iloc[:1].copy()
        a,b=apply_model_group(x,q,'original_plus_group')
        pd.testing.assert_frame_equal(a[x.columns],x)
        pd.testing.assert_frame_equal(b[q.columns],q)
        self.assertEqual(a.FLAT_MODEL_GROUP.tolist(),['type s','type s','maisonette'])
        self.assertEqual(a.FLAT_MODEL_GROUP.cat.categories.tolist(),['type s','maisonette'])

    def test_unseen_raw_subtype_maps_to_known_parent_without_query_vocab(self):
        x=pd.DataFrame({'FLAT_MODEL_CLEAN':pd.Categorical(['maisonette','type s1'])})
        q=pd.DataFrame({'FLAT_MODEL_CLEAN':pd.Categorical([None,None])},index=[5,6])
        raw_train=pd.Series(['maisonette','type s1'])
        raw_query=pd.Series(['improved maisonette','unseen family'],index=[5,6])
        a,b=apply_model_group(x,q,'original_plus_group',raw_train,raw_query)
        self.assertEqual(b.loc[5,'FLAT_MODEL_GROUP'],'maisonette')
        self.assertTrue(pd.isna(b.loc[6,'FLAT_MODEL_GROUP']))
        self.assertNotIn('unseen family',a.FLAT_MODEL_GROUP.cat.categories)
        pd.testing.assert_series_equal(b.FLAT_MODEL_CLEAN,q.FLAT_MODEL_CLEAN)


if __name__=='__main__':unittest.main()
