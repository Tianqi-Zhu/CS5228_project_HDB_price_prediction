import unittest
import pandas as pd
from analysis.experiments.canonical_rental_dedup import canonical_training_rows


def record(**changes):
    r=dict(RENT_APPROVAL_DATE='2023-01',TOWN='bedok',BLOCK='1',STREET='main road',FLAT_TYPE='4-room',
        FLAT_MODEL='improved',FLOOR_AREA_SQM=90.,FURNISHED='yes',LEASE_COMMENCE_DATE=1980,FEE=0.,MONTHLY_RENT=3000.)
    r.update(changes); return r


class DedupTests(unittest.TestCase):
    def test_future_labels_cannot_change_retained_context(self):
        context=pd.DataFrame([record(),record()],index=[1,2])
        future=pd.DataFrame([record(RENT_APPROVAL_DATE='2024-01'),record(RENT_APPROVAL_DATE='2024-01')],index=[3,4])
        alone,_=canonical_training_rows(context)
        full,_=canonical_training_rows(pd.concat([context,future]))
        self.assertEqual(full.loc[full.RENT_APPROVAL_DATE<'2024-01'].index.tolist(),alone.index.tolist())
        future['MONTHLY_RENT']=[100.,100000.]
        changed,_=canonical_training_rows(pd.concat([context,future]))
        self.assertEqual(changed.loc[changed.RENT_APPROVAL_DATE<'2024-01'].index.tolist(),alone.index.tolist())

    def test_rebuilt_history_really_excludes_removed_observations(self):
        from analysis.data import prepare_main,normalized_address
        from analysis.experiments.aligned_temporal_features import origin_history
        raw=pd.DataFrame([record(MONTHLY_RENT=1000),record(MONTHLY_RENT=1000),record(MONTHLY_RENT=3000)])
        clean,_=canonical_training_rows(raw)
        past=normalized_address(prepare_main(clean)); past['_MONTH']=pd.PeriodIndex(past.RENT_APPROVAL_DATE,freq='M'); past['PLANNING_AREA']='bedok'
        query=past.iloc[:1].copy(); query['_MONTH']=pd.Period('2023-02'); query['MONTHLY_RENT']=1e12
        features=origin_history(past,query)
        self.assertEqual(features.HIST_BLOCK_FLAT_COUNT.iloc[0],2)
        self.assertEqual(features.HIST_BLOCK_FLAT_MEAN.iloc[0],2000)
        self.assertEqual(features.RECENT_BLOCK_FLAT_3M_COUNT.iloc[0],2)

    def test_aliases_and_addresses_collapse_but_stable_first_id_is_kept(self):
        raw=pd.DataFrame([record(TOWN=' BEDOK ',STREET=' Main  Road ',FLAT_TYPE='4 room'),record()],index=[90,3])
        original=raw.copy(deep=True)
        clean,mapping=canonical_training_rows(raw)
        self.assertEqual(clean.index.tolist(),[90]); self.assertEqual(mapping.retained_row_id.tolist(),[90])
        self.assertEqual(mapping.removed_row_id.tolist(),[3]); self.assertEqual(clean.iloc[0].FLAT_TYPE,'4-room')
        pd.testing.assert_frame_equal(raw,original)

    def test_different_rent_month_or_property_attribute_is_retained(self):
        raw=pd.DataFrame([record(),record(MONTHLY_RENT=3100),record(RENT_APPROVAL_DATE='2023-02'),
                          record(FLOOR_AREA_SQM=91),record(BLOCK='2'),record(FURNISHED='no'),record(FEE=10)])
        clean,mapping=canonical_training_rows(raw)
        self.assertEqual(len(clean),7); self.assertEqual(len(mapping),0)

    def test_duplicate_mapping_and_idempotence(self):
        raw=pd.DataFrame([record(),record(),record(),record(MONTHLY_RENT=3100)],index=[4,8,9,12])
        clean,mapping=canonical_training_rows(raw)
        self.assertEqual(clean.index.tolist(),[4,12]); self.assertEqual(mapping.retained_row_id.tolist(),[4,4])
        twice,second=canonical_training_rows(clean)
        pd.testing.assert_frame_equal(clean,twice); self.assertEqual(len(second),0)

    def test_unit_identifier_when_available_prevents_collapsing_distinct_units(self):
        raw=pd.DataFrame([record(UNIT_ID='01-01'),record(UNIT_ID='02-01')])
        clean,_=canonical_training_rows(raw); self.assertEqual(len(clean),2)

    def test_no_hidden_test_target_or_duplicate_source_ids(self):
        with self.assertRaises(ValueError): canonical_training_rows(pd.DataFrame([record()]).drop(columns='MONTHLY_RENT'))
        with self.assertRaises(ValueError): canonical_training_rows(pd.DataFrame([record(),record()],index=[1,1]))


if __name__=='__main__': unittest.main()
