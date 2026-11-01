import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.experiments.amenity_quality_features import school_fields,mall_fields,rail_graph,station_bus_counts,operational_date
from analysis.experiments.amenity_quality_sources import parse_school_page
from analysis.compare_amenity_quality import information_origins
from analysis.refine_amenity_bus_names import canonical_stop_name


class QualityTests(unittest.TestCase):
    def test_published_bus_abbreviations_preserve_station_identity(self):
        for abbreviated,expanded in [("W'lands Nth Stn",'woodlands north Stn'),
                                      ("S'goon Stn",'serangoon Stn'),
                                      ("C'wealth Stn",'commonwealth Stn'),
                                      ('Botanic Gdns Stn','Botanic gardens Stn'),
                                      ('6th Ave Stn','sixth avenue Stn'),
                                      ('Holland V Stn','holland village Stn'),
                                      ('Upp Thomson Stn','upper Thomson Stn'),
                                      ('Yishun Nth Stn','Yishun north Stn')]:
            self.assertEqual(canonical_stop_name(abbreviated),expanded)
        self.assertEqual(canonical_stop_name('Northshore Dr'),'Northshore Dr')

    def test_school_availability_and_monotonic_access(self):
        schools=pd.DataFrame({'school_key':['a','b']})
        records=pd.DataFrame({'school_key':['a','b','a'],'year':[2020,2020,2023],
                              'log_demand':[1.,-1.,100.],'available_at':['2020-10-01','2020-10-01','2023-10-01']})
        cfg={'school_radius_m':1000,'school_decay_m':500}
        distance=np.array([[100.,1500.],[900.,1500.]])
        before=school_fields(distance,schools,records,'2020-09-30',cfg)
        self.assertTrue(before.AQ_SCHOOL_EXCESS_ACCESS_LOG.isna().all())
        a=school_fields(distance,schools,records,'2021-05-31',cfg)
        b=school_fields(distance,schools,records.iloc[:2],'2021-05-31',cfg)
        pd.testing.assert_frame_equal(a,b)
        self.assertGreater(a.AQ_SCHOOL_EXCESS_ACCESS_LOG.iloc[0],a.AQ_SCHOOL_EXCESS_ACCESS_LOG.iloc[1])
        self.assertEqual(a.AQ_SCHOOL_NEAREST_LOG_DEMAND.iloc[0],1.)

    def test_mall_unknown_not_zero_and_published_units(self):
        malls=pd.DataFrame({'NAME':['known','unknown']})
        records=pd.DataFrame({'mall':['known'],'year':[2020],'available_at':['2021-05-01'],'nla_sqm':[10000.],
                              'revenue_sgd_m':[20.],'footfall_m':[10.]})
        cfg={'mall_radius_m':2000,'mall_decay_m':750}
        distance=np.array([[100.,500.],[3000.,100.]])
        before=mall_fields(distance,malls,records,'2021-04-30',cfg)
        self.assertTrue(before.AQ_MALL_NLA_ACCESS_LOG.isna().all())
        result=mall_fields(distance,malls,records,'2021-05-31',cfg)
        self.assertAlmostEqual(result.AQ_MALL_NEAREST_LOG_NLA.iloc[0],np.log1p(10000))
        self.assertEqual(result.AQ_MALL_COVERAGE_2KM.iloc[0],.5)
        self.assertTrue(np.isnan(result.AQ_MALL_NEAREST_LOG_NLA.iloc[1]))
        self.assertTrue(np.isnan(result.AQ_MALL_NLA_ACCESS_LOG.iloc[1]))

    def test_bus_services_unique_across_directions_and_distance_guard(self):
        stops=json.dumps({'1':[103.85,1.3,'Town Stn','Road'],'2':[103.8501,1.3,'Opp Town Stn'],'3':[103.95,1.4,'Town Stn']})
        services=json.dumps({'10':{'routes':[['1'],['2']]},'20':{'routes':[['2']]},'30':{'routes':[['3']]}})
        p=Path('synthetic_test_inputs')
        with patch.object(Path,'read_text',side_effect=[stops,services]):
            nodes=pd.DataFrame({'NAME':['town mrt station'],'LONGITUDE':[103.85],'LATITUDE':[1.3]},index=['town'])
            named,nearby,audit=station_bus_counts(nodes,(pd.Timestamp('2021-01-01'),p/'stops.json',p/'services.json'),
                            {'station_named_bus_radius_m':500,'station_bus_radius_m':400})
            self.assertEqual(named[0],2); self.assertEqual(nearby[0],2)
            self.assertEqual({r['stop_id'] for r in audit},{'1','2'})
            unknown=station_bus_counts(nodes,None,{})
            self.assertTrue(np.isnan(unknown[0][0]))

    def test_real_rail_dates_no_future_and_correct_branch(self):
        stations=pd.read_csv('data/auxiliary/sg-mrt-stations.csv')
        nodes,active,g=rail_graph(stations,'2024-05-31')
        for name in ['hume','punggolcoast','tanjongrhu','brickland','sungeikadut','keppel']:
            self.assertNotIn(name,nodes.index)
        self.assertTrue(g.has_edge('promenade','bayfront'))
        self.assertTrue(g.has_edge('bayfront','marinabay'))
        self.assertFalse(g.has_edge('harbourfront','marinabay'))
        old,_,_=rail_graph(stations,'2021-06-30')
        self.assertNotIn('springleaf',old.index)
        middle,_,_=rail_graph(stations,'2021-09-30')
        self.assertIn('springleaf',middle.index)
        self.assertEqual(operational_date('te22a'),None)

    def test_external_only_warmup_origin(self):
        x=pd.DataFrame({'a':[1,2]},index=[7,9])
        b={'X_train':x,'train_month':['2021-01','2021-02']}
        base={'train_temporal_origins':pd.DataFrame({'forecast_origin':pd.PeriodIndex([None,'2021-01'],freq='M')},index=[7,9])}
        result=information_origins(b,base,'train')
        self.assertEqual(result.loc[7],pd.Period('2020-12','M'))
        self.assertEqual(result.loc[9],pd.Period('2021-01','M'))

    def test_archived_school_columns_pre_and_post_rule_change(self):
        p=Path('.cache/tabpfn_amenity_quality_80_20/sources')
        old=parse_school_page(p/'schools_2020_all.html',2020).set_index('school_key')
        new=parse_school_page(p/'schools_2023_all.html',2023).set_index('school_key')
        self.assertEqual(old.loc['anglochinesejunior','vacancies'],25)
        self.assertEqual(old.loc['anglochinesejunior','applications'],41)
        self.assertEqual(new.loc['anglochinesejunior','vacancies'],62)
        self.assertEqual(new.loc['anglochinesejunior','applications'],79)


if __name__=='__main__': unittest.main()
