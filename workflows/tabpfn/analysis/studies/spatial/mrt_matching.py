"""Descriptive MRT contrasts within overlapping observed covariate strata."""
import argparse
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.studies.spatial.common import OUTPUT, load_spatial_training

OUT = OUTPUT / 'mrt_matched'
LABELS = ['[0,0.3]', '(0.3,0.6]', '(0.6,1.0]', '(1.0,1.5]', '(1.5,2.0]', '(2.0,inf)']
COV = ['FLOOR_AREA_SQM', 'BUILDING_AGE_YEARS', 'MAX_FLOOR', 'DISTANCE_TO_CBD_KM']


def prepare(frame, area_width=10, age_width=5):
    data = frame.copy()
    data['distance_bin'] = pd.cut(data.NEAREST_OPEN_MRT_KM, [0, .3, .6, 1, 1.5, 2, np.inf], labels=LABELS, right=True, include_lowest=True)
    data['area_bin'] = np.floor(data.FLOOR_AREA_SQM / area_width)
    data['age_bin'] = np.floor(data.BUILDING_AGE_YEARS / age_width)
    data['building'] = data[['TOWN_CLEAN', 'BLOCK_CLEAN', 'STREET_CLEAN']].astype(str).agg('|'.join, axis=1)
    return data


def compare(data, keys, group, reference=LABELS[0]):
    """Each supported stratum receives h=n0*n1/(n0+n1) weight."""
    valid = data.dropna(subset=keys + ['distance_bin', 'MONTHLY_RENT'])
    arms = [valid[valid.distance_bin.eq(x)] for x in [reference, group]]
    agg = []
    for arm in arms:
        means = arm.groupby(keys, observed=True)[['MONTHLY_RENT'] + COV].mean()
        means['n'] = arm.groupby(keys, observed=True).size()
        agg.append(means)
    joined = agg[0].join(agg[1], how='inner', lsuffix='_ref', rsuffix='_other')
    raw = arms[1].MONTHLY_RENT.mean() - arms[0].MONTHLY_RENT.mean()
    result = {'reference_bin': reference, 'distance_bin': group, 'raw_difference_sgd': raw, 'supported_strata': len(joined), 'reference_total_rows': len(arms[0]), 'other_total_rows': len(arms[1])}
    balance = []
    if joined.empty:
        result.update(standardized_difference_sgd=np.nan, reference_supported_rows=0, other_supported_rows=0, reference_unique_buildings=0, other_unique_buildings=0, overlap_weight_sum=0)
        return result, balance
    h = joined.n_ref * joined.n_other / (joined.n_ref + joined.n_other)
    w = h/h.sum()
    result['standardized_difference_sgd'] = float((w * (joined.MONTHLY_RENT_other - joined.MONTHLY_RENT_ref)).sum())
    result['standardized_reference_mean_sgd'] = float((w*joined.MONTHLY_RENT_ref).sum())
    result['standardized_other_mean_sgd'] = float((w*joined.MONTHLY_RENT_other).sum())
    result['overlap_weight_sum'] = h.sum()
    for label, arm in zip(['reference', 'other'], arms):
        supported = arm.set_index(keys).index.isin(joined.index)
        result[label+'_supported_rows'] = int(supported.sum())
        result[label+'_unique_buildings'] = int(arm.loc[supported, 'building'].nunique())
    for cov in COV:
        # Pair-specific nonmissing support; means alone must not conceal missingness.
        ok = joined[[cov+'_ref', cov+'_other']].notna().all(axis=1)
        wc = h[ok]/h[ok].sum()
        standardized_delta = float((wc*(joined.loc[ok, cov+'_other'] - joined.loc[ok, cov+'_ref'])).sum()) if ok.any() else np.nan
        pooled_sd = np.sqrt((arms[0][cov].var() + arms[1][cov].var())/2)
        raw_delta = arms[1][cov].mean() - arms[0][cov].mean()
        balance.append({'reference_bin': reference, 'distance_bin': group, 'covariate': cov, 'raw_mean_difference': raw_delta, 'standardized_mean_difference': standardized_delta, 'raw_smd': raw_delta/pooled_sd if pooled_sd > 0 else np.nan, 'standardized_smd': standardized_delta/pooled_sd if pooled_sd > 0 else np.nan, 'balance_supported_strata': int(ok.sum()), 'reference_missing_rows': int(arms[0][cov].isna().sum()), 'other_missing_rows': int(arms[1][cov].isna().sum())})
    return result, balance


class Checks(unittest.TestCase):
    def test_distance_bin_boundaries(self):
        frame=pd.DataFrame({'NEAREST_OPEN_MRT_KM':[0,.3,.6,1,1.5,2,2.01], 'FLOOR_AREA_SQM':[50]*7, 'BUILDING_AGE_YEARS':[10]*7, 'TOWN_CLEAN':['a']*7, 'BLOCK_CLEAN':['1']*7, 'STREET_CLEAN':['x']*7})
        self.assertEqual(prepare(frame).distance_bin.astype(str).tolist(),[LABELS[0],LABELS[0],LABELS[1],LABELS[2],LABELS[3],LABELS[4],LABELS[5]])

    def fixture(self):
        rows=[]
        for town, base, counts in [('a', 1000, (90, 10)), ('b', 2000, (10, 90))]:
            for group, count in zip(LABELS[:2], counts):
                for i in range(count):
                    rows.append(dict(town=town, distance_bin=group, MONTHLY_RENT=base+100*(group==LABELS[1]), building=town+group, **{c:1. for c in COV}))
        return pd.DataFrame(rows)

    def test_composition_removed(self):
        r,_ = compare(self.fixture(), ['town'], LABELS[1])
        self.assertAlmostEqual(r['raw_difference_sgd'], 900)
        self.assertAlmostEqual(r['standardized_difference_sgd'], 100)

    def test_unsupported_excluded(self):
        data=self.fixture()
        extra=data.iloc[[0]].copy(); extra['town']='unsupported'; extra['distance_bin']=LABELS[1]; extra['MONTHLY_RENT']=100000
        r,_=compare(pd.concat([data,extra]), ['town'], LABELS[1])
        self.assertAlmostEqual(r['standardized_difference_sgd'],100)
        self.assertEqual(r['other_supported_rows'],100)
        r,_=compare(data,['town'],LABELS[-1])
        self.assertEqual(r['supported_strata'],0)
        self.assertTrue(np.isnan(r['standardized_difference_sgd']))


def main(frame=None):
    OUT.mkdir(parents=True, exist_ok=True)
    original=load_spatial_training() if frame is None else frame
    rows=[]; balances=[]
    for specification, area_width, age_width in [('primary_10sqm_5years',10,5),('sensitivity_20sqm_10years',20,10)]:
        data=prepare(original,area_width,age_width)
        keys=['RENT_APPROVAL_DATE','FLAT_TYPE_CLEAN','TOWN_CLEAN','area_bin','age_bin']
        for reference,group in [(LABELS[0],g) for g in LABELS[1:]]+[(LABELS[3],g) for g in LABELS[4:]]:
            r,b=compare(data,keys,group,reference)
            r['specification']=specification
            r['excluded_missing_stratum_or_distance_rows']=int(data[keys+['distance_bin','MONTHLY_RENT']].isna().any(axis=1).sum())
            rows.append(r)
            balances.extend([dict(x,specification=specification) for x in b])
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'contrasts.csv',index=False)
    pd.DataFrame(balances).to_csv(OUT/'covariate_balance.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    x=np.arange(5)
    first=summary.iloc[:5]
    ax.scatter(x-.15,first.raw_difference_sgd,label='Unadjusted composition')
    for i,(name,sub) in enumerate(summary[summary.reference_bin.eq(LABELS[0])].groupby('specification',sort=False)):
        ax.scatter(x+i*.15,sub.standardized_difference_sgd,label=name)
    ax.axhline(0,color='grey',linewidth=.7)
    ax.set(xticks=x,xticklabels=LABELS[1:],xlabel='Nearest open MRT distance (km), relative to 0–0.3 km',ylabel='Monthly rent difference (SGD)',title='Overlapping strata only: descriptive standardized contrasts')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'mrt_contrasts.png',dpi=160);plt.close(fig)
    notes='''# MRT距离：共同分层内的描述性比较\n\n参考组为最近开放MRT距离0≤d≤0.3公里，其余区间均左开右闭（最后一组d>2公里）。主分析在相同批准年月、房型、town、10平方米面积区间、5年楼龄区间内比较；敏感性分析改用20平方米和10年区间。面积和楼龄仅按区间对齐，不是数值完全相同。\n\n每个比较仅保留两组均有记录的分层，不对无共同支持的记录外推。分层权重h=n参考×n比较/(n参考+n比较)，再归一化；报告的是各分层租金均值差的加权平均。不同距离比较的共同支持总体不同，不能把曲线视为同一总体的剂量反应。未按租金匹配，未使用测试集。\n\ncontrasts.csv列出原始均值差、标准化均值差、两组支持记录数、独立建筑数与分层数；记录可能来自重复建筑，不能当作独立家庭抽样。covariate_balance.csv列出面积、楼龄、楼层上限、CBD距离的原始/标准化均值差与SMD（共同使用两组原始合并标准差）。楼层上限是建筑属性，不是出租单位楼层；余额计算仅使用该协变量两臂均非缺失的分层，另列缺失数。\n\n结果不是因果效应：未控制装修、实际楼层、地段细节等；辅助设施是当前快照，不保证批准时已运营。即使主分析支持较少，也同时报告更粗区间敏感性；不能仅选择支持更多或符合预期的一版。未报告置信区间，因为同一建筑多记录及分层权重需专门处理相关性。\n\n'''
    notes+='远距离回升另直接比较1.5–2与2+公里组相对1–1.5公里组。这里估计均值差，不能精确分解原始中位数曲线的回升。\n\n'
    for r in rows:
        notes+=f"- {r['specification']}，{r['distance_bin']}km相对{r['reference_bin']}km：支持{r['supported_strata']}层，两臂{r['reference_supported_rows']}/{r['other_supported_rows']}条、{r['reference_unique_buildings']}/{r['other_unique_buildings']}栋；标准化差{r['standardized_difference_sgd']:.1f} SGD。\n"
    (OUT/'notes_zh.md').write_text(notes)
    print(summary[['specification','distance_bin','supported_strata','reference_supported_rows','other_supported_rows','standardized_difference_sgd']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        unittest.main(argv=['analyze_mrt_matched'],exit=True)
    else:
        main()
