"""MRT descriptive OLS with absorbed month×flat-type and town fixed effects."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statistics import NormalDist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/spatial_features/mrt_adjusted'
LABELS=['0–0.3','0.3–0.6','0.6–1','1–1.5','1.5–2','2+']

def t_critical(df):
    if df<30: raise ValueError('Large-df t approximation requires at least 30 df')
    z=NormalDist().inv_cdf(.975)
    return z+(z**3+z)/(4*df)+(5*z**5+16*z**3+3*z)/(96*df**2)+(3*z**7+19*z**5+17*z**3-15*z)/(384*df**3)

def absorb(values, groups, tol=1e-10):
    result=np.asarray(values,float).copy()
    for iteration in range(1000):
        before=result.copy()
        for codes in groups:
            sums=np.zeros((codes.max()+1,result.shape[1])); np.add.at(sums,codes,result)
            counts=np.bincount(codes)
            result-= (sums/counts[:,None])[codes]
        if np.max(np.abs(result-before))<tol:
            return result,iteration+1
    raise RuntimeError('Fixed-effect projection did not converge')

def fe_rank(a,b):
    na,nb=a.max()+1,b.max()+1
    parent=np.arange(na+nb)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i,j in set(zip(a,b)):
        x,z=find(i),find(na+j)
        if x!=z: parent[x]=z
    return na+nb-len({find(i) for i in range(na+nb)})

def fit(y,x,groups,cluster):
    within,iterations=absorb(np.column_stack([y,x]),groups)
    wy,wx=within[:,0],within[:,1:]
    rank=np.linalg.matrix_rank(wx)
    if rank!=wx.shape[1]: raise ValueError('Non-identified slope columns')
    beta=np.linalg.lstsq(wx,wy,rcond=None)[0]
    residual=wy-np.einsum('ij,j->i',wx,beta)
    bread=np.linalg.inv(np.einsum('ni,nj->ij',wx,wx,optimize=False))
    score=np.zeros((cluster.max()+1,wx.shape[1])); np.add.at(score,cluster,wx*residual[:,None])
    n=len(y); g=len(score); k=int(fe_rank(*groups)+rank)
    meat=np.einsum('gi,gj->ij',score,score,optimize=False)
    covariance=(g/(g-1))*((n-1)/(n-k))*np.einsum('ij,jk,kl->il',bread,meat,bread,optimize=False)
    assert np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance).min()>-1e-7
    return beta,covariance,dict(n=n,clusters=g,full_design_rank=k,slope_rank=int(rank),fe_rank=int(k-rank),projection_iterations=iterations)

def test():
    rng=np.random.default_rng(72); n=600
    a=rng.integers(0,8,n); b=rng.integers(0,4,n); cluster=np.arange(n)//6
    x=rng.normal(size=(n,2)); y=np.einsum('ni,i->n',x,np.array([3.,-2.]),optimize=False)+rng.normal(size=8)[a]+rng.normal(size=4)[b]
    beta,_,_=fit(y,x,[a,b],cluster); np.testing.assert_allclose(beta,[3,-2],atol=1e-9)
    y=y+rng.normal(size=n)
    beta,cov,info=fit(y,x,[a,b],cluster)
    full=np.column_stack([np.ones(n),pd.get_dummies(a,drop_first=True,dtype=float),pd.get_dummies(b,drop_first=True,dtype=float),x])
    reference=np.linalg.lstsq(full,y,rcond=None)[0]
    residual=y-np.einsum('ij,j->i',full,reference)
    inverse=np.linalg.inv(np.einsum('ni,nj->ij',full,full,optimize=False))
    scores=[np.einsum('ni,n->i',full[cluster==g],residual[cluster==g],optimize=False) for g in np.unique(cluster)]
    meat=np.einsum('gi,gj->ij',np.asarray(scores),np.asarray(scores),optimize=False)
    expected=(100/99)*((n-1)/(n-full.shape[1]))*np.einsum('ij,jk,kl->il',inverse,meat,inverse,optimize=False)
    np.testing.assert_allclose(beta,reference[-2:],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(cov,expected[-2:,-2:],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(t_critical(100),1.9839715184496334,atol=1e-7)
    assert info['full_design_rank']==np.linalg.matrix_rank(full)
    print('Synthetic coefficient, cluster covariance and design-rank tests passed')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(ROOT/'data/processed/train_spatial_features.csv')
    needed=['MONTHLY_RENT','NEAREST_OPEN_MRT_KM','APPROVAL_MONTH_INDEX','FLAT_TYPE_CLEAN','TOWN_CLEAN','STREET_CLEAN','BLOCK_CLEAN','FLOOR_AREA_SQM','BUILDING_AGE_YEARS','MAX_FLOOR','DISTANCE_TO_CBD_KM']
    mask=original[needed].notna().all(axis=1)
    numeric=['MONTHLY_RENT','NEAREST_OPEN_MRT_KM','FLOOR_AREA_SQM','BUILDING_AGE_YEARS','MAX_FLOOR','DISTANCE_TO_CBD_KM']
    mask &= np.isfinite(original[numeric]).all(axis=1)
    d=original.loc[mask].copy()
    # Right-closed intervals: exact 0.3 enters 0–0.3, exact 2 enters 1.5–2.
    bins=pd.cut(d.NEAREST_OPEN_MRT_KM,[0,.3,.6,1,1.5,2,np.inf],right=True,include_lowest=True,labels=LABELS)
    assert bins.notna().all()
    month_type=d.APPROVAL_MONTH_INDEX.astype(str)+'|'+d.FLAT_TYPE_CLEAN
    groups=[pd.factorize(month_type)[0],pd.factorize(d.TOWN_CLEAN)[0]]
    building=d.TOWN_CLEAN+'|'+d.STREET_CLEAN+'|'+d.BLOCK_CLEAN
    clusters=pd.factorize(building)[0]
    dist=pd.get_dummies(bins,dtype=float).iloc[:,1:].to_numpy()
    normalized={c:(d[c].to_numpy(float)-d[c].mean())/d[c].std(ddof=0) for c in ['FLOOR_AREA_SQM','BUILDING_AGE_YEARS','MAX_FLOOR','DISTANCE_TO_CBD_KM']}
    x=np.column_stack([dist,normalized['FLOOR_AREA_SQM'],normalized['FLOOR_AREA_SQM']**2,normalized['BUILDING_AGE_YEARS'],normalized['BUILDING_AGE_YEARS']**2,normalized['MAX_FLOOR']])
    y=d.MONTHLY_RENT.to_numpy(float)
    beta,cov,info=fit(y,x,groups,clusters)
    sensitivity,scov,sinfo=fit(y,np.column_stack([x,normalized['DISTANCE_TO_CBD_KM']]),groups,clusters)
    critical=t_critical(info['clusters']-1)
    baseline=y[bins==LABELS[0]].mean()
    records=[]
    for i,label in enumerate(LABELS):
        pick=bins==label; effect=0 if i==0 else beta[i-1]; se=0 if i==0 else np.sqrt(cov[i-1,i-1])
        records.append(dict(distance_bin_km=label,n_rows=int(pick.sum()),n_buildings=int(building[pick].nunique()),raw_mean=float(y[pick].mean()),raw_mean_difference=float(y[pick].mean()-baseline),adjusted_difference=effect,cluster_se=se,ci_low=effect-critical*se,ci_high=effect+critical*se,adjusted_with_cbd=0 if i==0 else sensitivity[i-1],sparse_buildings=building[pick].nunique()<30))
    result=pd.DataFrame(records); result.to_csv(OUT/'mrt_bin_effects.csv',index=False)
    contrasts=[]
    for far in [4,5]:
        v=np.zeros(len(beta)); v[far-1]=1; v[2]=-1
        effect=np.einsum('i,i->',v,beta,optimize=False)
        se=np.sqrt(np.einsum('i,ij,j->',v,cov,v,optimize=False))
        cbd_effect=np.einsum('i,i->',v,sensitivity[:len(beta)],optimize=False)
        cbd_se=np.sqrt(np.einsum('i,ij,j->',v,scov[:len(beta),:len(beta)],v,optimize=False))
        contrasts.append(dict(contrast=f'{LABELS[far]} minus 1–1.5 km',adjusted_difference=effect,cluster_se=se,ci_low=effect-critical*se,ci_high=effect+critical*se,adjusted_with_cbd=cbd_effect,cbd_cluster_se=cbd_se,cbd_ci_low=cbd_effect-critical*cbd_se,cbd_ci_high=cbd_effect+critical*cbd_se))
    ct=pd.DataFrame(contrasts); ct.to_csv(OUT/'far_distance_contrasts.csv',index=False)
    pd.DataFrame(cov).to_csv(OUT/'coefficient_cluster_covariance.csv',index=False)
    qa={**info,'source_rows':len(original),'excluded_rows':int((~mask).sum()),'missing_by_required_column':{key:int(value) for key,value in original[needed].isna().sum().items()},'month_type_levels':int(month_type.nunique()),'town_levels':int(d.TOWN_CLEAN.nunique()),'ci_distribution':'Student t critical value approximated by third-order Cornish-Fisher expansion, df=independent building clusters minus 1','bin_boundaries':'right closed, left open; baseline [0,0.3], final bin (2,infinity)','sensitivity':sinfo}
    (OUT/'analysis_manifest.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,5)); pos=np.arange(6)
    ax.bar(pos-.18,result.raw_mean_difference,width=.35,label='Raw mean difference')
    ax.bar(pos+.18,result.adjusted_difference,width=.35,label='Adjusted OLS difference',yerr=[result.adjusted_difference-result.ci_low,result.ci_high-result.adjusted_difference],capsize=4)
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(pos,LABELS); ax.set(xlabel='Nearest open MRT distance bin (km)',ylabel='Monthly rent difference vs 0–0.3 km (SGD)',title='Raw means and adjusted differences (building-cluster 95% CI)'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'mrt_adjusted_differences.png',dpi=160); plt.close(fig)
    def table(frame):
        lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
        for _,r in frame.iterrows(): lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in r)+' |')
        return '\n'.join(lines)
    report=f'''# MRT 距离：控制房屋属性与时间后的租金差异

完整训练集 {len(original):,} 行；本分析使用共同完整样本 {len(d):,} 行、{info['clusters']:,} 个楼栋，排除 {(~mask).sum():,} 行。缺失项、设计秩和收敛次数见 [分析清单](analysis_manifest.json)。原始均值差与调整差使用完全相同样本；它们均为相对于 0–0.3km 基准组的 SGD/月差异，原始统计不是中位数。

OLS 控制精确“日历月份 × 规范化户型”交互固定效应和 town 固定效应，另控制中心化/标准化面积及其平方、楼龄及其平方、楼栋最高楼层。交互类别 {month_type.nunique()} 个，town {d.TOWN_CLEAN.nunique()} 个；固定效应投影后斜率列全部可识别，完整设计秩 {info['full_design_rank']}。MAX_FLOOR 不是出租单元楼层。固定效应通过迭代去均值吸收，与完整虚拟变量 OLS 等价；小型独立合成数据已与完整虚拟变量设计的系数及直接逐组计算的聚类协方差逐项对照。

95% 区间使用规范化 town+street+block 楼栋聚类 sandwich 协方差，有限样本修正 G/(G−1)×(N−1)/(N−K)，t 分布自由度 G−1。因环境 SciPy 二进制不可导入，t 临界值使用至自由度负三次项的 Cornish–Fisher 展开（大自由度近似，当前临界值 {critical:.8f}；测试对照 df=100 的已知值误差 <1e−7）。不同楼栋独立是假设，同 town/相邻楼栋仍可能共同变化，区间不能完全排除空间依赖。距离分箱以右端点闭合，基准组为 0≤d≤0.3km，精确 2km 属于 1.5–2 组。

{table(result[['distance_bin_km','n_rows','n_buildings','raw_mean_difference','adjusted_difference','ci_low','ci_high','adjusted_with_cbd']])}

![MRT 原始与调整均值差](mrt_adjusted_differences.png)

额外增加 CBD 直线距离的敏感性系数列于 adjusted_with_cbd，仍为同一完整样本。它检查 town 内中心性解释，但不是对所有位置混杂的控制。不同调整组的区间重叠与否不能替代直接组间对比；针对远距离“回升”，直接比较以下两组与 1–1.5km，并列出主模型与加入 CBD 距离后的楼栋聚类标准误和 95% 区间：

{table(ct)}

设施使用当前 open MRT 快照及球面直线距离，不能当成交易当月站点开放历史或步行距离。本模型结果依赖控制项、函数形式和组内共同支持；按 <30 个楼栋标记的稀疏组见 [系数表](mrt_bin_effects.csv)。楼栋数比交易行数更接近独立支持量，计数也不能保证跨 town/户型/月份存在充分重叠。调整关联不能解释为 MRT 对租金的因果影响，亦不代表预测增益。楼栋聚类纠正重复记录不确定性，但不会修正遗漏混杂。

复现：`.venv/bin/python scripts/analyze_mrt_adjusted.py --test`，然后 `.venv/bin/python scripts/analyze_mrt_adjusted.py`。不修改源数据，不训练预测模型。
'''
    (OUT/'MRT_adjusted_summary_zh.md').write_text(report,encoding='utf-8')
    print(result.to_string(index=False)); print(ct.to_string(index=False)); print(json.dumps(qa,indent=2))

if __name__=='__main__':
    import sys
    test() if '--test' in sys.argv else main()
