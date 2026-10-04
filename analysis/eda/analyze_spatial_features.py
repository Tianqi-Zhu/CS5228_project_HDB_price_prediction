"""Reproducible descriptive spatial feature EDA; no predictive model fitting."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.eda.spatial_common import OUTPUT, load_spatial_training

OUT = OUTPUT
TABLES = OUT / 'tables'
FIG = OUT / 'figures'

def corr(a, b, rank=False):
    if rank:
        a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]

def main(frame=None):
    TABLES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_spatial_training() if frame is None else frame
    y = pd.to_numeric(df.MONTHLY_RENT, errors='coerce')
    numeric = [c for c in df.select_dtypes(include='number') if c not in ['MONTHLY_RENT', 'SOURCE_ROW_ID', 'ID']]
    months = df.APPROVAL_MONTH_INDEX
    recent = months >= months.max() - 11
    controls = {'APPROVAL_YEAR', 'APPROVAL_MONTH', 'APPROVAL_MONTH_INDEX', 'APPROVAL_MONTH_SIN', 'APPROVAL_MONTH_COS', 'FLOOR_AREA_SQM'}
    rows = []
    for c in numeric:
        x = pd.to_numeric(df[c], errors='coerce')
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        rv = valid & recent
        adjusted = np.nan
        n_adjusted = 0
        if c not in controls and x[valid].nunique() > 1:
            av = valid & df.FLOOR_AREA_SQM.notna() & months.notna() & df.FLAT_TYPE_CLEAN.notna()
            # Dummy controls rebuilt on the exact pairwise-valid subset. Standardize
            # area; least squares handles redundant/rank-deficient columns safely.
            sub = df.loc[av]
            area = sub.FLOOR_AREA_SQM.to_numpy(float)
            area = (area - area.mean()) / area.std()
            dummy = pd.get_dummies(sub[['APPROVAL_MONTH_INDEX', 'FLAT_TYPE_CLEAN']].astype(str), drop_first=True, dtype=float).to_numpy()
            design = np.column_stack([np.ones(len(sub)), area, dummy])
            pair = np.column_stack([x[av].to_numpy(), y[av].to_numpy()])
            pair = (pair - pair.mean(axis=0)) / pair.std(axis=0)
            coefficients = np.linalg.lstsq(design, pair, rcond=None)[0]
            # einsum avoids platform BLAS matmul floating-point warning artifacts.
            fitted = np.einsum('ij,jk->ik', design, coefficients)
            residual = pair - fitted
            assert np.isfinite(residual).all(), f'Non-finite residuals for {c}'
            adjusted = corr(residual[:, 0], residual[:, 1])
            n_adjusted = len(sub)
        rows.append(dict(feature=c, n_valid=int(valid.sum()), missing_or_nonfinite=int((~valid).sum()), unique_values=int(x[valid].nunique()), pearson=corr(x[valid], y[valid]), spearman=corr(x[valid], y[valid], True), adjusted_pearson=adjusted, n_adjusted=n_adjusted, recent_n=int(rv.sum()), recent_pearson=corr(x[rv], y[rv]), recent_spearman=corr(x[rv], y[rv], True)))
    stats = pd.DataFrame(rows)
    stats['abs_spearman'] = stats.spearman.abs()
    stats['abs_adjusted_pearson'] = stats.adjusted_pearson.abs()
    stats['raw_strength'] = pd.cut(stats.abs_spearman, [-np.inf, .3, .5, np.inf], labels=['weak', 'moderate', 'high'], right=False)
    stats = stats.sort_values('abs_spearman', ascending=False)
    stats.to_csv(TABLES / 'all_numeric_correlations.csv', index=False)
    stats.sort_values('abs_adjusted_pearson', ascending=False).to_csv(TABLES / 'adjusted_correlations.csv', index=False)
    stats[['feature','recent_n','recent_pearson','recent_spearman','spearman']].sort_values('recent_spearman', key=lambda s: s.abs(), ascending=False).to_csv(TABLES / 'recent_12month_correlations.csv', index=False)
    groups = []
    distances = [c for c in numeric if c.startswith('NEAREST_') or c == 'DISTANCE_TO_CBD_KM']
    for c in distances + [c for c in numeric if c.endswith('COUNT_1000M')]:
        edges = [-np.inf, 0, 1, 2, 3, 5, np.inf] if 'COUNT_' in c else ([0, .3, .6, 1, 1.5, 2, np.inf] if c != 'DISTANCE_TO_CBD_KM' else [0, 3, 6, 9, 12, 15, np.inf])
        bins = pd.cut(df[c], edges, include_lowest=True, right=True)
        for label, vals in y.groupby(bins, observed=False):
            groups.append(dict(feature=c, bin=str(label), n=int(vals.count()), mean=vals.mean(), median=vals.median(), q25=vals.quantile(.25), q75=vals.quantile(.75), iqr=vals.quantile(.75)-vals.quantile(.25)))
    grouped = pd.DataFrame(groups)
    grouped.to_csv(TABLES / 'distance_and_count_bins.csv', index=False)
    plt.rcParams.update({'figure.dpi':140, 'font.size':9})
    def bars(col, file, title):
        top = stats.dropna(subset=[col]).sort_values(col, key=lambda s:s.abs()).tail(15)
        fig, ax = plt.subplots(figsize=(9,6)); ax.barh(top.feature, top[col], color=np.where(top[col]>=0,'#2878a5','#d77942')); ax.axvline(0,color='black',lw=.7); ax.set(xlim=(-1,1), xlabel='Correlation coefficient', title=title); fig.tight_layout(); fig.savefig(FIG/file); plt.close(fig)
    bars('spearman','01_raw_spearman.png','Full train: top absolute Spearman correlations')
    bars('adjusted_pearson','02_adjusted_pearson.png','Partial Pearson: month category + flat type + area controlled')
    selected = stats.dropna(subset=['spearman']).head(14).feature.tolist()
    matrix = df[selected].corr(method='spearman'); matrix.to_csv(TABLES/'feature_redundancy_spearman.csv')
    fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r'); ax.set_xticks(range(len(selected)),selected,rotation=90); ax.set_yticks(range(len(selected)),selected); fig.colorbar(im,ax=ax,label='Spearman'); ax.set_title('Feature redundancy: selected top 14 raw associations'); fig.tight_layout(); fig.savefig(FIG/'03_feature_redundancy.png'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,c in zip(axes,['NEAREST_OPEN_MRT_KM','NEAREST_MALL_KM','DISTANCE_TO_CBD_KM']):
        v=df[c].notna() & y.notna(); im=ax.hexbin(df.loc[v,c],y[v],gridsize=45,mincnt=1,bins='log',cmap='viridis'); ax.set(xlabel=c,ylabel='Monthly rent (SGD)'); fig.colorbar(im,ax=ax,label='Record count (log scale)')
    fig.tight_layout(); fig.savefig(FIG/'04_distance_hexbin.png'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,c in zip(axes,['NEAREST_OPEN_MRT_KM','NEAREST_MALL_KM','NEAREST_PRIMARY_SCHOOL_KM']):
        g=grouped[grouped.feature==c]; ax.errorbar(range(len(g)),g['median'],yerr=[g['median']-g.q25,g.q75-g['median']],fmt='o-',capsize=3); ax.set_xticks(range(len(g)),g.bin,rotation=45,ha='right'); ax.set(title=c,ylabel='Rent median with IQR (SGD)');
        for i,(_,r) in enumerate(g.iterrows()): ax.annotate(f'n={r.n:,}',(i,r.q75),fontsize=6,rotation=30)
    fig.tight_layout(); fig.savefig(FIG/'05_distance_bins.png'); plt.close(fig)
    spatial=stats[stats.feature.str.contains('NEAREST|COUNT_|DISTANCE_|LATITUDE|LONGITUDE|MAX_FLOOR|BUILDING_AGE|YEAR_COMPLETED')].dropna(subset=['spearman','recent_spearman'])
    stable = spatial.sort_values('abs_spearman',ascending=False).head(10).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,6)); positions=np.arange(len(stable)); ax.barh(positions-.18,stable.spearman,height=.35,label='Full train'); ax.barh(positions+.18,stable.recent_spearman,height=.35,label='Recent 12 months'); ax.set_yticks(positions,stable.feature); ax.axvline(0,color='gray',lw=.5); ax.set(xlabel='Spearman correlation',title='Spatial/property association direction stability'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'06_recent_stability.png'); plt.close(fig)
    def md_table(frame, cols):
        result=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
        for _,r in frame.iterrows(): result.append('| '+' | '.join(f'{r[c]:.4f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
        return '\n'.join(result)
    raw=spatial.sort_values('abs_spearman',ascending=False).head(10)
    adj=spatial.sort_values('abs_adjusted_pearson',ascending=False).head(10)
    text=f'''# 空间与房屋特征：月租关联分析

训练集共 {len(df):,} 行；仅使用训练租金标签进行描述分析，未训练预测模型。最近 12 个月定义为 APPROVAL_MONTH_INDEX 最大值往前 11 个月（含两端），共 {recent.sum():,} 行。

## 特征与统计方法

内存特征表保留原始列和行序，增加规范化地址/类别、楼栋经纬度、最高楼层、竣工年份与区域、时间、距租约起始年数和楼龄，以及 MRT、商场、小学和 CBD 球面直线距离与 500m/1km/2km 内设施坐标数。字段定义见 [feature_dictionary.csv](feature_dictionary.csv)，实现以 `analysis/pipeline/auxiliary.py` 为准。MAX_FLOOR 是楼栋最高楼层；租约年数不等于楼龄。

所有数值列（排除目标、ID、SOURCE_ROW_ID）均在 [完整相关表](tables/all_numeric_correlations.csv) 中报告 Pearson、Spearman、有效样本数、缺失/非有限值数和唯一值数；常数列保留并给出空相关值。Spearman 使用平均秩后的 Pearson，无显著性检验。按绝对 Spearman 值排序；|r|≥0.5 为高、0.3≤|r|<0.5 为中等、<0.3 为弱，仅为描述标签，绝非显著性或预测价值阈值。

调整关联采用标准化后的 x、租金分别对“审批年月类别、规范化户型类别、面积”进行最小二乘残差化，再对两组残差计算 Pearson。每个特征使用其与目标和控制项均有效的记录；控制项本身不列入调整排名。设计矩阵存在共线性时 numpy 最小二乘仍可给出投影残差。这是线性控制后的关联，不是因果效果，也不是调整后的 Spearman。日历月类别只用于本次训练集描述，不建议直接用于未来月份外推。

## 空间及楼栋特征实际排名

全体数值特征按绝对 Spearman 排名（含原始面积与时间控制）：

{md_table(stats.head(8),['feature','n_valid','pearson','spearman'])}

达到高相关描述阈值的全部特征：{', '.join(stats.loc[stats.abs_spearman>=.5,'feature']) or '无'}。时间趋势强，不能把其相关强度归于空间特征。

按全训练集绝对 Spearman 排名：

{md_table(raw,['feature','n_valid','pearson','spearman','recent_spearman'])}

按绝对调整 Pearson 排名：

{md_table(adj,['feature','adjusted_pearson','n_adjusted'])}

完整数值特征排名、调整排名和近期排名分别见 [原始相关](tables/all_numeric_correlations.csv)、[调整关联](tables/adjusted_correlations.csv)、[近期窗口](tables/recent_12month_correlations.csv)。上述空间/楼栋特征中，全样本最高绝对 Spearman 为 {spatial.abs_spearman.max():.4f}；达到高相关阈值的个数为 {(spatial.abs_spearman>=.5).sum()}，达到中等区间的个数为 {((spatial.abs_spearman>=.3)&(spatial.abs_spearman<.5)).sum()}。弱单变量关联不能证明特征对预测无用；经纬度线性或秩相关弱，也不能证明空间位置不重要，区域关系可能非线性且与户型、面积和时间交互。

![全样本秩相关](figures/01_raw_spearman.png)
![控制混杂后的线性关联](figures/02_adjusted_pearson.png)
![特征冗余](figures/03_feature_redundancy.png)

## 距离分箱与近期稳定性

[分箱统计](tables/distance_and_count_bins.csv) 提供每组 n、均值、中位数、Q25、Q75 和 IQR；距离按公里分组，1km 设施计数按 0、1、2、3、4–5、6+ 分组。不同组的房型、年份、面积和区域构成可能不同，原始分箱差异不能解释为邻近设施造成的租金溢价。最近 12 月排名仅检查样本窗口下的方向和强度变化，不是独立验证集。

![距离与租金密度](figures/04_distance_hexbin.png)
![距离分箱中位数及四分位范围](figures/05_distance_bins.png)
![近期关联稳定性](figures/06_recent_stability.png)

## 数据边界和后续使用

设施表是当前快照，open/planned 状态不是逐月历史状态；近期或历史记录都可能含时间错配。距离是球面直线距离，不是步行距离。设施计数按唯一坐标位置，不是不同 MRT 线路/站代码的数量；共址记录去重规则见构造清单。全部 11 个 planned MRT 坐标与 open 站坐标重合，planned 距离实际反映这些共址站点子集，不能解释为新增独立交通可达性，也不能解释为历史可用交通。

同栋、同房屋属性或月份出现多次交易，记录并非独立同分布；本报告未给出依赖独立性假设的 p 值。报告完整样本的重复记录权重，后续应按时间切分并检查楼栋/地点重复影响。高度冗余的不同半径计数、距离、年份及楼龄见相关矩阵；不应把各自相关值当作独立证据。

所有构造特征未使用租金或测试目标；调整分析仅使用训练标签。若后续采用 KNN，应在训练划分中拟合缺失处理、缩放和编码，再变换验证/测试数据；经纬度、公里距离、平方米和年份不能未经缩放直接混合。应采用时间验证评估新增空间特征的增益，不能由本次相关排名替代预测验证。

复现：在仓库根目录运行 `python -m analysis.main spatial-controls`。特征在内存中由统一管线生成，输出六张图和五张统计 CSV；源数据不修改。
'''
    (OUT/'EDA_summary_zh.md').write_text(text,encoding='utf-8')
    print(stats[['feature','spearman','adjusted_pearson']].to_string(index=False))
    print('Wrote',OUT)

if __name__ == '__main__':
    main()
