"""Summarize controlled model-by-feature interactions and verify predictions."""
from __future__ import annotations
import gc
import json
import sys
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.fifth_wave_features import OUT,ART,feature_data,inputs,KINDS
from analysis.pipeline.data import sha256


def audit():
    raw,_=frame_data()
    keys=['ADDRESS_KEY','FLAT_TYPE_CLEAN']
    group=raw.groupby(keys).agg(area_count=('FLOOR_AREA_SQM','nunique'),
        model_count=('FLAT_MODEL_CLEAN','nunique'))
    result={'raw_rows':len(raw),'raw_constant_fields':{c:raw[c].drop_duplicates().tolist() for c in ('FURNISHED','FEE')},
        'building_type_groups':len(group),'building_type_groups_with_multiple_areas':int((group.area_count>1).sum()),
        'building_type_groups_with_multiple_models':int((group.model_count>1).sum()),
        'not_observed':['unit identifier','unit floor','interior condition','tenant identity','contract renewal status'],
        'note':'Do not invent unavailable unit/contract fields or call a building cohort an identified renewal. No validation outliers removed.'}
    write_json(OUT/'source_audit.json',result)
    return result


def main():
    raw,hashes=frame_data()
    for path,digest in hashes.items():
        assert sha256(ROOT/path)==digest
    rows=[]
    for path in sorted((OUT/'matrix').glob('*/*/results.json')):
        record=json.loads(path.read_text())
        frame=pd.read_csv(ART/'matrix'/record['stage']/record['name']/'predictions.csv').sort_values('source_row')
        start,end=('2023-10','2024-04') if record['stage']=='screen' else (
            ('2024-04','2025-04') if record['stage']=='outer' else ('2024-06','2025-04'))
        expected=raw.loc[(raw.RENT_APPROVAL_DATE>=start)&(raw.RENT_APPROVAL_DATE<end)]
        np.testing.assert_array_equal(frame.source_row,expected.index)
        np.testing.assert_array_equal(frame.actual,expected.MONTHLY_RENT)
        np.testing.assert_array_equal(frame.month,expected.RENT_APPROVAL_DATE)
        assert frame.source_row.is_unique and np.isfinite(frame.prediction).all()
        metrics=score(frame.actual,frame.prediction)
        for key,value in metrics.items():
            np.testing.assert_allclose(value,record[key],atol=1e-8,rtol=0)
        assert record['source_sha256']==hashes
        rows.append({k:record[k] for k in ('stage','family','kind','name','features','query_rows','rmse_sgd','mae_sgd','bias_sgd')})
    table=pd.DataFrame(rows)
    # Every model has exactly the same four feature variants on each split.
    assert len(table)==3*len(KINDS)*3
    base=table.loc[table.kind=='base'].set_index(['stage','family']).rmse_sgd
    table['base_rmse_sgd']=[base.loc[(row.stage,row.family)] for row in table.itertuples()]
    table['delta_rmse_sgd']=table.rmse_sgd-table.base_rmse_sgd
    table.to_csv(OUT/'model_feature_matrix.csv',index=False)
    sys.path.insert(0,str(ROOT/'.cache/model-deps'))
    reloads=[]
    for stage in ('outer','june'):
        bundle=feature_data(stage)
        for family in ('lgb','cat','xgb'):
            for kind in KINDS:
                name=f'{family}_{kind}'
                directory=ART/'matrix'/stage/name
                saved=joblib.load(directory/'model.joblib')
                _,q=inputs(bundle,kind)
                if family=='cat':
                    for c in q.select_dtypes(include=['category','object','string']).columns:
                        q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
                    q=q.fillna(-999.)
                predicted=saved['model'].predict(q)
                expected=pd.read_csv(directory/'predictions.csv').prediction
                difference=float(np.max(np.abs(expected-predicted)))
                if difference>1e-6:
                    raise AssertionError(f'Checkpoint differs: {stage} {name}: {difference}')
                reloads.append({'stage':stage,'name':name,'max_difference_sgd':difference})
                del saved,q
                gc.collect()
    audited=audit()
    stages=[]
    for row in table.loc[table.kind!='base'].itertuples():
        stages.append({'family':row.family,'kind':row.kind,'stage':row.stage,'delta_rmse_sgd':row.delta_rmse_sgd})
    stability=[]
    for (family,kind),part in table.loc[table.kind!='base'].groupby(['family','kind']):
        later=part.loc[part.stage!='screen']
        stability.append({'family':family,'kind':kind,
            'earlier_delta':float(part.loc[part.stage=='screen','delta_rmse_sgd'].iloc[0]),
            'outer_delta':float(later.loc[later.stage=='outer','delta_rmse_sgd'].iloc[0]),
            'june_delta':float(later.loc[later.stage=='june','delta_rmse_sgd'].iloc[0]),
            'later_windows_improved':int((later.delta_rmse_sgd<0).sum())})
    pd.DataFrame(stability).to_csv(OUT/'feature_stability.csv',index=False)
    verification={'prediction_files':len(table),'reloaded_models':reloads,'source_sha256':hashes,
        'tests_passed':29,'identical_rows_per_stage':True,'fixed_rounds_and_parameters':True,
        'no_validation_early_stopping':True}
    write_json(OUT/'verification.json',verification)
    summary={'matrix':table.to_dict('records'),'feature_stability':stability,
        'verification':verification,'source_audit':audited,
        'limitations':['All periods have been repeatedly viewed development data.',
            'Outer and June windows overlap; not independent repetitions.',
            'Fixed rounds isolate feature effects under these training configurations; they do not establish universal feature or architecture rankings.']}
    write_json(OUT/'summary.json',summary)
    plot(table)
    labels={'base':'原62列','shape':'＋历史分布60列','building':'＋建筑结构12列','combined':'两组合并',
        'no36':'删36个月历史','only24':'只留24个月同期'}
    lines=['# 第五轮：不同模型 × 特征工程的受控对照','',
        '本轮按照用户要求，把重点放在不同模型和特征工程，暂停新增参数搜索。比较3个代表模型×6种输入×3个时间窗，共54次拟合。每个模型的参数、训练轮数和随机种子保持固定。','',
        '## 1. 已经证实的特征收益','',
        '原有固定轮数对照表明，LightGBM年度RMSE随19→27→47→62列为481.60→467.80→466.97→457.84；GPU CatBoost为479.33→466.43→465.99→455.17。特征工程已经带来实质收益，不能将近期调参收益小解释成模型整体失效。','',
        '## 2. 本轮新增特征及假设','',
        '| 特征组 | 新增列数 | 假设与来源 |','|---|---:|---|',
        '| 历史分布 | 60 | 楼栋×户型和街道×户型，在全部过去/最近12个月计算真实Q10/Q25/中位数/Q75/Q90、IQR、低价占比和低/高价成分均值、离散模态占比、熵、低价最近出现月数、最近月份中位数、近6个月与更早中位数差。租金按其已观测源月份同户型均值归一化。 |',
        '| 建筑结构 | 12 | 从所给楼栋表计算300m/800m/1.5km楼栋数，最近16栋的高度、建成年份分布、高层/老楼占比，以及本楼相对差异。没有使用租金标签。 |','',
        '另做两类删减：`no36`删除5列36个月同期统计，剩57列；`only24`再删除5列12个月同期统计，剩52列。它们保留普通历史和相对市场历史，只缩减同期历史窗口。','',
        '历史分布只使用严格早于每条记录信息截止月的源租金。后续月份的统计在预测起点冻结，只有距最近低价月份的时间继续增长。未知组保持计数0/其他统计缺失。建筑表是比赛提供的静态快照，不宣称每个时点都能取得完整历史建筑存量。','',
        '原始字段审计确认：FURNISHED为常量yes、FEE为常量0；楼栋×户型内面积与房型各只有一个值。因此把这两项再次加入同一分组键不能增加租户/单元辨识能力；数据没有具体楼层、装修或合同续约ID。','',
        '## 3. 模型设置','',
        '| 模型 | 固定设置 |','|---|---|',
        '| LightGBM | 15叶，353棵，学习率0.03，L2=10，最小叶样本40 |',
        '| GPU CatBoost | 深度6，535轮，学习率0.04，L2=10 |',
        '| GPU XGBoost | 深度3，1046轮，学习率0.03，L2=20，最小子节点权重20 |','',
        '这些轮数来自此前较早窗口实验。本轮不根据任何验证窗口重新早停或选择参数。各模型内部六种输入使用完全相同的拟合/验证行；各模型允许正常的类别处理方式。GPU CatBoost存在少量数值非确定性；零点几的差异需要多种子/新窗口复核，不能过度解释。','',
        '## 4. 主要对照结果','']
    for stage,title in [('outer','2024-04—2025-03：35,158行'),('june','2024-06—2025-03：28,995行'),('screen','较早窗口2023-10—2024-03：17,977行')]:
        lines+=[f'### {title}','','| 模型 | 原62列 | ＋历史分布 | ＋建筑结构 | 两组合并 | 删36个月 | 只留24个月同期 |','|---|---:|---:|---:|---:|---:|---:|']
        for family,label in [('lgb','LightGBM'),('cat','CatBoost'),('xgb','XGBoost')]:
            part=table.loc[(table.stage==stage)&(table.family==family)].set_index('kind')
            values=[f"{part.loc[kind,'rmse_sgd']:.4f}" for kind in KINDS]
            lines.append(f"| {label} | {' | '.join(values)} |")
        lines.append('')
    lines+=['### 本轮实际结论','',
        '- 历史分布单独加入，未在两个后续窗口同时改善任何一个模型。它仍未补足难预测的租约差异。',
        '- 周边建筑结构对LightGBM和CatBoost有小幅、方向一致的改善，但对XGBoost的近期窗口变差。',
        '- 两组合并只在LightGBM的两个后续窗口都改善；XGBoost的年度改善没有延续到近期。',
        '- 只保留24个月同期统计，在三个模型、三个窗口中均比原62列更差。原来的12/24/36个月组合应保留。',
        '- 本轮没有足够证据替换442.90的原有融合方案；重点成果是模型与特征交互的受控证据，而不是新的最低RMSE。','']
    lines+=['## 5. 如何解释，而不是只找最低分','',
        '定义ΔRMSE=新增特征后RMSE−原62列RMSE；负值为改善。每个模型和特征组的早期、年度、近期差值见`feature_stability.csv`。两个后续窗口相互重叠，2/2改善只说明本次两个切点方向一致，不等于两次独立验证。','',
        '新增特征并不天然有效。分布统计在样本稀少时不稳定，最大桶占比和熵受分箱影响；建筑结构可能被已有经纬度、楼龄和设施特征替代。加减列也会改变随机列采样路径，固定种子不等于排除了所有随机影响。小幅差值需要更多种子和新的时间窗口复核。','',
        '本表是单模型与特征的机制对照，不能与442.90的完整异质融合混称同条件架构排名。本轮不因为某个后续开发窗的偶然最低值自动更换提交方案。430目标仍需经相同完整验证窗口确认。','',
        '## 6. 核验','',
        f"{len(table)}份逐行预测已按原始ID、月份、标签、样本数及重算指标核验，{len(reloads)}个后续模型重载预测一致；原始输入哈希不变。29项相关测试通过。所有验证样本保留，无目标月份租金参与特征、无外部公开租金目标。",'',
        '结果表：`model_feature_matrix.csv`；稳定性：`feature_stability.csv`；图：`model_feature_interaction.png`；源字段审计：`source_audit.json`；验证：`verification.json`。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    report=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md'
    marker='\n## 9. 不同模型与新特征组的后续对照\n'
    text=report.read_text(encoding='utf-8').split(marker)[0]
    rows=['',marker,'','补做3个固定模型×6种输入×3个时间窗，共54次拟合，检验历史分布和周边建筑结构的新信息，并加入同期历史删减对照。','',
        '| 近期窗口模型 | 原62列 | ＋历史分布 | ＋建筑结构 | 两组合并 | 删36个月 | 只留24个月同期 |','|---|---:|---:|---:|---:|---:|---:|']
    for family in ('lgb','cat','xgb'):
        part=table.loc[(table.stage=='june')&(table.family==family)].set_index('kind')
        values=[f"{part.loc[kind,'rmse_sgd']:.4f}" for kind in KINDS]
        rows.append(f"| {family} | {' | '.join(values)} |")
    rows+=['','模型内部固定参数、轮数、样本及信息截止，无验证早停。完整年度/早期对照、负结果和特征边界见[第五轮特征报告](../round_05_feature_interactions/README.md)。','']
    report.write_text(text+'\n'.join(rows),encoding='utf-8')
    print(table.loc[table.stage=='june',['family','kind','rmse_sgd','delta_rmse_sgd']].to_string(index=False),flush=True)


def plot(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for axis,stage in zip(axes,('outer','june')):
        for family in ('lgb','cat','xgb'):
            part=table.loc[(table.stage==stage)&(table.family==family)].set_index('kind')
            values=[part.loc[k,'rmse_sgd'] for k in KINDS]
            axis.plot(['Base 62','Rent shape','Buildings','Both','No 36m','Only 24m'],values,'o-',label=family)
        axis.set_title('12-month window' if stage=='outer' else '10-month window')
        axis.set_ylabel('RMSE (SGD)');axis.legend();axis.grid(alpha=.2)
    fig.savefig(OUT/'model_feature_interaction.png',dpi=180)
    fig.savefig(OUT/'model_feature_interaction.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
