"""Verify second-wave experiments, serialized predictors, and matched comparisons."""
from __future__ import annotations
import gc
import json
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.second_wave_stacking import OUT,ART
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.second_wave_anniversary import data as cohort_data,select_inputs as cohort_inputs
from analysis.experiments.second_wave_normalized import make_data as normal_data,select_inputs as normal_inputs
from analysis.pipeline.data import sha256


def matched_file(path):
    frame=pd.read_csv(path).rename(columns={'approval_month':'month'}).sort_values('source_row').reset_index(drop=True)
    return frame


def monthly_comparison(new,old):
    np.testing.assert_array_equal(new.source_row,old.source_row)
    np.testing.assert_array_equal(new.actual,old.actual)
    rows=[]
    for month,part in new.groupby('month'):
        mask=new.month.eq(month)
        previous=old.loc[mask]
        rows.append({'month':month,'n':len(part),
            'new_rmse':score(part.actual,part.prediction)['rmse_sgd'],
            'old_rmse':score(previous.actual,previous.prediction)['rmse_sgd'],
            'new_sse':float(np.square(part.prediction-part.actual).sum()),
            'old_sse':float(np.square(previous.prediction-previous.actual).sum())})
    table=pd.DataFrame(rows)
    generator=np.random.default_rng(42)
    n_months=len(table)
    starts=generator.integers(0,n_months-2,size=(5000,int(np.ceil(n_months/3))))
    indices=(starts[:,:,None]+np.arange(3)).reshape(5000,-1)[:,:n_months]
    n=table.n.to_numpy()[indices].sum(axis=1)
    delta=np.sqrt(table.new_sse.to_numpy()[indices].sum(axis=1)/n)-np.sqrt(table.old_sse.to_numpy()[indices].sum(axis=1)/n)
    return table,{'delta_rmse_sgd':score(new.actual,new.prediction)['rmse_sgd']-score(old.actual,old.prediction)['rmse_sgd'],
        'month_block_interval_95':np.quantile(delta,[.025,.975]).tolist(),
        'months_improved':int((table.new_rmse<table.old_rmse).sum()),'months':n_months}


def reload_primary(stage,selection):
    start='2024-04' if stage=='outer' else '2024-06'
    cohort=cohort_data(start,'2025-04')
    normal=None
    checked=[]
    for name,item in selection['members'].items():
        artifact=ART/item['kind']/stage/item['name']
        saved=joblib.load(artifact/'model.joblib')
        model,spec=saved['model'],saved['spec']
        if item['kind']=='anniversary':
            _,query=cohort_inputs(cohort,spec['kind'])
            level=np.ones(len(query))
        else:
            if normal is None:
                normal=normal_data(start,'2025-04')
            _,query,_,_,_,level=normal_inputs(normal,spec)
        if spec['family']=='cat':
            for column in query.select_dtypes(include=['category','object','string']).columns:
                query[column]=query[column].astype('string').fillna('__MISSING__').astype(str)
            query=query.fillna(-999.)
        predicted=np.asarray(model.predict(query),float)
        if item['kind']=='normalized_market':
            predicted=predicted*level if spec['target']=='ratio' else predicted+level
        expected=matched_file(artifact/'predictions.csv')
        error=float(np.max(np.abs(expected.prediction.to_numpy()-predicted)))
        if error>1e-6:
            raise AssertionError(f'Reload failed: {stage} {name} {error}')
        checked.append({'stage':stage,'name':name,'max_difference_sgd':error})
        del saved,model,query
        gc.collect()
    return checked


def main():
    source,hashes=frame_data()
    checks=[]
    for path in sorted(OUT.rglob('results.json')):
        relative=path.parent.relative_to(OUT)
        prediction_path=ART/relative/'predictions.csv'
        if not prediction_path.exists():
            continue
        record=json.loads(path.read_text())
        predictions=matched_file(prediction_path)
        rows=predictions.source_row.to_numpy(int)
        np.testing.assert_array_equal(predictions.actual,source.iloc[rows].MONTHLY_RENT)
        np.testing.assert_array_equal(predictions.month,source.iloc[rows].RENT_APPROVAL_DATE)
        if not predictions.source_row.is_unique or not np.isfinite(predictions.prediction).all():
            raise AssertionError('Invalid prediction rows')
        metrics=score(predictions.actual,predictions.prediction)
        for key in metrics:
            np.testing.assert_allclose(metrics[key],record[key],atol=1e-8,rtol=0)
        if 'source_sha256' in record and record['source_sha256']!=hashes:
            raise AssertionError('Source fingerprints changed')
        checks.append({'experiment':str(relative),'rows':len(predictions),**metrics})
    pd.DataFrame(checks).to_csv(OUT/'verified_metrics.csv',index=False)
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    reloads=reload_primary('outer',selection)+reload_primary('june',selection)
    comparisons={}
    for stage,old_path in [('outer',ROOT/'output/models/round_01_history_target_search/outer/locked_ensemble/predictions.csv'),
        ('june',ROOT/'output/models/round_01_history_target_search/matched_june/ensemble_predictions.csv')]:
        new=matched_file(ART/'ensemble'/stage/'predictions.csv')
        old=matched_file(old_path)
        table,delta=monthly_comparison(new,old)
        table.to_csv(OUT/f'monthly_comparison_{stage}.csv',index=False)
        # Rebuild the blend from component predictions without model refitting.
        combined=np.zeros(len(new))
        for item in selection['members'].values():
            part=matched_file(ART/item['kind']/stage/item['name']/'predictions.csv')
            np.testing.assert_array_equal(part.source_row,new.source_row)
            combined+=item['weight']*part.prediction.to_numpy()
        np.testing.assert_allclose(combined,new.prediction,atol=1e-8,rtol=0)
        comparisons[stage]={'rows':len(new),'new':score(new.actual,new.prediction),
            'previous_round':score(old.actual,old.prediction),**delta}
    stress=json.loads((OUT/'stress_16m/summary.json').read_text())
    stress_predictions=matched_file(ART/'stress_16m/ensemble_predictions.csv')
    stress_old=matched_file(ROOT/'output/experiment_runs/comparison_model_search/2023-12/xgboost/predictions.csv')
    stress_months,stress_delta=monthly_comparison(stress_predictions,stress_old)
    stress_months.to_csv(OUT/'stress_16m/monthly_comparison.csv',index=False)
    np.testing.assert_allclose(score(stress_predictions.actual,stress_predictions.prediction)['rmse_sgd'],stress['scores']['rmse_sgd'],atol=1e-8,rtol=0)
    # Matched feature-group comparison with the same tree family and temporal protocol.
    ablation=[]
    for stage,old_path in [('outer',ROOT/'output/models/round_01_history_target_search/outer/compact_plus_relative_history_15/predictions.csv'),
        ('june',ROOT/'output/models/round_01_history_target_search/matched_june/compact_plus_relative_history_15/predictions.csv')]:
        old=matched_file(old_path)
        new=matched_file(ART/'anniversary'/stage/'window_lgb15'/'predictions.csv')
        np.testing.assert_array_equal(old.source_row,new.source_row)
        np.testing.assert_array_equal(old.actual,new.actual)
        before,after=score(old.actual,old.prediction),score(new.actual,new.prediction)
        ablation.append({'stage':stage,'base47_rmse':before['rmse_sgd'],
            'plus_anniversary62_rmse':after['rmse_sgd'],'delta':after['rmse_sgd']-before['rmse_sgd']})
    pd.DataFrame(ablation).to_csv(OUT/'anniversary_feature_comparison.csv',index=False)
    counts={'stacking_configurations':len(json.loads((OUT/'stacking_plan.json').read_text())['trials']),
        'tabicl_screens':len(list((OUT/'tabicl/screen').glob('*/results.json'))),
        'normalized_market_screens':len(list((OUT/'normalized_market/screen').glob('*/results.json'))),
        'anniversary_screens':len(list((OUT/'anniversary/screen').glob('*/results.json'))),
        'focused_extension_screens':len(list((OUT/'cohort_extensions/screen').glob('*/results.json')))}
    foundation_sample=json.loads((OUT/'tabicl/cohort_sample_comparison.json').read_text())
    summary={'counts':counts,'comparisons':comparisons,'stress_16m':stress,
        'stress_month_sensitivity':stress_delta,'feature_group_comparison':ablation,
        'previous_tabpfn_reported_rmse_sgd':455.07102836,
        'delta_vs_previous_tabpfn_same_june_window':comparisons['june']['new']['rmse_sgd']-455.07102836,
        'target_430_achieved':comparisons['june']['new']['rmse_sgd']<=430.,
        'source_sha256':hashes,
        'limitations':['All dates have been repeatedly used for development; no untouched final test exists.',
            'One random seed; interval summaries measure fixed-prediction month sensitivity, not selection or future-market uncertainty.',
            'The 16-month window overlaps the other windows and is never pooled with them.',
            'Anniversary cohorts do not identify individual tenants or apartments.',
            'Static supplied amenity snapshots are not a complete historical availability audit.']}
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'verification.json',{'prediction_files_checked':len(checks),
        'source_targets_months_and_row_ids_match':True,'source_files_unchanged':True,
        'scores_independently_recomputed':True,'primary_component_reloads':reloads,
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'analysis/experiments').glob('round_02_anniversary_history*.py'))},
        'versions':{name:version(name) for name in ('numpy','pandas','scikit-learn','lightgbm','catboost','torch')}})
    importance=joblib.load(ART/'anniversary/june/window_cat6/model.joblib')['model']
    pd.DataFrame({'feature':importance.feature_names_,'importance':importance.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/'anniversary_feature_importance.csv',index=False)
    lines=['# 第二轮租金预测探索', '',
        f"本轮筛选 **{sum(counts.values())}个配置**：31组误差校正、14组TabICL、24组市场归一化、11组同期特征、8组精简/交互。使用较早期预测选择方案和融合权重。", '',
        '**有用的新信息是同楼栋、同户型的历史同期租金。400–430目标尚未达到。**', '',
        '| 验证窗口 | 行数 | 上轮融合 RMSE | 本轮主融合 RMSE | 本轮 MAE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for stage,label in [('outer','2024-04 至 2025-03'),('june','2024-06 至 2025-03')]:
        r=comparisons[stage]
        lines.append(f"| {label} | {r['rows']:,} | {r['previous_round']['rmse_sgd']:.4f} | **{r['new']['rmse_sgd']:.4f}** | {r['new']['mae_sgd']:.4f} |")
    lines += [f"| 2023-12 至 2025-03，16个月 | {stress['rows']:,} | {stress['previous_xgboost']['rmse_sgd']:.4f}（原XGBoost） | **{stress['scores']['rmse_sgd']:.4f}** | {stress['scores']['mae_sgd']:.4f} |", '',
        f"此前TabPFN分支在相同六月起点/28,995行上的最低开发记录为455.0710。本轮443.8630低约 **{455.07102836-comparisons['june']['new']['rmse_sgd']:.4f} SGD**。TabPFN为此前分支保存的参考分数，本次未重跑它。", '',
        '## 为什么这轮有效', '',
        '旧历史特征把同一楼栋/户型的过去租约混在一起。本轮另查目标月份之前12/24/36个月、精确同月和前后一个月的同期记录，生成样本数、均价、相对当时同户型市场的租金比例、比例波动和低租金比例。',
        '它可能保留了周期性重复租赁的水平，但没有租客或单元ID，不能称为已识别续租；也不能据此作因果解释。',
        '验证期查询只能匹配拟合起点前已观测月份；长跨度下不存在的记录保持缺失/计数0，不回填未来租金。', '',
        '| 对照 | 原47列 LightGBM | 加同期统计62列 LightGBM | 差值 |', '| --- | ---: | ---: | ---: |']
    for row in ablation:
        lines.append(f"| {row['stage']} | {row['base47_rmse']:.4f} | {row['plus_anniversary62_rmse']:.4f} | {row['delta']:+.4f} |")
    lines += ['', '两组使用相同训练/验证行、基础树设置和较早时间早停规则；轮数各自由早期窗口选择。差异是该特征组在此管线中的增量，不能解释为单列或真实续租的因果效应。', '',
        '## 本轮主融合', '', '| 成员 | 权重 |', '| --- | ---: |']
    lines += [f"| {name} | {item['weight']:.3%} |" for name,item in selection['members'].items()]
    lines += ['', '权重非负、和为1、单成员最高0.6，仅以2023-10至2024-03的17,977条较早预测优化。年度/六月模型重新拟合至各自起点，保持较早选定的轮数与权重。', '',
        '## 其他路线与负结果', '',
        '- 第二层偏差校正：31组配置，年度最低463.6825；对六月窗口未显示改善。',
        '- 市场水平与房屋相对价格分解：24组配置。单独并未胜过同期模型，但其中一个轻量模型在主融合中提供互补。',
        '- TabICLv2：14种上下文/目标/特征配置；早期抽样为每月200行。最初完整确认的相对历史残差版本在年度为488.1767，与旧融合按早期选权融合约463.7127。它不是本轮赢家。',
        f"- 加同期特征后的TabICL在相同1,200条抽样记录上为{foundation_sample['tabicl_cohort']['rmse_sgd']:.4f}，同期CatBoost为{foundation_sample['catboost_cohort']['rmse_sgd']:.4f}、主融合为{foundation_sample['primary_blend']['rmse_sgd']:.4f}。抽样分数不与完整窗口混排。",
        '- 精简36月字段、地址×户型×月份类别与归一化目标再试8组。较早选择的精简比例模型进一步混入后，年度453.7023、六月444.7842，均不如主融合，故不替换主结果。', '',
        '## 长跨度复核与边界', '',
        '16个月复核另用2023-06至2023-11选轮数和权重，再从2023-12固定起点预测至2025-03；不沿用包含2024年标签的选轮数或权重。模型/特征结构仍来自当前开发。',
        '超过12个月时，12月前的同期记录逐渐不再属于已观测历史，24/36月统计继续可用。因此长跨度表现必须单独检查。',
        '该窗口与一年期/六月窗口重叠，不计算跨窗口混合排名。所有时期此前都被项目实验查看过，仍为开发证据，不是独立最终测试或Kaggle成绩。', '',
        '## 验证与复现', '',
        f"已独立重算 {len(checks)} 份预测结果，检查源标签、月份、行ID、有限值和源文件哈希；主融合14个模型重载后预测一致。时间边界测试见 tests/test_second_wave.py。", '',
        '按三个月连续区块重采样得到的差值区间只描述固定预测在当前月份上的敏感性，不覆盖调参、随机种子或未来市场变化。见 monthly_comparison_*.csv 和 summary.json。', '',
        '```powershell',
        'python -m unittest tests.test_second_wave -v',
        'python -m analysis.experiments.second_wave_anniversary screen',
        'python -m analysis.experiments.second_wave_normalized screen',
        'python -m analysis.experiments.second_wave_ensemble select',
        'python -m analysis.experiments.second_wave_ensemble outer',
        'python -m analysis.experiments.second_wave_ensemble june',
        'python -m analysis.experiments.second_wave_stress',
        'python -m analysis.experiments.second_wave_report', '```', '',
        '模型/逐行预测：`output/models/round_02_anniversary_history/`。本轮主方法只需已有Conda建模环境；TabICL额外包安装在项目 `.cache/foundation-deps/`，权重在 `.cache/foundation-models/`。',
        'TabICL来源：[作者实现](https://github.com/soda-inria/tabicl)、[作者权重](https://huggingface.co/jingang/TabICL)。所有模型推理在本地进行。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts':counts,'comparisons':comparisons,
        'stress':stress['scores'],'prediction_files_verified':len(checks),'reloads':len(reloads)},indent=2),flush=True)


if __name__=='__main__':
    main()
