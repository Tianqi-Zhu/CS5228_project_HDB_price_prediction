"""Verify blend-only experiments, refit their weights, export predictions."""
from __future__ import annotations
import json
import shutil
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.ninth_wave_blending import OUT,ART,SPECS,FOLLOWUP,data,fit,predict,fitting_pool
from analysis.experiments.best_ensemble_submission import prepare
from analysis.experiments.canonical_final import validate_submission
from analysis.experiments.second_wave_report import monthly_comparison
from analysis.pipeline.data import sha256


def weight_table(model):
    result=pd.DataFrame({'component':model['names'],'global':model['global_weights']})
    for level,group in model['group_weights'].items():result[level]=group['weights']
    return result


def run():
    earlier=data('screen');annual=data('outer');recent=data('june')
    sources={'screen':earlier,'outer':annual,'june':recent}
    checks=[]
    for stage,item in sources.items():
        if stage=='screen':
            from analysis.experiments.ninth_wave_blending import subset
            item=subset(item,item['month']>='2024-02')
        for spec in SPECS+FOLLOWUP:
            directory=ART/stage/spec['name']
            model=json.loads((directory/'model.json').read_text(encoding='utf-8'))
            cutoff={'screen':'2024-02','outer':'2024-04','june':'2024-06'}[stage]
            assert model['fit_last_month']<cutoff
            frame=pd.read_csv(directory/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,item['row'])
            np.testing.assert_array_equal(frame.actual,item['actual'])
            restored=predict(model,item)
            np.testing.assert_allclose(restored,frame.prediction,atol=1e-8,rtol=0)
            record=json.loads((OUT/stage/spec['name']/'results.json').read_text(encoding='utf-8'))
            actual=score(item['actual'],frame.prediction.to_numpy())
            assert abs(actual['rmse_sgd']-record['rmse_sgd'])<1e-8
            for weights in [model['global_weights']]+[g['weights'] for g in model['group_weights'].values()]:
                np.testing.assert_allclose(np.sum(weights),1.,atol=1e-8)
                assert max(weights)<=.60000001
                assert min(weights)>=(-.05000001 if spec.get('signed') else -1e-8)
            checks.append({'stage':stage,'name':spec['name'],
                'max_reload_difference':float(np.max(np.abs(restored-frame.prediction.to_numpy())))})
    comparison=pd.read_csv(OUT/'comparison.csv').set_index('name')
    winner=str(comparison.june.idxmin())
    winner_spec=next(s for s in SPECS+FOLLOWUP if s['name']==winner)
    early_selected=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']['name']
    write_json(OUT/'development_candidate.json',{'name':winner,'spec':winner_spec,
        'recent':float(comparison.loc[winner,'june']),'annual':float(comparison.loc[winner,'outer']),
        'earlier_selected':early_selected,'earlier_selected_recent':float(comparison.loc[early_selected,'june']),
        'selection_basis':'Lowest later development RMSE among 31 recipes; post-hoc exploration, not a prelocked independent test.'})
    intervals=[]
    for stage,item in [('outer',annual),('june',recent)]:
        table=pd.read_csv(ART/stage/winner/'predictions.csv').sort_values('source_row').reset_index(drop=True)
        previous=pd.DataFrame({'source_row':item['row'],'month':item['month'],'actual':item['actual'],
            'prediction':item['matrix']@item['prior']})
        monthly,summary=monthly_comparison(table,previous)
        monthly.to_csv(OUT/f'{stage}_monthly.csv',index=False)
        intervals.append({'stage':stage,**summary})
    write_json(OUT/'monthly_comparison.json',{'results':intervals,
        'note':'Descriptive development-window resampling after configuration search; not independent confirmatory evidence.'})
    weight_table(json.loads((ART/'june'/winner/'model.json').read_text(encoding='utf-8'))).to_csv(OUT/'weights_june.csv',index=False)
    write_json(OUT/'verification.json',{'replays':checks,'count':len(checks),
        'tests':{'command':'python -m unittest tests.test_ninth_wave','passed':5},
        'query_labels_used_in_fit':False,'all_scores_recomputed':True})
    # Standard final refit: update only fusion weights using historical
    # out-of-origin forecasts whose labels are now part of the 150k train set.
    pool=fitting_pool(earlier,annual,'2025-04',winner_spec['update'])
    final_model=fit(pool,winner_spec)
    assert final_model['fit_last_month']<'2025-04'
    destination=ROOT/'output/submissions/components/round_09_grouped_blend';destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'blend_model.json',final_model)
    weight_table(final_model).to_csv(OUT/'weights_full.csv',index=False)
    bundle,hashes=prepare()
    old=ROOT/'output/submissions/components/history_macro_ensemble'
    old_summary=json.loads((old/'summary.json').read_text(encoding='utf-8'))
    seed_summary=json.loads((ROOT/'output/submissions/components/round_08_ft_seed_average/summary.json').read_text(encoding='utf-8'))
    assert hashes==old_summary['source_sha256']==seed_summary['source_sha256']
    matrix=[];base_manifest=[]
    for name in final_model['names']:
        path=(ROOT/'output/submissions/components/round_08_ft_seed_average/ft_predictions.npy' if name=='ft_no36_seedmean'
            else old/'models'/name/'predictions.npy')
        values=np.load(path);assert values.shape==(50000,) and np.isfinite(values).all()
        matrix.append(values);base_manifest.append({'name':name,'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(path)})
    context=pd.DataFrame({'support24':(bundle['ann_val'].ANN_BLOCK_24_R1_N.to_numpy()>0).astype(str),
        'support36':(bundle['ann_val'].ANN_BLOCK_36_R1_N.to_numpy()>0).astype(str),
        'type':bundle['x_val'].FLAT_TYPE_CLEAN.astype(str).to_numpy()})
    context['support_pair']=context.support24+'|'+context.support36
    assert winner_spec['gate']!='disagreement','Full disagreement export needs the four individual seed forecasts.'
    item={'matrix':np.column_stack(matrix),'context':context,'names':final_model['names']}
    predicted=predict(final_model,item)
    restored=json.loads((destination/'blend_model.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(predict(restored,item),predicted,atol=1e-9,rtol=0)
    submission=validate_submission(pd.read_csv(ROOT/'data/example-submission.csv'),predicted)
    path=destination/'kaggle_blend_support_centered.csv';submission.to_csv(path,index=False)
    reread=pd.read_csv(path)
    np.testing.assert_array_equal(reread.Id,np.arange(50000))
    np.testing.assert_allclose(reread.Predicted,predicted,atol=1e-8,rtol=0)
    ready=ROOT/'output/submissions/candidates'/path.name;shutil.copy2(path,ready)
    assert sha256(path)==sha256(ready)
    for source,digest in hashes.items():assert sha256(ROOT/source)==digest
    group_counts=context[winner_spec['gate']].value_counts().to_dict() if winner_spec['gate']!='none' else {'all':50000}
    summary={'recipe':winner,'local_recent_rmse':float(comparison.loc[winner,'june']),
        'local_annual_rmse':float(comparison.loc[winner,'outer']),
        'selection':'post-hoc development minimum, including explicitly recorded follow-up search',
        'meta_fit_rows':final_model['fit_rows'],'meta_fit_last_month':final_model['fit_last_month'],
        'test_rows':50000,'base_training_rows':150000,'new_base_training':False,
        'component_manifest':base_manifest,'test_groups':group_counts,'source_sha256':hashes,
        'path':str(path.relative_to(ROOT)).replace('\\','/'),'ready_path':str(ready.relative_to(ROOT)).replace('\\','/'),
        'sha256':sha256(path),'actual_kaggle_score':None,'uploaded':False,
        'note':'Only blend weights refit on historical OOT forecasts; cached base models already fit on all 150k labels. Local metrics describe chronological development refits, not test data.'}
    write_json(destination/'summary.json',summary)
    write_json(ready.parent/'ninth_wave_submission_manifest.json',summary)
    make_report(comparison,winner,early_selected,intervals,summary)
    print('BLEND EXPORT',json.dumps(summary,ensure_ascii=False,indent=2))


def make_report(comparison,winner,early_selected,intervals,export):
    lines=['# 第九轮：只改变融合方式','',
        f'本轮31种融合配置的最低近期开发RMSE为 **{comparison.loc[winner,"june"]:.4f}**，年度为 **{comparison.loc[winner,"outer"]:.4f}**。前一轮为442.7032/451.4310，原历史主方案为442.9025/451.1810。单模型和输入特征均保持不变。','',
        '## 方案与信息边界','',
        '组件为原主融合的10个叶子预测＋新FT四种子平均，共11路。比较非负和为1的加权平均、带正则的权重收缩、按24/36月历史覆盖/户型/种子分歧分组、有限负权重、冻结权重与使用最新已观测预测更新权重。非负方案每路最高0.6。',
        '有效变化是以月份为单位，将拟合样本的各模型误差减去当月误差均值，再学习权重；目的是降低历史时期整体租金水平偏差对权重的干扰。分组权重向全局权重收缩。预测时直接组合模型输出，不读取待预测月份租金，不对验证标签反推校准。',
        f'最低候选 `{winner}` 按24个月、36个月同期历史是否存在，分为最多4组；全局正则0.003，组内权重向全局权重收缩的正则0.003。少于500个拟合样本的组回退到全局权重，未知组同样回退。','',
        '## 选择口径','',
        f'先用2023-10至2024-01拟合、2024-02至03选配置，早期选中 `{early_selected}`，后期近期RMSE为 {comparison.loc[early_selected,"june"]:.4f}。本文最低候选是查看后期结果后的探索选择，不能称为早期锁定方案或独立测试最佳。初批22种，随后另有9种分组收缩对照，均保留完整结果。',
        '年度权重仅用2024-04之前的样本；六月权重最多加入4—5月的年度模型预测，且预测由4月之前拟合的单模型产生。两个原点的总损失权重各占一半。冻结权重对照仅使用早期6个月。年度和近期窗口部分重叠，也都是多轮使用过的开发集。','',
        '## 主要结果','',
        '| 方案 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    names=['previous_history','previous_ft_seedmean','global_mse_0.003','global_month_centered_0.003',
        'support24_month_centered','followup_support24_centered_0.003',winner,early_selected]
    for name in dict.fromkeys(names):
        row=comparison.loc[name];lines.append(f'| {name} | {row["outer"]:.4f} | {row["june"]:.4f} |')
    lines+=['','完整31种对照见[comparison.csv](comparison.csv)。正负结果均保留，原提交文件保持不变。','',
        '## 稳定性与核查','']
    for item in intervals:
        lo,hi=item['month_block_interval_95']
        lines.append(f'- {item["stage"]}：相对前一轮差值 {item["delta_rmse_sgd"]:.4f}；{item["months_improved"]}/{item["months"]}个月改善；描述性月块区间 [{lo:.4f}, {hi:.4f}]。')
    lines+=['','重采样是在配置搜索后的开发数据上进行，只描述波动，不是独立确认的置信区间。共31×3=93次全局/分组权重拟合流程，包含重复的冻结/更新年度对照，不能计作93个独立模型。93份JSON权重均重新加载并复现预测，RMSE及约束重新核对；5项信息截止、权重约束、月度中心化、组回退和预测不读取标签的测试通过。','',
        '## 全量提交文件','',
        '复用此前已在全部150,000条训练标签上拟合的单模型及其50,000条预测，只重训融合权重。最终融合使用2023-10至2024-03及2024-04至2025-03的历史样本外预测拟合，原点总损失等权，标签截止2025-03；不读取测试标签。最终权重和六月验证权重因拟合截止不同而不同，均单独保存。',
        f'- [新融合CSV](../../../{export["ready_path"]})。',
        '- [提交清单](../../../output/submissions/components/round_09_grouped_blend/summary.json)。',
        '- [六月验证权重](weights_june.csv)、[全量融合权重](weights_full.csv)、[模型复现](verification.json)。','',
        '新文件尚未提交，Kaggle分数未知。用户反馈原历史/宏观两份文件均约415，正常方案最佳约411；不能把本地改善直接换算成榜单改善。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':run()
