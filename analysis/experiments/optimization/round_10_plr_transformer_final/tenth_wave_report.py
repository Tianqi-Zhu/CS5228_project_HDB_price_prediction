"""Replay architecture experiments and compare them under a fixed blend rule."""
from __future__ import annotations
import gc
import json
import joblib
import numpy as np
import pandas as pd
import torch
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.tenth_wave_structures import OUT,ART,SPECS,dependencies,predict_saved,fit as fit_network
from analysis.experiments.tenth_wave_ensemble import selected,extended,components,variants,recipe
from analysis.experiments.ninth_wave_blending import fitting_pool,fit as fit_blend,predict as blend_predict,subset
from analysis.experiments.second_wave_report import monthly_comparison


def run():
    dependencies();torch.set_num_threads(4)
    checks=[];rows=[];network_representatives=selected()
    has_tree=(OUT/'tree_selection.json').exists()
    representatives=selected(has_tree)
    for stage in ('screen','outer','june'):
        bundle=make_data(stage)
        names=[spec['name'] for spec in SPECS] if stage=='screen' else list(network_representatives.values())
        for name in names:
            record=json.loads((OUT/stage/name/'results.json').read_text(encoding='utf-8'))
            expected=pd.read_csv(ART/stage/name/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(expected.source_row,bundle['row_val'])
            np.testing.assert_array_equal(expected.actual,bundle['y_val'])
            payload=joblib.load(ART/stage/name/'model.joblib')
            replay=predict_saved(payload,bundle)
            np.testing.assert_allclose(replay,expected.prediction,atol=1e-5,rtol=0)
            checks.append({'kind':'network','stage':stage,'name':name,
                'max_reload_difference':float(np.max(np.abs(replay-expected.prediction.to_numpy())))})
            rows.append({key:record[key] for key in ('stage','name','epochs','parameters','features','rows','rmse_sgd','mae_sgd','bias_sgd','seconds')})
            del payload;gc.collect();torch.cuda.empty_cache()
        if has_tree:
            from analysis.experiments.tenth_wave_trees import SPECS as TREE_SPECS,inputs as tree_inputs
            _,query,_=tree_inputs(bundle)
            tree_names=[spec['name'] for spec in TREE_SPECS] if stage=='screen' else [representatives['tree']]
            for name in tree_names:
                record=json.loads((OUT/'trees'/stage/name/'results.json').read_text(encoding='utf-8'))
                expected=pd.read_csv(ART/'trees'/stage/name/'predictions.csv').sort_values('source_row')
                np.testing.assert_array_equal(expected.source_row,bundle['row_val'])
                np.testing.assert_array_equal(expected.actual,bundle['y_val'])
                payload=joblib.load(ART/'trees'/stage/name/'model.joblib')
                replay=payload['model'].predict(query)
                np.testing.assert_allclose(replay,expected.prediction,atol=1e-6,rtol=0)
                checks.append({'kind':'tree','stage':stage,'name':name,
                    'max_reload_difference':float(np.max(np.abs(replay-expected.prediction.to_numpy())))})
                rows.append({key:record[key] for key in ('stage','name','features','rows','rmse_sgd','mae_sgd','bias_sgd','seconds')}
                    |{'epochs':833,'parameters':np.nan})
                del payload
        item=extended(stage,representatives)
        if stage=='screen':item=subset(item,item['month']>='2024-02')
        for name,chosen in variants(has_tree).items():
            query=components(item,chosen)
            saved=json.loads((ART/'ensemble'/stage/name/'model.json').read_text(encoding='utf-8'))
            expected=pd.read_csv(ART/'ensemble'/stage/name/'predictions.csv').sort_values('source_row')
            replay=blend_predict(saved,query)
            np.testing.assert_array_equal(expected.source_row,query['row'])
            np.testing.assert_allclose(replay,expected.prediction,atol=1e-8,rtol=0)
            cutoff={'screen':'2024-02','outer':'2024-04','june':'2024-06'}[stage]
            assert saved['fit_last_month']<cutoff
            if name=='baseline' and stage!='screen':
                previous=pd.read_csv(ROOT/'output/models/round_09_grouped_blending'/stage/recipe()['name']/'predictions.csv').sort_values('source_row')
                np.testing.assert_array_equal(previous.source_row,expected.source_row)
                np.testing.assert_allclose(previous.prediction,expected.prediction,atol=1e-8,rtol=0)
            checks.append({'kind':'blend','stage':stage,'name':name,
                'max_reload_difference':float(np.max(np.abs(replay-expected.prediction.to_numpy())))})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'model_metrics.csv',index=False)
    single=metrics[metrics.stage!='screen'].pivot(index='name',columns='stage',values='rmse_sgd').sort_values('june')
    if has_tree:
        single.loc['cat_symmetric_saved']=[json.loads((ROOT/'output/experiment_runs/round_02_anniversary_history/anniversary'/stage/'window_cat6/results.json').read_text(encoding='utf-8'))['rmse_sgd'] for stage in single.columns]
        single=single.sort_values('june')
    single.to_csv(OUT/'single_comparison.csv')
    table=pd.read_csv(OUT/'ensemble_metrics.csv')
    blended=table[table.stage!='screen'].pivot(index='name',columns='stage',values='rmse_sgd').sort_values('june')
    blended.to_csv(OUT/'blend_comparison.csv')
    best=str(blended.june.idxmin());improves=float(blended.loc['baseline','june']-blended.loc[best,'june'])>1e-6
    winning_weights=json.loads((ART/'ensemble/june'/best/'model.json').read_text(encoding='utf-8'))
    new_indices=[i for i,name in enumerate(winning_weights['names']) if name.startswith(('tenth/','tree/'))]
    route_weights={'global':float(sum(winning_weights['global_weights'][i] for i in new_indices))}
    route_weights.update({level:float(sum(group['weights'][i] for i in new_indices))
        for level,group in winning_weights['group_weights'].items()})
    improves=improves and max(route_weights.values())>1e-5
    write_json(OUT/'verification.json',{'replays':checks,'base_count':len(rows),
        'network_count':sum(c['kind']=='network' for c in checks),'tree_count':sum(c['kind']=='tree' for c in checks),
        'blend_count':len(checks)-len(rows),'baseline_reproduced':True,
        'tests':{'command':'python -m unittest tests.test_tenth_wave tests.test_ninth_wave','passed':9}})
    summary={'best_blend':best,'recent_rmse':float(blended.loc[best,'june']),
        'annual_rmse':float(blended.loc[best,'outer']),'beats_previous_recent':improves,
        'previous_recent':float(blended.loc['baseline','june']),
        'previous_annual':float(blended.loc['baseline','outer']),
        'selection':'Later development comparison, not an independent test.','new_model_total_weights':route_weights,
        'early_selected':json.loads((OUT/'ensemble_selection.json').read_text(encoding='utf-8'))['selected']['name']}
    monthly=[]
    if improves:
        for stage in ('outer','june'):
            before=pd.read_csv(ART/'ensemble'/stage/'baseline/predictions.csv').sort_values('source_row').reset_index(drop=True)
            after=pd.read_csv(ART/'ensemble'/stage/best/'predictions.csv').sort_values('source_row').reset_index(drop=True)
            month,description=monthly_comparison(after,before)
            month.to_csv(OUT/f'{stage}_monthly.csv',index=False);monthly.append({'stage':stage,**description})
        summary['export']=export(best,summary)
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'monthly_comparison.json',{'results':monthly,'note':'Descriptive reused development evidence after model/configuration selection.'})
    write_report(single,blended,summary,rows)
    update_indexes(single,blended,summary)
    print('ARCHITECTURE SINGLE MODELS\n'+single.to_string())
    print('ARCHITECTURE BLENDS\n'+blended.to_string())
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def export(variant,summary):
    import shutil
    from analysis.experiments.best_ensemble_submission import prepare
    from analysis.experiments.canonical_final import validate_submission
    from analysis.pipeline.data import sha256
    from analysis.experiments.ninth_wave_report import weight_table
    chosen=variants((OUT/'tree_selection.json').exists())[variant]
    earlier=extended('screen',chosen);annual=extended('outer',chosen)
    model=fit_blend(fitting_pool(earlier,annual,'2025-04'),recipe())
    destination=ROOT/'output/submissions/components/round_10_final_structure';destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'blend_model.json',model);weight_table(model).to_csv(OUT/'weights_full.csv',index=False)
    bundle,hashes=prepare();matrices=[];manifest=[]
    source=json.loads((ROOT/'output/submissions/components/round_09_grouped_blend/summary.json').read_text(encoding='utf-8'))
    assert hashes==source['source_sha256']
    old_paths={item['name']:ROOT/item['path'] for item in source['component_manifest']}
    for name in model['names']:
        if name in old_paths:
            path=old_paths[name];prediction=np.load(path)
        else:
            is_tree=name.startswith('tree/')
            network_name=name.split('/',1)[1]
            if not is_tree:
                spec=next(spec for spec in SPECS if spec['name']==network_name)
                epochs=json.loads((OUT/'screen'/network_name/'results.json').read_text(encoding='utf-8'))['epochs']
            else:
                from analysis.experiments.tenth_wave_trees import inputs as tree_inputs
                from catboost import CatBoostRegressor
                tree_x,tree_q,categories=tree_inputs(bundle)
                parameters=joblib.load(ART/'trees/june'/network_name/'model.joblib')['parameters']
            directory=destination/'models'/network_name;directory.mkdir(parents=True,exist_ok=True)
            checkpoint=directory/'model.joblib';path=directory/'predictions.npy'
            if checkpoint.exists():
                payload=joblib.load(checkpoint);assert payload['source_sha256']==hashes
                prediction=payload['model'].predict(tree_q) if is_tree else predict_saved(payload,bundle)
            else:
                if is_tree:
                    network=CatBoostRegressor(**parameters);network.fit(tree_x,bundle['y_fit'],cat_features=categories)
                    prediction=network.predict(tree_q);count=833;payload={'model':network,'parameters':parameters}
                else:payload,prediction,count=fit_network(spec,bundle,epochs)
                payload.update(source_sha256=hashes,epochs=count)
                joblib.dump(payload,checkpoint,compress=1)
                restored=joblib.load(checkpoint)
                replay=restored['model'].predict(tree_q) if is_tree else predict_saved(restored,bundle)
                np.testing.assert_allclose(replay,prediction,atol=1e-5,rtol=0)
                del restored
            np.save(path,prediction);del payload;gc.collect();torch.cuda.empty_cache()
            print('FULL NEW STRUCTURE',network_name,'complete',flush=True)
        assert len(prediction)==50000 and np.isfinite(prediction).all()
        matrices.append(prediction);manifest.append({'name':name,'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(path)})
    ann=bundle['ann_val'];context=pd.DataFrame({'support24':(ann.ANN_BLOCK_24_R1_N.to_numpy()>0).astype(str),
        'support36':(ann.ANN_BLOCK_36_R1_N.to_numpy()>0).astype(str)})
    context['support_pair']=context.support24+'|'+context.support36
    item={'matrix':np.column_stack(matrices),'context':context,'names':model['names']}
    prediction=blend_predict(model,item)
    restored=json.loads((destination/'blend_model.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(blend_predict(restored,item),prediction,atol=1e-8,rtol=0)
    output=validate_submission(pd.read_csv(ROOT/'data/example-submission.csv'),prediction)
    path=destination/'kaggle_structure_blend.csv';output.to_csv(path,index=False)
    reread=pd.read_csv(path);np.testing.assert_array_equal(reread.Id,np.arange(50000))
    np.testing.assert_allclose(reread.Predicted,prediction,atol=1e-8,rtol=0)
    ready=ROOT/'output/submissions/candidates'/path.name;shutil.copy2(path,ready)
    assert sha256(path)==sha256(ready)
    for source_path,digest in hashes.items():assert sha256(ROOT/source_path)==digest
    result={'variant':variant,'path':str(path.relative_to(ROOT)).replace('\\','/'),
        'ready_path':str(ready.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(path),
        'local_recent_rmse':summary['recent_rmse'],'local_annual_rmse':summary['annual_rmse'],
        'base_fit_rows':150000,'test_rows':50000,'meta_fit_rows':model['fit_rows'],
        'meta_fit_last_month':model['fit_last_month'],'component_manifest':manifest,'source_sha256':hashes,
        'actual_kaggle_score':None,'uploaded':False}
    write_json(destination/'summary.json',result);write_json(ready.parent/'tenth_wave_submission_manifest.json',result)
    return result


def write_report(single,blended,summary,rows):
    lines=['# 第十轮：模型结构实际训练对照','',
        '本轮探索数值token、显式交叉和条件专家三条神经结构路线；含原始线性token控制组，共9个较早窗口配置。另并行探索2个非对称CatBoost树结构。神经网络从较早窗口选每条路线代表和轮数，后续固定轮数训练；树模型固定原对称树选出的833轮。神经网络统一使用去掉五列36月同期统计的62个候选输入，并学习相对24月历史基准的残差。树模型沿用原对称树的62列window历史输入、直接预测租金；两组62列的内容不同。','',
        '## 结构与实现来源','',
        '- FT-Transformer：使用[作者RTDL骨干](https://arxiv.org/abs/2106.11959)，比较线性、分段线性PLE和周期PLR数值token；嵌入来自[作者实现](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/README.md)，方法对应[数值嵌入论文](https://arxiv.org/abs/2203.05556)。96宽2层与128宽3层为两个容量配置。新组合由项目适配，不宣称逐位复现论文成绩。',
        '- Deep & Cross：按[DCN-V2](https://arxiv.org/abs/2008.13535)的全矩阵交叉公式实现显式交互分支，并联两层深度分支。比较线性嵌入＋2个交叉层、PLR嵌入＋3个交叉层；为项目实现，未宣称作者代码原样复现。',
        '- 条件专家：项目实现，共享PLR/类别嵌入和128维主干，4或8个回归专家，软门控仅使用输入特征；MSE加0.001负载均衡项，没有根据查询真实租金指定专家。','',
        '- 非对称树：按[CatBoost官方参数](https://catboost.ai/docs/en/references/training-parameters/common#grow_policy)比较Depthwise深度6和Lossguide深度8/最多64叶；最小叶样本20。与原CPU对称树保持输入、目标、833轮、学习率0.045、L2=15和seed42一致，其余结构容量限制单独记录，不能将效果归因于一个参数。','',
        '## 完整窗口单模型结果','',
        '| 代表模型 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for name,r in single.iterrows():lines.append(f'| {name} | {r["outer"]:.4f} | {r["june"]:.4f} |')
    lines+=['','单模型与融合的指标分开比较。它们是有限预算、单随机种子的结构探索，不能据此判定某架构普遍优劣。','',
        '## 固定融合规则下的增量','',
        '保持第九轮24/36月覆盖分组、月度中心化误差、全局及组内正则0.003不变，仅改变是否加入新模型预测。每个原点重新拟合融合权重，但不再调融合超参数。baseline完整复现441.9591/450.8650，避免将融合规则改变算成结构收益。','',
        '| 新增路线 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for name,r in blended.iterrows():lines.append(f'| {name} | {r["outer"]:.4f} | {r["june"]:.4f} |')
    if summary['beats_previous_recent']:
        lines+=['',f'本轮最低候选 `{summary["best_blend"]}`：近期 **{summary["recent_rmse"]:.4f}**，年度 **{summary["annual_rmse"]:.4f}**。这是后期开发集比较后的选择，不能当作预先锁定的独立测试最佳。',
            f'已全量重训所需新模型并导出[50,000行新CSV](../../../{summary["export"]["ready_path"]})，新Kaggle分数未知。']
    else:lines+=['','本轮未降低近期441.9591，因此保留第九轮方案及其提交文件；新结构和全部负结果均归档。']
    lines+=['','## 验证口径','',
        f'本轮共{len(rows)}次基础模型拟合（19次神经网络、4次树模型），另有{len(blended)}种纳入路线×3窗口的融合权重流程。基础训练只访问拟合标签；后期不用查询标签早停。融合年度标签截止2024-03，六月截止2024-05；全量融合截止2025-03。较早窗口本身已用于模型、epoch和历轮特征选择，因此所有分数仍是反复查看的开发证据，年度和近期窗口部分重叠。',
        f'早期内部选择的神经网络融合纳入路线为 `{summary["early_selected"]}`；加入树模型的额外对照另行记录。所有{len(rows)}份基础模型及{len(blended)*3}份融合权重均重新加载复现，baseline匹配此前结果，9项相关结构/信息边界测试通过。单模型表包含不同输入组，不能跨组作架构公平排名。','',
        '[单模型全部记录](model_metrics.csv)、[融合全部记录](ensemble_metrics.csv)、[选择记录](selection.json)、[复现核查](verification.json)、[总结](summary.json)。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def update_indexes(single,blended,summary):
    changed=summary['beats_previous_recent']
    headline=(f'新结构加入固定融合后的最低候选近期{summary["recent_rmse"]:.4f}、年度{summary["annual_rmse"]:.4f}；新CSV已导出，尚未提交。'
        if changed else '新增结构未降低近期441.9591，继续保留第九轮融合及其提交文件。')
    path=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md'
    text=path.read_text(encoding='utf-8')
    status='<!-- tenth-wave-status:start -->\n**第十轮结构更新（2026-11-05）：** 完成数值嵌入FT、Deep & Cross、条件专家及非对称树实际训练：19次神经网络和4次树拟合，固定第九轮融合规则对照。'+headline+' 完整表格和选择口径见[第14节](#14-模型结构的实际训练对照)。\n<!-- tenth-wave-status:end -->\n\n'
    if '<!-- tenth-wave-status:start -->' not in text:
        first,rest=text.split('\n',1);text=first+'\n\n'+status+rest.lstrip('\n')
    marker='<!-- tenth-wave-details:start -->'
    text=text.split(marker)[0].rstrip()
    lines=['',marker,'## 14. 模型结构的实际训练对照','',
        '神经网络组统一62个输入，训练线性/分段线性/周期数值token的FT-Transformer、全矩阵Deep & Cross、4/8个条件回归专家。FT骨干和数值嵌入使用已有作者包，交叉网络与专家结构为项目实现。9个较早配置先选轮数和5条代表路线，后期固定轮数，共19次网络拟合。另训练2种非对称CatBoost结构，在早期选1种评估后期，共4次树拟合；它们使用原CatBoost的62列window输入，与神经网络的62列内容不同，故不作跨组公平排名。','',
        '| 代表网络 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for name,row in single.iterrows():lines.append(f'| {name} | {row["outer"]:.4f} | {row["june"]:.4f} |')
    lines+=['',f'固定第九轮融合目标、正则和分组方式，比较不加入新模型、分别加入、全部加入，共{len(blended)}种纳入路线；baseline复现441.9591/450.8650。','',
        '| 纳入路线 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for name,row in blended.iterrows():lines.append(f'| {name} | {row["outer"]:.4f} | {row["june"]:.4f} |')
    lines+=['',headline,'',
        f'早期内部选择的神经路线为`{summary["early_selected"]}`；树模型加入为额外探索。上表后期最低属于反复查看的开发集探索结果，不能称为预先锁定的独立测试最佳。后期基础模型不使用查询标签早停，融合标签按各预测原点截止；两个窗口部分重叠。19份网络、4份树模型和{len(blended)*3}份融合权重重新加载复现，9项相关测试通过。','',
        '[完整结构报告](../round_10_plr_transformer_final/README.md)、[单模型指标](../round_10_plr_transformer_final/model_metrics.csv)、[融合对照](../round_10_plr_transformer_final/blend_comparison.csv)、[复现核查](../round_10_plr_transformer_final/verification.json)。']
    if changed:lines+=['',f'[新结构融合CSV](../../../{summary["export"]["ready_path"]})，50,000行，尚未上传，Kaggle成绩未知。']
    lines+=['<!-- tenth-wave-details:end -->','']
    path.write_text(text+'\n'+'\n'.join(lines),encoding='utf-8')
    for relative,link in [('experiment/README.md','04_optimization_history/10_plr_transformer_final.md'),
        ('output/experiment_runs/README.md','round_10_plr_transformer_final/README.md')]:
        path=ROOT/relative;body=path.read_text(encoding='utf-8')
        line=f'2026-11-05：[第十轮模型结构对照]({link})。{headline}\n\n'
        if link not in body:
            first,rest=body.split('\n',1);path.write_text(first+'\n\n'+line+rest.lstrip('\n'),encoding='utf-8')


if __name__=='__main__':run()
