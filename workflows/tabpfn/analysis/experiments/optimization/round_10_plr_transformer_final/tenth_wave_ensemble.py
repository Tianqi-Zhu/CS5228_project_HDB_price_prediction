"""Test new network structures with the ninth-wave blending recipe fixed."""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.ninth_wave_blending import data as previous_data,subset,fitting_pool,fit,predict
from analysis.experiments.tenth_wave_structures import OUT,ART


def selected(include_tree=False):
    result=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['representatives']
    if include_tree:
        result=result|{'tree':json.loads((OUT/'tree_selection.json').read_text(encoding='utf-8'))['name']}
    return result


def extended(stage,representatives):
    item=previous_data(stage)
    names=list(item['names']);matrices=[item['matrix']]
    for family,name in sorted(representatives.items()):
        path=ART/'trees'/stage/name/'predictions.csv' if family=='tree' else ART/stage/name/'predictions.csv'
        table=pd.read_csv(path).sort_values('source_row')
        np.testing.assert_array_equal(table.source_row,item['row'])
        np.testing.assert_array_equal(table.actual,item['actual'])
        names.append(('tree/' if family=='tree' else 'tenth/')+name);matrices.append(table.prediction.to_numpy(float)[:,None])
    item=item|{'names':names,'matrix':np.column_stack(matrices),
        'prior':np.r_[item['prior'],np.zeros(len(representatives))]}
    return item


def variants(include_tree=False):
    representatives=selected()
    result={'baseline':{},'all_new':representatives}|{f'only_{family}':{family:name}
        for family,name in representatives.items()}
    if include_tree:
        all_representatives=selected(True)
        result|={'only_tree':{'tree':all_representatives['tree']},'all_new_plus_tree':all_representatives}
    return result


def recipe():
    specification=json.loads((ROOT/'output/experiment_runs/round_09_grouped_blending/development_candidate.json').read_text(encoding='utf-8'))['spec']
    return specification.copy()


def components(item,representatives):
    wanted=[('tree/' if family=='tree' else 'tenth/')+name for family,name in sorted(representatives.items())]
    indices=[i for i,name in enumerate(item['names']) if not name.startswith(('tenth/','tree/')) or name in wanted]
    return item|{'names':[item['names'][i] for i in indices],
        'matrix':item['matrix'][:,indices],'prior':item['prior'][indices]}


def run(stage):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    if stage=='screen':
        write_json(OUT/'ensemble_plan.json',{'variants':variants(),'recipe':recipe(),
            'notes':'Hold ninth-wave 24/36-support month-centered blend objective, prior, and penalties fixed. Vary only inclusion of new network predictions. Selection split and all periods are repeatedly viewed development evidence.'})
    records=[];all_representatives=selected()
    earlier_all=extended('screen',all_representatives)
    annual_all=None if stage=='screen' else extended('outer',all_representatives)
    query_all=earlier_all if stage=='screen' else annual_all if stage=='outer' else extended('june',all_representatives)
    for name,representatives in variants().items():
        earlier=components(earlier_all,representatives)
        if stage=='screen':
            train=earlier['month']<'2024-02';pool=subset(earlier,train);query=subset(earlier,~train)
        else:
            annual=components(annual_all,representatives)
            query=components(query_all,representatives)
            origin='2024-04' if stage=='outer' else '2024-06'
            pool=fitting_pool(earlier,annual,origin)
        model=fit(pool,recipe());prediction=predict(model,query)
        directory=ART/'ensemble'/stage/name;directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'model.json',model)
        restored=json.loads((directory/'model.json').read_text(encoding='utf-8'))
        np.testing.assert_allclose(predict(restored,query),prediction,atol=1e-8,rtol=0)
        pd.DataFrame({'source_row':query['row'],'month':query['month'],'actual':query['actual'],
            'prediction':prediction}).to_csv(directory/'predictions.csv',index=False)
        record={'stage':stage,'name':name,'representatives':representatives,
            'fit_last_month':model['fit_last_month'],'fit_rows':model['fit_rows'],
            'rows':len(prediction),**score(query['actual'],prediction)}
        write_json(OUT/'ensemble'/stage/name/'results.json',record);records.append(record)
        print('STRUCTURE BLEND',stage,name,record['rmse_sgd'],flush=True)
    if stage=='screen':
        path=OUT/'ensemble_selection.json'
        if not path.exists():write_json(path,{'selected':min(records,key=lambda r:r['rmse_sgd']),
            'decision_before_later_evaluation':True})
    records=[]
    for path in (OUT/'ensemble').glob('*/*/results.json'):
        records.append(json.loads(path.read_text(encoding='utf-8')))
    pd.DataFrame([{k:v for k,v in r.items() if k!='representatives'} for r in records]).to_csv(OUT/'ensemble_metrics.csv',index=False)


def run_extra():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    all_representatives=selected(True)
    write_json(OUT/'tree_blend_plan.json',{'variants':{k:v for k,v in variants(True).items() if k in ('only_tree','all_new_plus_tree')},
        'recipe':recipe(),'note':'Additional tree inclusion after neural screen; reused development periods, not independent confirmation.'})
    all_data={stage:extended(stage,all_representatives) for stage in ('screen','outer','june')}
    records=pd.read_csv(OUT/'ensemble_metrics.csv').to_dict('records')
    for stage in ('screen','outer','june'):
        for name in ('only_tree','all_new_plus_tree'):
            representatives=variants(True)[name]
            earlier=components(all_data['screen'],representatives)
            if stage=='screen':
                train=earlier['month']<'2024-02';pool=subset(earlier,train);query=subset(earlier,~train)
            else:
                annual=components(all_data['outer'],representatives)
                query=components(all_data[stage],representatives)
                pool=fitting_pool(earlier,annual,'2024-04' if stage=='outer' else '2024-06')
            model=fit(pool,recipe());prediction=predict(model,query)
            directory=ART/'ensemble'/stage/name;directory.mkdir(parents=True,exist_ok=True)
            write_json(directory/'model.json',model)
            pd.DataFrame({'source_row':query['row'],'month':query['month'],'actual':query['actual'],
                'prediction':prediction}).to_csv(directory/'predictions.csv',index=False)
            record={'stage':stage,'name':name,'fit_last_month':model['fit_last_month'],
                'fit_rows':model['fit_rows'],'rows':len(prediction),**score(query['actual'],prediction)}
            write_json(OUT/'ensemble'/stage/name/'results.json',record);records.append(record)
            print('TREE BLEND',stage,name,record['rmse_sgd'],flush=True)
    pd.DataFrame(records).drop_duplicates(['stage','name'],keep='last').to_csv(OUT/'ensemble_metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['screen','outer','june'])
    run(parser.parse_args().stage)
