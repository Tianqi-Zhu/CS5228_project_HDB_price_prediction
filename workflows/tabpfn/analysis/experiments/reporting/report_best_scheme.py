"""Make the lowest forecasting RMSE an explicit report recommendation."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,score,write_json

OUT=ROOT/'output/experiment_runs/00_report_and_final'
NAME='最优历史特征融合（Best ensemble）'


def build():
    third=ROOT/'output/experiment_runs/round_03_tree_transformer_ensemble/ablation_models.json'
    records=json.loads(third.read_text(encoding='utf-8'))['records']
    chosen={stage:next(r for r in records if r['stage']==stage and r['kind']=='models')
        for stage in ('outer','june')}
    second=json.loads((ROOT/'output/experiment_runs/round_02_anniversary_history/ensemble_selection.json').read_text(encoding='utf-8'))['members']
    previous_weight=chosen['june']['previous_weight']
    members=[]
    for name,item in second.items():
        members.append({'name':'round_02_anniversary_history/'+name,'experiment_root':'round_02_anniversary_history',
            'kind':item['kind'],'artifact_name':item['name'],
            'weight':previous_weight*item['weight']})
    for name,item in chosen['june']['members'].items():
        members.append({'name':'round_03_tree_transformer_ensemble/'+name,'experiment_root':'round_03_tree_transformer_ensemble',
            'kind':item['kind'],'artifact_name':item['name'],'weight':item['weight']})
    np.testing.assert_allclose(sum(m['weight'] for m in members),1.,atol=1e-12,rtol=0)
    verified=[]
    for stage in ('outer','june'):
        saved=pd.read_csv(ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv').sort_values('source_row')
        combined=np.zeros(len(saved))
        for member in members:
            directory=ROOT/'output/models'/member['experiment_root']/member['kind']/stage/member['artifact_name']
            frame=pd.read_csv(directory/'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(frame.source_row,saved.source_row)
            np.testing.assert_array_equal(frame.actual,saved.actual)
            np.testing.assert_array_equal(frame.month,saved.month)
            combined+=member['weight']*frame.prediction.to_numpy(float)
            if stage=='june':
                report=ROOT/'output/experiment_runs'/member['experiment_root']/member['kind']/stage/member['artifact_name']/'results.json'
                record=json.loads(report.read_text(encoding='utf-8'))
                spec=record['spec']
                member['family']=spec['family']
                member['features']=record['features']
                member['iterations']=record['iterations']
                member['target']=spec.get('target','raw')
                member['checkpoint_june']=str((directory/'model.joblib').relative_to(ROOT)).replace('\\','/')
        np.testing.assert_allclose(combined,saved.prediction,atol=1e-8,rtol=0)
        metrics=score(saved.actual,combined)
        for key,value in metrics.items():
            np.testing.assert_allclose(value,chosen[stage][key],atol=1e-8,rtol=0)
        verified.append({'stage':stage,'rows':len(saved),**metrics,
            'prediction_path':f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv',
            'max_reconstruction_difference_sgd':float(np.max(np.abs(combined-saved.prediction)))})
    pd.DataFrame(members).to_csv(OUT/'best_ensemble_members.csv',index=False,encoding='utf-8-sig')
    outer=chosen['outer'];june=chosen['june']
    result={'name':NAME,'recommended_role':'正文最终最优方案，必须与单模型对照一起报告',
        'forecasting_data_scope':'train.csv + supplied auxiliary; no public rental target matching',
        'annual':{'months':['2024-04','2025-03'],'rows':outer['rows'],
            **{k:outer[k] for k in ('rmse_sgd','mae_sgd','bias_sgd')}},
        'recent':{'months':['2024-06','2025-03'],'rows':june['rows'],
            **{k:june[k] for k in ('rmse_sgd','mae_sgd','bias_sgd')}},
        'top_level_weights':{'second_wave_ensemble':previous_weight,
            **{name:item['weight'] for name,item in june['members'].items()}},
        'leaf_members':members,'verification':verified,
        'selection_note':'Weights were selected on earlier chronological predictions, but keeping only the new-model group was selected after viewing later development results. This is the lowest development score, not an independent test score.',
        'feature_note':'The base blend combines normalized market rent and exact/window/broad 12/24/36-month cohorts. FT and residual XGBoost add past-level shrinkage features. Member predictions are all reconstructed to SGD rent before convex averaging.'}
    write_json(OUT/'best_ensemble.json',result)
    row={'family':'best_ensemble','model':NAME,
        'report_purpose':'最终最佳预测方案；连接历史特征与异质模型互补性的报告结论',
        'features':'成员分别22/62/67/87列','fitting_rows':114842,'validation_rows':outer['rows'],
        'period':'2024-04..2025-03','selection':'开发集评估后选择新增模型组；提前选择的组件权重保留',
        'source':'output/experiment_runs/00_report_and_final/best_ensemble.json',
        **{k:outer[k] for k in ('rmse_sgd','mae_sgd','bias_sgd')},
        'recent_period':'2024-06..2025-03','recent_validation_rows':june['rows'],
        'recent_rmse_sgd':june['rmse_sgd'],'recent_mae_sgd':june['mae_sgd'],
        'comparison_scope':'整条增强特征融合管线；不属于同27列的架构对照'}
    lines=['','### 正文第8项：最优历史特征融合（必须纳入推荐组）','',
        '这是报告的最终性能方案，不能只在附录或实验进展中一笔带过。它与前7个单模型一起构成推荐的完整模型组；前7项用于架构比较，本项用于展示特征工程与模型互补性结合后的最佳开发表现。','',
        '| 验证范围 | 行数 | RMSE（SGD） | MAE（SGD） | 平均偏差 |','|---|---:|---:|---:|---:|',
        f"| 2024-04—2025-03 | {outer['rows']:,} | **{outer['rmse_sgd']:.4f}** | {outer['mae_sgd']:.4f} | {outer['bias_sgd']:+.4f} |",
        f"| 2024-06—2025-03 | {june['rows']:,} | **{june['rmse_sgd']:.4f}** | {june['mae_sgd']:.4f} | {june['bias_sgd']:+.4f} |",'',
        '融合组成（预测均先还原成SGD月租，再加权）：','',
        '| 组成 | 权重 | 主要信息 |','|---|---:|---|',
        f"| 第二轮历史特征树融合（内含7个LightGBM/CatBoost成员） | {previous_weight:.4%} | 相对市场租金、12/24/36个月同期历史；精确月与邻月、楼栋与更粗粒度 |"]
    labels={'models/ft_transformer':('FT-Transformer残差模型','62列历史特征＋5列过去市场水平/收缩特征'),
        'models/xgb5_residual':('XGBoost深度5残差模型','同上，预测相对过去基准的残差'),
        'models/xgb7':('XGBoost深度7直接模型','62列历史特征，直接预测SGD租金')}
    for name,item in june['members'].items():
        label,features=labels[name]
        lines.append(f"| {label} | {item['weight']:.4%} | {features} |")
    lines+=['','所以这是**4个顶层组成、展开后10个模型成员**的融合，不是一个独立的“442.90单模型”。全部展开权重、特征数量、迭代数和检查点位置见`best_ensemble_members.csv`。',
        '','核心解释是：同期历史先降低单模型误差，树模型与FT的不同误差再通过融合进一步降低RMSE。不是单独将FT换进27列主表就能达到442.90。',
        '','该方案使用train和所给auxiliary，符合当前数据范围。组别是在查看后续开发结果后选出的，因此正文应称为“最佳开发方案”；预先锁定完整融合的443.0052可以作为稳健性对照。两个窗口重叠，不能合并样本量。','']
    return result,row,lines


if __name__=='__main__':
    result,row,_=build()
    path=OUT/'recommended_models.csv'
    existing=pd.read_csv(path)
    existing=existing.loc[existing.family!='best_ensemble']
    pd.concat([existing,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False,encoding='utf-8-sig')
    print(json.dumps({'recommended_models':len(existing)+1,'annual_rmse':result['annual']['rmse_sgd'],
        'recent_rmse':result['recent']['rmse_sgd'],'leaf_members':len(result['leaf_members'])},indent=2))
