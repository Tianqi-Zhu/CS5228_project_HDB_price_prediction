"""Materialize group-removal diagnostics; explicitly mark post-evaluation selection."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import frame_data,score,write_json
from analysis.experiments.second_wave_report import matched_file,monthly_comparison
from analysis.experiments.third_wave_training import OUT,ART,SECOND_ART


def main():
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    reference=pd.read_csv(OUT/'frozen_group_comparison.csv')
    source,hashes=frame_data()
    records=[]
    for stage in ('outer','june'):
        old=matched_file(SECOND_ART/'ensemble'/stage/'predictions.csv')
        for kind in ('training','models','features','mixture'):
            members={name:item for name,item in selection['members'].items() if item['kind']==kind}
            previous_weight=1-sum(item['weight'] for item in members.values())
            prediction=previous_weight*old.prediction.to_numpy()
            for item in members.values():
                part=matched_file(ART/kind/stage/item['name']/'predictions.csv')
                np.testing.assert_array_equal(part.source_row,old.source_row)
                prediction+=item['weight']*part.prediction.to_numpy()
            expected=reference.loc[(reference.stage==stage)&(reference.new_group_added_to_old_blend==kind)].iloc[0]
            metrics=score(old.actual,prediction)
            for key in metrics:
                np.testing.assert_allclose(metrics[key],expected[key],atol=1e-8,rtol=0)
            result=old.copy()
            result['prediction']=prediction
            np.testing.assert_array_equal(result.actual,source.iloc[result.source_row].MONTHLY_RENT)
            np.testing.assert_array_equal(result.month,source.iloc[result.source_row].RENT_APPROVAL_DATE)
            destination=ART/'ablations'/stage/kind
            destination.mkdir(parents=True,exist_ok=True)
            result.to_csv(destination/'predictions.csv',index=False)
            table,delta=monthly_comparison(result,old)
            record={'stage':stage,'kind':kind,'rows':len(result),'previous_weight':previous_weight,
                'members':members,**metrics,**delta,
                'note':'Group removal was inspected after later-period evaluation. Weights were frozen earlier, but the preferred group is a development-set selection.'}
            records.append(record)
            table.to_csv(destination/'monthly_comparison.csv',index=False)
    best_june=min((r for r in records if r['stage']=='june'),key=lambda r:r['rmse_sgd'])
    result={'records':records,'best_june':best_june,'source_sha256':hashes,
        'ablation_prediction_files_verified':len(records),
        'post_evaluation_diagnostic':True}
    write_json(OUT/'ablation_models.json',result)
    summary=json.loads((OUT/'summary.json').read_text())
    summary['development_ablations']=result
    summary['verification']['ablation_prediction_files_verified']=len(records)
    summary['best_development_june']={'rmse_sgd':best_june['rmse_sgd'],'kind':best_june['kind'],
        'selection':'Group chosen after development evaluation; not an independent test score.'}
    write_json(OUT/'summary.json',summary)
    verification=json.loads((OUT/'verification.json').read_text())
    verification['ablation_prediction_files_verified']=len(records)
    write_json(OUT/'verification.json',verification)
    text=(OUT/'README.md').read_text(encoding='utf-8')
    marker='\n## 开发集消融的最低值\n'
    text=text.split(marker)[0]
    lines=[marker,'','把本轮新增的特征组、模型组、训练策略组及混合专家组分别加到前轮融合中，其他新增组的权重还给前轮融合。使用原有冻结权重，不重新拟合系数。','',
        '| 保留的本轮新增组 | 一年 RMSE | 近期 RMSE |','|---|---:|---:|']
    for kind in ('training','models','features','mixture'):
        outer=next(r for r in records if r['stage']=='outer' and r['kind']==kind)
        june=next(r for r in records if r['stage']=='june' and r['kind']==kind)
        lines.append(f"| {kind} | {outer['rmse_sgd']:.4f} | {june['rmse_sgd']:.4f} |")
    lines+=['','最低开发值来自仅保留本轮新增模型组：前轮融合权重约 76.26%，FT-Transformer 15.99%，XGBoost 残差模型 5.52%，XGBoost 深度 7 为 2.23%。一年 451.1810，近期 442.9025。该方案近期 MAE 为 323.7295，前轮为 322.9187。',
        '','**这是查看后续开发窗口后的分组选择。预先锁定的完整第三轮方案仍为一年 451.3097／近期 443.0052；不能把消融最低值称为独立测试成绩。**',
        '','八份消融预测均保留全部对应验证行，标签及月份与源数据逐行核对，指标与分组对照重新计算一致。文件位于 `output/models/round_03_tree_transformer_ensemble/ablations/`。','']
    (OUT/'README.md').write_text(text+'\n'.join(lines),encoding='utf-8')
    print(json.dumps({'best_development_june':summary['best_development_june'],'verified_ablation_predictions':len(records)},indent=2),flush=True)


if __name__=='__main__':
    main()
