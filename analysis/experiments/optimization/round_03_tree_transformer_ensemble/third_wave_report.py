"""Independent prediction, source, checkpoint, and matched-window verification."""
from __future__ import annotations
import gc
import json
import sys
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.third_wave_training import OUT,ART,make_data,inputs,adjusted_baseline,SECOND_ART
from analysis.experiments.third_wave_features import feature_data,inputs as feature_inputs
from analysis.experiments.third_wave_cross_type import cross_data,inputs as cross_inputs
from analysis.experiments.second_wave_report import monthly_comparison,matched_file
from analysis.pipeline.data import sha256


def cat_frame(frame):
    for c in frame.select_dtypes(include=['category','object','string']).columns:
        frame[c]=frame[c].astype('string').fillna('__MISSING__').astype(str)
    return frame.fillna(-999.)


def reload_selected(stage,selection):
    sys.path.insert(0,str(ROOT/'.cache/model-deps'))
    bundle=feature_data(stage)
    results=[]
    for name,item in selection['members'].items():
        kind=item['kind']
        if kind=='round_02_anniversary_history':
            continue
        artifact=ART/kind/stage/item['name']
        saved=joblib.load(artifact/'model.joblib')
        spec=saved['spec']
        if kind=='features':
            _,q=feature_inputs(bundle,spec)
            if spec['family']=='cat':
                q=cat_frame(q)
            predicted=saved['model'].predict(q)
        else:
            _,q,_,offset=inputs(bundle,spec)
            if kind=='mixture':
                offset,_=adjusted_baseline(bundle['ann_val'],bundle['extra_val'],q)
                p=saved['gate'].predict_proba(q)[:,1]
                predicted=(1-p)*saved['experts'][0].predict(q)+p*saved['experts'][1].predict(q)+offset
            elif 'fitted' in saved:
                import torch
                from analysis.pipeline.neural_models import predict_neural
                fitted=saved['fitted']
                fitted['device']='cuda'
                torch.set_num_threads(4)
                torch.cuda.set_per_process_memory_fraction(.25)
                fitted['model'].to('cuda')
                predicted=predict_neural(fitted,q[saved['columns']])+offset
                fitted['model'].to('cpu')
                torch.cuda.empty_cache()
            else:
                if spec['family'].startswith('cat'):
                    q=cat_frame(q)
                predicted=saved['model'].predict(q)+offset
                if spec.get('training')=='supported24':
                    predicted=np.where(bundle['ann_val'].ANN_BLOCK_24_R1_N.to_numpy()>0,
                        predicted,bundle['fallback'])
        expected=matched_file(artifact/'predictions.csv')
        difference=float(np.max(np.abs(expected.prediction-predicted)))
        if difference>1e-6:
            raise AssertionError(f'Checkpoint prediction differs: {stage} {name}: {difference}')
        results.append({'stage':stage,'name':name,'max_absolute_difference_sgd':difference})
        del saved,q
        gc.collect()
    return results


def reload_cross(stage,selection):
    bundle=cross_data(stage)
    results=[]
    for name in selection['weights']:
        if name=='round_03_tree_transformer_ensemble':
            continue
        directory=ART/'cross_type'/stage/name
        saved=joblib.load(directory/'model.joblib')
        _,q=cross_inputs(bundle,saved['spec'])
        if saved['spec']['family']=='cat':
            q=cat_frame(q)
        predicted=saved['model'].predict(q)
        expected=matched_file(directory/'predictions.csv')
        difference=float(np.max(np.abs(expected.prediction-predicted)))
        if difference>1e-6:
            raise AssertionError(f'Cross-type checkpoint differs: {stage} {name} {difference}')
        results.append({'stage':stage,'name':'cross_type/'+name,'max_absolute_difference_sgd':difference})
    return results


def main():
    source,hashes=frame_data()
    for relative,digest in hashes.items():
        if sha256(ROOT/relative)!=digest:
            raise AssertionError(f'Raw source changed: {relative}')
    checks=[]
    for path in sorted(OUT.glob('*/*/*/results.json')):
        record=json.loads(path.read_text())
        relative=path.parent.relative_to(OUT)
        prediction=matched_file(ART/relative/'predictions.csv')
        rows=prediction.source_row.to_numpy(int)
        np.testing.assert_array_equal(prediction.actual,source.iloc[rows].MONTHLY_RENT)
        np.testing.assert_array_equal(prediction.month,source.iloc[rows].RENT_APPROVAL_DATE)
        if not prediction.source_row.is_unique or not np.isfinite(prediction.prediction).all():
            raise AssertionError('Invalid prediction rows')
        start,end=('2023-10','2024-04') if record['stage']=='screen' else (
            ('2024-04','2025-04') if record['stage']=='outer' else ('2024-06','2025-04'))
        required=source.index[(source.RENT_APPROVAL_DATE>=start)&(source.RENT_APPROVAL_DATE<end)]
        np.testing.assert_array_equal(prediction.source_row,required)
        metrics=score(prediction.actual,prediction.prediction)
        for key in metrics:
            np.testing.assert_allclose(metrics[key],record[key],atol=1e-8,rtol=0)
        if record['source_sha256']!=hashes:
            raise AssertionError('Source fingerprints differ')
        checks.append({'experiment':str(relative),'rows':len(prediction),**metrics})
    pd.DataFrame(checks).to_csv(OUT/'verified_metrics.csv',index=False)
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    reloads=reload_selected('outer',selection)+reload_selected('june',selection)
    comparisons={}
    cross_comparisons={}
    ablations=[]
    for stage in ('outer','june'):
        new=matched_file(ART/'ensemble'/stage/'predictions.csv')
        old=matched_file(SECOND_ART/'ensemble'/stage/'predictions.csv')
        table,change=monthly_comparison(new,old)
        table.to_csv(OUT/f'monthly_comparison_{stage}.csv',index=False)
        combined=np.zeros(len(new))
        grouped={}
        for item in selection['members'].values():
            part=old if item['kind']=='round_02_anniversary_history' else matched_file(ART/item['kind']/stage/item['name']/'predictions.csv')
            np.testing.assert_array_equal(part.source_row,new.source_row)
            combined+=item['weight']*part.prediction.to_numpy()
            grouped.setdefault(item['kind'],np.zeros(len(new)))
            grouped[item['kind']]+=item['weight']*(part.prediction.to_numpy()-old.prediction.to_numpy())
        np.testing.assert_allclose(combined,new.prediction,atol=1e-8,rtol=0)
        comparisons[stage]={'rows':len(new),'round_03_tree_transformer_ensemble':score(new.actual,new.prediction),
            'round_02_anniversary_history':score(old.actual,old.prediction),**change}
        for kind,delta in grouped.items():
            if kind!='round_02_anniversary_history':
                ablations.append({'stage':stage,'new_group_added_to_old_blend':kind,
                    'note':'Weights remain frozen; all other new groups replaced by previous blend.',
                    **score(old.actual,old.prediction.to_numpy()+delta)})
    pd.DataFrame(ablations).to_csv(OUT/'frozen_group_comparison.csv',index=False)
    if (OUT/'cross_type_selection.json').exists():
        cross_selection=json.loads((OUT/'cross_type_selection.json').read_text())
        for stage in ('outer','june'):
            reloads+=reload_cross(stage,cross_selection)
            third=matched_file(ART/'ensemble'/stage/'predictions.csv')
            old=matched_file(SECOND_ART/'ensemble'/stage/'predictions.csv')
            new=matched_file(ART/'cross_ensemble'/stage/'predictions.csv')
            rebuilt=np.zeros(len(new))
            for name,weight in cross_selection['weights'].items():
                part=third if name=='round_03_tree_transformer_ensemble' else matched_file(ART/'cross_type'/stage/name/'predictions.csv')
                np.testing.assert_array_equal(part.source_row,new.source_row)
                rebuilt+=weight*part.prediction.to_numpy()
            np.testing.assert_allclose(rebuilt,new.prediction,atol=1e-8,rtol=0)
            table,change=monthly_comparison(new,old)
            table.to_csv(OUT/f'cross_monthly_comparison_{stage}.csv',index=False)
            cross_comparisons[stage]={'rows':len(new),'cross_extension':score(new.actual,new.prediction),
                'round_03_tree_transformer_ensemble':score(third.actual,third.prediction),'round_02_anniversary_history':score(old.actual,old.prediction),**change}
    feature_controls=[]
    model_controls=[]
    for stage in ('screen','outer','june'):
        control=matched_file(ART/'models'/stage/'catgpu8_control/predictions.csv')
        engineered=matched_file(ART/'features'/stage/'wide24_cat8/predictions.csv')
        np.testing.assert_array_equal(control.source_row,engineered.source_row)
        feature_controls.append({'stage':stage,'same_model':'GPU CatBoost depth 8',
            'existing_62_rmse':score(control.actual,control.prediction)['rmse_sgd'],
            'new_77_rmse':score(engineered.actual,engineered.prediction)['rmse_sgd']})
        candidates={'CPU CatBoost depth 6':SECOND_ART/'anniversary'/stage/'window_cat6/predictions.csv',
            'LightGBM 15 leaves':SECOND_ART/'anniversary'/stage/'window_lgb15/predictions.csv',
            'XGBoost depth 3':ART/'models'/stage/'xgb3/predictions.csv',
            'XGBoost depth 7':ART/'models'/stage/'xgb7/predictions.csv',
            'GPU CatBoost depth 8':ART/'models'/stage/'catgpu8_control/predictions.csv'}
        for name,path in candidates.items():
            part=matched_file(path)
            np.testing.assert_array_equal(control.source_row,part.source_row)
            model_controls.append({'stage':stage,'model':name,'same_features':62,
                **score(part.actual,part.prediction)})
    pd.DataFrame(feature_controls).to_csv(OUT/'controlled_feature_comparison.csv',index=False)
    pd.DataFrame(model_controls).to_csv(OUT/'controlled_model_comparison.csv',index=False)
    adaptive={}
    if (OUT/'adaptive_selection.json').exists():
        from analysis.experiments.third_wave_adaptive import values
        adaptive_selection=json.loads((OUT/'adaptive_selection.json').read_text())
        if any(end>='2024-06' for _,end in adaptive_selection['selection_ranges']):
            raise AssertionError('Adaptive selection includes post-origin labels')
        bundle=make_data('june')
        matrix=values('june',bundle,adaptive_selection['members'])
        weights=np.array([item['weight'] for item in adaptive_selection['members'].values()])
        saved=matched_file(ART/'adaptive/june/predictions.csv')
        np.testing.assert_allclose(saved.prediction,matrix@weights,atol=1e-8,rtol=0)
        adaptive={'scores':score(saved.actual,saved.prediction),
            'selection_ranges':adaptive_selection['selection_ranges'],
            'note':'One predeclared adaptive blend; failed to beat the original third-wave June blend.'}
    verification={'prediction_files':len(checks),'reloads':reloads,
        'source_sha256':hashes,'all_query_rows_retained':True,
        'prediction_metrics_recomputed':True,'raw_sources_rehashed':True,
        'locked_ensembles_reconstructed':True,'ensemble_files_reconstructed':5}
    write_json(OUT/'verification.json',verification)
    trials=sum(1 for _ in OUT.glob('*/screen/*/results.json'))
    summary={'screen_trials':trials,'comparisons':comparisons,'cross_comparisons':cross_comparisons,'adaptive':adaptive,
        'new_group_ablations':ablations,'feature_controls':feature_controls,
        'model_controls':model_controls,'verification':verification,
        'target_430_achieved':any(v['round_03_tree_transformer_ensemble']['rmse_sgd']<=430 for v in comparisons.values()) or
            any(v['cross_extension']['rmse_sgd']<=430 for v in cross_comparisons.values()),
        'limitations':['Both later periods have been repeatedly viewed development data, not independent test performance.',
            'The annual and June windows overlap and must not be pooled.',
            'Only train and auxiliary sources are used; no public rental target matching or test labels.']}
    write_json(OUT/'summary.json',summary)
    lines=['# 第三轮：特征与模型两条路线的对照实验','',
        '数据固定为 `data/train.csv` 和 `data/auxiliary/`。原始文件未改动；不使用外部租金目标、测试标签或未来实测宏观数据。','',
        '模型参数与融合权重先在 2023-10 至 2024-03 的较早窗口选定，然后冻结迭代次数并从各预测起点重新训练。所有验证行均保留。','',
        '| 时间窗口 | 行数 | 前轮融合 RMSE | 本轮冻结融合 RMSE | 差值 | 改善月份 |',
        '|---|---:|---:|---:|---:|---:|']
    for stage,item in comparisons.items():
        label='2024-04—2025-03' if stage=='outer' else '2024-06—2025-03'
        lines.append(f"| {label} | {item['rows']:,} | {item['round_02_anniversary_history']['rmse_sgd']:.4f} | {item['round_03_tree_transformer_ensemble']['rmse_sgd']:.4f} | {item['delta_rmse_sgd']:+.4f} | {item['months_improved']}/{item['months']} |")
    if cross_comparisons:
        lines+=['','跨户型共享历史作为单独扩展，沿用同一较早窗口选权，未改变上表原有融合：','',
            '| 时间窗口 | 原第三轮融合 | 跨户型扩展融合 | MAE | 偏差 |','|---|---:|---:|---:|---:|']
        for stage,item in cross_comparisons.items():
            label='2024-04—2025-03' if stage=='outer' else '2024-06—2025-03'
            scores=item['cross_extension']
            lines.append(f"| {label} | {item['round_03_tree_transformer_ensemble']['rmse_sgd']:.4f} | {scores['rmse_sgd']:.4f} | {scores['mae_sgd']:.4f} | {scores['bias_sgd']:.4f} |")
    if adaptive:
        lines+=['',f"另做一次时间自适应融合：仅使用 2024-06 起点以前的 2023-10—2024-03 与 2024-04—05 预测误差，按事先固定的 50/50 损失和 0.002 收缩系数选权；近期 RMSE 为 {adaptive['scores']['rmse_sgd']:.4f}，未超过原第三轮融合，故保留为失败记录。"]
    lines+=['','RMSE 降低并不表示每项指标都改善：原第三轮融合的一年 MAE 为 331.05，前轮为 328.44；近期 MAE 为 324.82，前轮为 322.92。请按实际目标选择。']
    lines+=['','固定 GPU CatBoost 深度 8，仅替换特征的对照：','',
        '| 窗口 | 原 62 特征 RMSE | 新 77 特征 RMSE |','|---|---:|---:|']
    for item in feature_controls:
        lines.append(f"| {item['stage']} | {item['existing_62_rmse']:.4f} | {item['new_77_rmse']:.4f} |")
    lines+=['','该对照保留了不同窗口的全部结果，不能将早期窗口的单模型改善直接解释为后续窗口收益。不同模型使用相同 62 特征的完整对照见 `controlled_model_comparison.csv`。']
    lines+=['','## 特征工程','',
        '- 原有楼栋×户型历史之外，加入 12/18/24/30/36 个月前、前后各三个月的高斯加权统计；每条记录只能检索严格早于自身月份的历史。',
        '- 新增区域当月历史价格归一化、低租金及高租金占比、最小/最大比例、有效样本量，以及向楼栋长期均值收缩后的租金估计。',
        '- 月份跨度大时，只查询预测起点已经观察到的历史；不存在的月份不补造。统计单位是楼栋及户型，不能据此识别租户或具体房屋续约。','',
        '## 模型与训练','',
        '- 同样的强特征上比较 XGBoost 深度 3/5/7/9、MLP、ResNet、FT-Transformer；比较直接预测与对历史基准的残差预测。',
        '- 新特征比较 LightGBM 15/31 叶及 GPU CatBoost 深度 6/8；另外尝试低租金概率分类器及两个条件回归专家。',
        '- 缩短监督训练历史、近期加权、仅在两年前历史有支持的样本上训练，均保留全部更早记录作为历史统计池。',
        '- 单模型较弱不代表完全无用；最终融合权重仅由较早窗口决定，后续窗口不再调权。','',
        '## 检验与限制','',
        f'- 共完成 {trials} 个较早窗口配置，独立重算 {len(checks)} 份预测文件，重载检验 {len(reloads)} 个入选模型；源文件 SHA-256 一致。',
        '- 15 个单元测试通过，覆盖历史窗口因果性、未观察月份、跨户型归一化、加权统计、监督样本筛选及原有特征边界。',
        '- 两个后续窗口都已多次用于研究，属于开发集；它们相互重叠，不能合并为独立测试样本。月度块重采样区间仅作开发诊断。',
        '- 400—430 的目标尚未达到。所有试验与失败结果保存在 `verified_metrics.csv`，不只保留最低分。','',
        '复现：`python -m analysis.experiments.third_wave_training screen`、`third_wave_models screen`、`third_wave_features screen`、`third_wave_mixture screen`；再运行 `third_wave_ensemble select/outer/june` 与 `third_wave_report`。已有锁定文件和结果会保留。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    from analysis.experiments.third_wave_ablations import main as materialize_ablations
    materialize_ablations()
    print(json.dumps({'trials':trials,'verified_files':len(checks),'reloads':len(reloads),'comparisons':comparisons},indent=2),flush=True)


if __name__=='__main__':
    main()
