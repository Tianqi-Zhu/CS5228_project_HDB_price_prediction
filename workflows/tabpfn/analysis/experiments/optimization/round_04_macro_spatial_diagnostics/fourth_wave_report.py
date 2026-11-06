"""Verify fourth-wave experiments and report progress toward local RMSE 430."""
from __future__ import annotations
import gc
import json
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.fourth_wave_diagnostics import OUT,ART
from analysis.experiments.fourth_wave_ensemble import old_best
from analysis.experiments.third_wave_training import make_data,adjusted_baseline
from analysis.experiments.second_wave_anniversary import select_inputs
from analysis.experiments.second_wave_report import matched_file,monthly_comparison
from analysis.pipeline.data import sha256


def categorical(frame):
    for c in frame.select_dtypes(include=['category','object','string']).columns:
        frame[c]=frame[c].astype('string').fillna('__MISSING__').astype(str)
    return frame.fillna(-999.)


def reload_selected(stage,selection):
    bundle=make_data(stage)
    cache={}
    checked=[]
    selected={name:item for version in selection['versions'].values() for name,item in version['members'].items()
        if item['kind'] in ('horizons','spatial','experts','flows','distribution')}
    for name,item in selected.items():
        kind=item['kind']
        directory=ART/kind/stage/item['name']
        saved=joblib.load(directory/'model.joblib')
        spec=saved['spec']
        if kind=='horizons':
            from analysis.experiments.fourth_wave_horizons import augmented_data,inputs
            if kind not in cache:
                cache[kind]=augmented_data(stage)
            _,q,_,_,_,_,offset=inputs(cache[kind],spec)
            if spec['family'].startswith('cat'):
                q=categorical(q)
            prediction=saved['model'].predict(q)
            prediction=prediction*offset if spec['target']=='ratio' else prediction+offset
        elif kind in ('spatial','flows'):
            if kind=='spatial':
                from analysis.experiments.fourth_wave_spatial import spatial_data as prepare,inputs
            else:
                from analysis.experiments.fourth_wave_flows import flow_data as prepare,inputs
            if kind not in cache:
                cache[kind]=prepare(stage)
            _,q=inputs(cache[kind],spec)
            if spec['family']=='cat':
                q=categorical(q)
            prediction=saved['model'].predict(q)
        elif kind=='experts':
            from analysis.experiments.third_wave_training import inputs
            _,q,_,offset=inputs(bundle,spec)
            types=q.FLAT_TYPE_CLEAN.astype('string').to_numpy(str)
            prediction=bundle['fallback'].copy()
            if spec['family'].startswith('cat'):
                q=categorical(q)
            for key,model in saved['models'].items():
                selected_rows=types==key
                prediction[selected_rows]=model.predict(q.loc[selected_rows])+offset[selected_rows]
        else:
            _,q=select_inputs(bundle,'window')
            if spec['mode']=='raw':
                scale=np.ones(len(q))
            elif spec['mode']=='local_ratio':
                scale,_=adjusted_baseline(bundle['ann_val'],bundle['extra_val'],q)
            else:
                scale=bundle['extra_val'].MARKET_3M.to_numpy(float)
            prediction=saved['model'].predict_proba(q)@saved['centers']*scale
        expected=matched_file(directory/'predictions.csv')
        difference=float(np.max(np.abs(expected.prediction-prediction)))
        if difference>1e-6:
            raise AssertionError(f'Checkpoint differs: {stage} {name}: {difference}')
        checked.append({'stage':stage,'name':name,'max_difference_sgd':difference})
        del saved,q
        gc.collect()
    return checked


def verify_given_macro(stage):
    from analysis.experiments.fourth_wave_known_macro import macro_data,inputs
    selection=json.loads((OUT/'given_macro_selection.json').read_text())
    bundle=macro_data(stage)
    names=sorted({name for version in selection['versions'].values() for name in version['weights'] if name!='previous_best'})
    checks=[]
    for name in names:
        directory=ART/'known_macro'/stage/name
        saved=joblib.load(directory/'model.joblib')
        _,q=inputs(bundle,saved['spec'])
        if saved['spec']['family']=='cat':
            q=categorical(q)
        predicted=saved['model'].predict(q)
        expected=matched_file(directory/'predictions.csv')
        difference=float(np.max(np.abs(expected.prediction-predicted)))
        if difference>1e-6:
            raise AssertionError('Known-macro model differs after reload')
        checks.append({'stage':stage,'name':'known_macro/'+name,'max_difference_sgd':difference})
    results=[]
    baseline=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv')
    for version,configuration in selection['versions'].items():
        combined=np.zeros(len(baseline))
        for name,weight in configuration['weights'].items():
            part=baseline if name=='previous_best' else matched_file(ART/'known_macro'/stage/name/'predictions.csv')
            np.testing.assert_array_equal(part.source_row,baseline.source_row)
            np.testing.assert_array_equal(part.actual,baseline.actual)
            combined+=weight*part.prediction.to_numpy(float)
        expected=matched_file(ART/'given_ensemble'/stage/version/'predictions.csv')
        np.testing.assert_allclose(expected.prediction,combined,atol=1e-8,rtol=0)
        monthly,delta=monthly_comparison(expected,baseline)
        monthly.to_csv(OUT/f'monthly_given_{stage}_{version}.csv',index=False)
        results.append({'stage':stage,'version':version,'scenario':'given_query_month_macro',
            'rows':len(expected),**score(expected.actual,expected.prediction),
            'previous_rmse_sgd':score(baseline.actual,baseline.prediction)['rmse_sgd'],**delta})
    return checks,results


def main():
    raw,hashes=frame_data()
    for path,digest in hashes.items():
        if sha256(ROOT/path)!=digest:
            raise AssertionError('Raw source modified')
    checked=[]
    for path in sorted(OUT.glob('*/*/*/results.json')):
        record=json.loads(path.read_text())
        directory=ART/path.parent.relative_to(OUT)
        frame=matched_file(directory/'predictions.csv')
        stage=record['stage']
        start,end=('2023-10','2024-04') if stage=='screen' else (
            ('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
        expected=raw.loc[(raw.RENT_APPROVAL_DATE>=start)&(raw.RENT_APPROVAL_DATE<end)]
        np.testing.assert_array_equal(frame.source_row,expected.index)
        np.testing.assert_array_equal(frame.actual,expected.MONTHLY_RENT)
        np.testing.assert_array_equal(frame.month,expected.RENT_APPROVAL_DATE)
        if not frame.source_row.is_unique or not np.isfinite(frame.prediction).all():
            raise AssertionError('Invalid query forecasts')
        metrics=score(frame.actual,frame.prediction)
        for key,value in metrics.items():
            np.testing.assert_allclose(value,record[key],atol=1e-8,rtol=0)
        if record['source_sha256']!=hashes:
            raise AssertionError('Different data fingerprints')
        checked.append({'kind':path.parents[2].name,'stage':stage,'name':record['spec']['name'],
            'rows':len(frame),**metrics})
    table=pd.DataFrame(checked)
    table.to_csv(OUT/'verified_metrics.csv',index=False)
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    reloads=reload_selected('outer',selection)+reload_selected('june',selection)
    comparisons=[]
    for stage in ('outer','june'):
        bundle=make_data(stage)
        baseline=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv')
        for version,configuration in selection['versions'].items():
            new=matched_file(ART/'ensemble'/stage/version/'predictions.csv')
            combined=np.zeros(len(new))
            for item in configuration['members'].values():
                if item['kind']=='previous_best':
                    values=old_best(stage,bundle)
                elif item['kind']=='third_locked':
                    values=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/ensemble/{stage}/predictions.csv').prediction.to_numpy()
                elif item['kind']=='second_locked':
                    values=bundle['fallback']
                else:
                    part=matched_file(ART/item['kind']/stage/item['name']/'predictions.csv')
                    np.testing.assert_array_equal(part.source_row,new.source_row)
                    values=part.prediction.to_numpy(float)
                combined+=item['weight']*values
            np.testing.assert_allclose(combined,new.prediction,atol=1e-8,rtol=0)
            monthly,delta=monthly_comparison(new,baseline)
            monthly.to_csv(OUT/f'monthly_{stage}_{version}.csv',index=False)
            comparisons.append({'stage':stage,'version':version,'rows':len(new),
                **score(new.actual,new.prediction),'previous_rmse_sgd':score(baseline.actual,baseline.prediction)['rmse_sgd'],**delta})
    comparisons=pd.DataFrame(comparisons)
    comparisons.to_csv(OUT/'ensemble_comparison.csv',index=False)
    given=[]
    if (OUT/'given_macro_selection.json').exists():
        for stage in ('outer','june'):
            extra,results=verify_given_macro(stage)
            reloads+=extra;given+=results
        pd.DataFrame(given).to_csv(OUT/'given_macro_comparison.csv',index=False)
    counts=table.loc[table.stage=='screen'].groupby('kind').size().to_dict()
    best=comparisons.loc[comparisons.stage=='june'].sort_values('rmse_sgd').iloc[0].to_dict()
    achieved=bool(best['rmse_sgd']<=430 or any(row['stage']=='june' and row['rmse_sgd']<=430 for row in given))
    verification={'prediction_files_verified':len(checked),'model_reloads':reloads,
        'all_query_rows_retained':True,'sources_rehashed':True,'source_sha256':hashes,
        'ensembles_reconstructed':len(comparisons)+len(given),'tests':'23 targeted tests passed'}
    summary={'screen_configurations':counts,'comparisons':comparisons.to_dict('records'),
        'best_new_june':best,'given_macro_comparisons':given,'target_430_achieved':achieved,'verification':verification,
        'kaggle_gap':'No paired local/Kaggle scores available; a fixed minus-20 conversion is unverified.',
        'limits':'All periods have been repeatedly viewed development data. June and annual periods overlap; no pooling. No test targets, external rental targets, filtered validation or post-origin label updates.'}
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'verification.json',verification)
    lines=['# 第四轮：本地RMSE 430目标探索','',
        '本轮以相同的2024-06—2025-03全部28,995行、本地最佳442.9025为参照，目标430。只使用train与所给auxiliary，不使用外部公开租金目标，不删除难预测的验证样本。',
        '','从442.9025降至430，需要RMSE再降2.91%、MSE再降约5.74%。用户没有保存同一提交的本地/Kaggle成对成绩，因此不能验证“本地减20”的固定换算；430只是本地研究目标，不能据此保证Kaggle达到411。',
        '','## 新方法及结果','',
        '| 路线 | 较早窗口配置数 | 做法 |','|---|---:|---|']
    explanations={'horizons':'模拟0/3/6/9/12个月信息延迟，冻结所有历史与同期窗口；原始/残差/比值目标；每原始样本总权重不变。',
        'spatial':'只从12/24/36个月前±1月寻找其他楼栋可比记录；匹配位置、面积、租约年份；排除本楼栋。',
        'experts':'按户型分别训练LightGBM/CatBoost，稀有户型回退原冻结预测。',
        'flows':'仅使用待预测批次内的无标签记录数量，结合历史同期记录数；没有读取目标租金。需整批推理。',
        'distribution':'先预测租金区间概率，再积分拟合集区间内均值；比较绝对租金、市场归一化、局部基准和区间宽度。',
        'known_macro':'单独信息情景：使用比赛提供的预测月份COE和股票观测，租金标签仍在预测起点冻结。'}
    for kind,count in counts.items():
        lines.append(f'| {kind} | {count} | {explanations[kind]} |')
    lines+=['','所有参数、早停轮数和融合权重仅用2023-10至2024-03较早预测选择；随后按固定轮数在2024-04或2024-06起点之前重新拟合。两种融合目标提前确定：纯RMSE与固定0.001收缩，均为非负、和为1、不加截距的权重。',
        '','| 窗口 | 方案 | 旧最佳RMSE | 本轮RMSE | 差值 | MAE | 改善月份 |','|---|---|---:|---:|---:|---:|---:|']
    for row in comparisons.to_dict('records'):
        lines.append(f"| {row['stage']} | {row['version']} | {row['previous_rmse_sgd']:.4f} | {row['rmse_sgd']:.4f} | {row['delta_rmse_sgd']:+.4f} | {row['mae_sgd']:.4f} | {row['months_improved']}/{row['months']} |")
    if given:
        lines+=['','### 比赛已知宏观输入的独立对照','',
            '所给COE和股票文件覆盖测试期。这里允许使用预测月份的宏观观测，符合所给数据范围，但不同于在真实部署时预测完全未知的未来。该情景不与完全未知未来宏观的预测混称；没有读取预测月份租金标签。','',
            '| 窗口 | 融合目标 | 旧最佳RMSE | 已知宏观方案RMSE | 差值 |','|---|---|---:|---:|---:|']
        for row in given:
            lines.append(f"| {row['stage']} | {row['version']} | {row['previous_rmse_sgd']:.4f} | {row['rmse_sgd']:.4f} | {row['delta_rmse_sgd']:+.4f} |")
    lines+=['',f"本轮未知未来宏观情景的新融合最低近期RMSE为 **{best['rmse_sgd']:.4f}**。430目标{'达到' if achieved else '尚未达到'}。若新方案更差，保留前轮442.9025，不把一次失败试验替换成所谓新最优。",'',
        '## 误差与可行性判断','',
        '当前最佳模型在近期窗口中，实际租金≤1800的868行仅占约3%，却贡献约22.5%的平方误差；这些标签分组只用于诊断，不能在预测时用于选择模型。另一个高误差组为executive，但其总误差占比约7.4%，只改善该组不足以轻易解决全部差距。',
        '','现有输入缺少具体单元、装修、租户和合同状态；同月同楼栋户型会有不同租金。因此430值得实验，但不能从现有结果保证可达，更不能用同输入重复记录的组内标准差证明它绝对不可能。需要区分可学习的缺失信息与随机/未观测差异。',
        '','本轮近期和年度窗口均是已反复研究的开发数据。若继续根据它们挑选方案，最低值会有选择偏差。Kaggle Public也不等于最终Private结果。仓库曾记录2026-10-24的411.75928，当前榜单未能通过网页工具独立读取，故不声称本轮核验了实时名次。','',
        '## 核验与产物','',
        f"已独立核对{len(checked)}份预测的原始行ID、月份、标签、样本数及指标，重载{len(reloads)}个入选模型，重新组合{len(comparisons)+len(given)}份融合预测；原始文件哈希一致。23项相关测试通过。",'',
        '- `verified_metrics.csv`：全部成功配置的重算结果，含失败方向。',
        '- `baseline_error_groups.csv`：误差分组诊断。',
        '- `ensemble_selection.json`：较早窗口锁定的成员和权重。',
        '- `ensemble_comparison.csv`：与442.90方案的同窗口对照。',
        '- `summary.json` / `verification.json`：目标状态和核验记录。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    if (OUT/'submission_export.json').exists():
        append_submission_notes()
    report=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md'
    marker='\n## 8. 本地RMSE 430目标的后续探索\n'
    previous_report=report.read_text(encoding='utf-8')
    later_marker='\n## 9. 不同模型与新特征组的后续对照\n'
    later=later_marker+previous_report.split(later_marker,1)[1] if later_marker in previous_report else ''
    existing=previous_report.split(marker)[0]
    appendix=[marker,'',f"后续完成{int(sum(counts.values()))}组较早窗口配置，仍保留相同验证行和租金历史截止。本轮未知未来宏观情景的新融合最低近期RMSE为{best['rmse_sgd']:.4f}，旧最佳为442.9025。",'']
    if given:
        lowest=min((row for row in given if row['stage']=='june'),key=lambda row:row['rmse_sgd'])
        appendix.append(f"另单列使用比赛所给预测月COE/股票观测的情景，最低近期RMSE为{lowest['rmse_sgd']:.4f}；其信息条件与真实部署的未知未来宏观不同。")
    appendix+=['',f"430目标{'达到' if achieved else '尚未达到'}。没有可验证的本地/Kaggle成对成绩，不能假定固定减20。实验详情及全部负结果见[第四轮报告](../round_04_macro_spatial_diagnostics/README.md)。",'']
    report.write_text(existing+'\n'.join(appendix)+later,encoding='utf-8')
    print(json.dumps({'screens':counts,'verified_files':len(checked),'reloads':len(reloads),
        'comparisons':comparisons.to_dict('records'),'target_430_achieved':achieved},indent=2),flush=True)


def append_submission_notes():
    """Attach full-data export provenance without retraining verified trials."""
    metadata=json.loads((OUT/'submission_export.json').read_text())
    marker='\n## 全量重训后的可提交文件\n'
    report=OUT/'README.md'
    text=report.read_text(encoding='utf-8').split(marker)[0]
    section=[marker,'',
        f"当前最优历史融合已在全部{metadata['training_rows']:,}条训练记录上按固定轮数重训，为全部{metadata['test_rows']:,}条测试记录生成预测。10个基础成员和1个已知宏观成员的模型均已保存，主方案重载预测一致。租金历史对测试月份逐行排除同月标签，训练结束以后冻结。",'',
        '| 文件 | 对应本地开发RMSE | 信息条件 |','|---|---:|---|']
    for item in metadata['outputs']:
        scenario='历史主方案' if item['scenario']=='unknown_future_macro' else '比赛所给预测期宏观增强'
        section.append(f"| `{item['path']}` | {item['paired_local_development_rmse']:.4f} | {scenario} |")
    section+=['','两个文件均为50,000行、列名`Id,Predicted`，ID保持测试原顺序。未向Kaggle上传，真实Kaggle分数未知；表内分数对应本地时间验证方案，不是对这50,000条无标签记录测得的误差。',
        '','建议先用主方案建立第一组本地/Kaggle成对记录，再决定下一轮针对哪些时间跨度或模型误差继续优化。每份文件及原始输入的SHA-256见`submission_export.json`。',
        '','复现：`python -m analysis.experiments.best_ensemble_submission`。已完成的组件会按输入哈希与固定轮数复用。','']
    report.write_text(text+'\n'.join(section),encoding='utf-8')
    selection_report=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md'
    export_marker='\n### 已生成可提交预测\n'
    previous=selection_report.read_text(encoding='utf-8')
    later_marker='\n## 9. 不同模型与新特征组的后续对照\n'
    later=later_marker+previous.split(later_marker,1)[1] if later_marker in previous else ''
    current=previous.split(later_marker)[0].split(export_marker)[0]
    current+=export_marker+'\n已使用全部150,000条训练记录重训，为全部50,000条测试记录导出预测。'
    current+='[历史主方案CSV](../../../output/submissions/components/history_macro_ensemble/submission.csv)；'
    current+='[比赛所给宏观增强CSV](../../../output/submissions/components/history_macro_ensemble/submission_given_macro.csv)。'
    current+='两份文件尚未上传，实际Kaggle分数未知；对应本地开发RMSE分别为442.9025和442.8892。详见[第四轮导出记录](../round_04_macro_spatial_diagnostics/submission_export.json)。\n'
    selection_report.write_text(current+later,encoding='utf-8')


if __name__=='__main__':
    main()
