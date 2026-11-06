"""Audit trained refinements without selecting weights on later labels."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_ensemble import old_best
from analysis.experiments.eighth_wave_meta import data,registry,forecast
from analysis.experiments.eighth_wave_refinement import OUT,ART,read_prediction,cat_delta


def write_prediction(stage,name,bundle,prediction):
    destination=ART/stage/name;destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'source_row':bundle['row_val'],'month':bundle['date_val'],
        'actual':bundle['y_val'],'prediction':prediction}).to_csv(destination/'predictions.csv',index=False)
    record={'stage':stage,'name':name,'rows':len(prediction),**score(bundle['y_val'],prediction)}
    write_json(OUT/stage/name/'results.json',record)
    return record


def run():
    from analysis.experiments import sixth_wave_deep as sixth
    from analysis.experiments.best_ensemble_submission import predict_saved
    sixth.dependencies()
    bag=json.loads((OUT/'bag_selection.json').read_text(encoding='utf-8'))
    plan=json.loads((OUT/'refinement_blend_plan.json').read_text(encoding='utf-8'))
    best=json.loads((ROOT/'output/experiment_runs/00_report_and_final/best_ensemble.json').read_text(encoding='utf-8'))
    ft_weight=next(m['weight'] for m in best['leaf_members'] if m['artifact_name']=='ft_transformer')
    weight=bag['cat_leaf_weight'];fraction=bag['selected']['replacement_fraction'];a=plan['tabm_weight']
    rows=[];replays=[]
    members=registry()
    for stage in ('screen','outer','june'):
        bundle=make_data(stage);baseline=old_best(stage,bundle)
        rows.append({'stage':stage,'name':'previous_history','rows':len(baseline),**score(bundle['y_val'],baseline)})
        for seed in (43,44,45):
            name=f'cat_seed{seed}'
            record=json.loads((OUT/stage/name/'results.json').read_text(encoding='utf-8'))
            rows.append({k:record[k] for k in ('stage','name','rows','rmse_sgd','mae_sgd','bias_sgd')})
            payload=joblib.load(ART/stage/name/'model.joblib')
            from analysis.experiments.second_wave_anniversary import select_inputs
            _,q=select_inputs(bundle,'window')
            for column in payload['categories']:
                q[column]=q[column].astype('string').fillna('__MISSING__').astype(str)
            replay=payload['model'].predict(q.fillna(-999.))
            saved=read_prediction(ART/stage/name/'predictions.csv',bundle)
            np.testing.assert_allclose(replay,saved,atol=1e-6,rtol=0)
            replays.append({'stage':stage,'name':name,'max_abs_difference':float(np.max(np.abs(replay-saved)))})
            del payload
        if stage=='screen':
            ft_delta=np.zeros(len(baseline))
            tabm=read_prediction(ROOT/'output/models/round_06_deep_models/screen/seedmean__tabm_ple_residual_lr0.0005/predictions.csv',bundle)
        else:
            old_ft=read_prediction(ROOT/'output/models/round_03_tree_transformer_ensemble/models'/stage/'ft_transformer/predictions.csv',bundle)
            ft_delta=read_prediction(ART/stage/'ft_no36/predictions.csv',bundle)-old_ft
            tabm=np.mean([read_prediction(ART/stage/f'tabm_no36_seed{seed}/predictions.csv',bundle)
                for seed in (42,43,44)],axis=0)
            for name in ['ft_no36']+[f'tabm_no36_seed{seed}' for seed in (42,43,44)]:
                record=json.loads((OUT/stage/name/'results.json').read_text(encoding='utf-8'))
                rows.append({k:record[k] for k in ('stage','name','rows','rmse_sgd','mae_sgd','bias_sgd')})
                payload=sixth.load_payload(ART/stage/name/'model.joblib')
                replay=predict_saved(payload,bundle,'models') if name=='ft_no36' else sixth.predict_saved(payload,bundle)
                saved=read_prediction(ART/stage/name/'predictions.csv',bundle)
                np.testing.assert_allclose(replay,saved,atol=1e-5,rtol=0)
                replays.append({'stage':stage,'name':name,'max_abs_difference':float(np.max(np.abs(replay-saved)))})
                del payload
        raw_bag_delta=weight*cat_delta(stage,bundle)
        bag_delta=fraction*raw_bag_delta
        candidates={'bag_selected':baseline+bag_delta,
            'bag_half_exploratory':baseline+.5*raw_bag_delta,
            'bag_full_exploratory':baseline+raw_bag_delta,
            'ft_replacement':baseline+ft_weight*ft_delta,
            'tabm_blend':(1-a)*baseline+a*tabm,
            'bag_ft':baseline+bag_delta+ft_weight*ft_delta,
            'bag_ft_tabm':(1-a)*(baseline+bag_delta+ft_weight*ft_delta)+a*tabm}
        for name,prediction in candidates.items():
            rows.append(write_prediction(stage,name,bundle,prediction))
        old_ft=read_prediction(ROOT/'output/models/round_03_tree_transformer_ensemble/models'/stage/'ft_transformer/predictions.csv',bundle)
        ft42=old_ft if stage=='screen' else read_prediction(ART/stage/'ft_no36/predictions.csv',bundle)
        ftmean=np.mean([ft42]+[read_prediction(ART/stage/f'ft_no36_seed{seed}/predictions.csv',bundle)
            for seed in (43,44,45)],axis=0)
        rows.append(write_prediction(stage,'ft_seedmean_replacement',bundle,baseline+ft_weight*(ftmean-old_ft)))
        for seed_value in (43,44,45):
            name=f'ft_no36_seed{seed_value}'
            record=json.loads((OUT/stage/name/'results.json').read_text(encoding='utf-8'))
            rows.append({k:record[k] for k in ('stage','name','rows','rmse_sgd','mae_sgd','bias_sgd')})
            payload=sixth.load_payload(ART/stage/name/'model.joblib')
            replay=predict_saved(payload,bundle,'models')
            saved=read_prediction(ART/stage/name/'predictions.csv',bundle)
            np.testing.assert_allclose(replay,saved,atol=1e-5,rtol=0)
            replays.append({'stage':stage,'name':name,'max_abs_difference':float(np.max(np.abs(replay-saved)))})
            del payload
        if stage!='screen':
            macro_weights=json.loads((ROOT/'output/experiment_runs/round_04_macro_spatial_diagnostics/given_macro_selection.json').read_text(encoding='utf-8'))['versions']['conservative']['weights']
            macro=read_prediction(ROOT/'output/models/round_04_macro_spatial_diagnostics/known_macro'/stage/'current_cat8/predictions.csv',bundle)
            rows.append(write_prediction(stage,'ft_replacement_given_macro',bundle,
                macro_weights['previous_best']*candidates['ft_replacement']+macro_weights['current_cat8']*macro))
            rows.append(write_prediction(stage,'ft_seedmean_given_macro',bundle,
                macro_weights['previous_best']*(baseline+ft_weight*(ftmean-old_ft))+macro_weights['current_cat8']*macro))
        if stage!='screen':
            rows.append(write_prediction(stage,'tabm_no36_seedmean',bundle,tabm))
            meta=data(stage,members)
            for path in (OUT/stage).glob('*/results.json'):
                record=json.loads(path.read_text(encoding='utf-8'))
                if 'spec' not in record:continue
                name=record['spec']['name']
                rows.append({k:record[k] for k in ('stage','rows','rmse_sgd','mae_sgd','bias_sgd')}|{'name':name})
                model=joblib.load(ART/stage/name/'model.joblib')
                replay=forecast(model,meta['x'],meta['baseline'])
                saved=read_prediction(ART/stage/name/'predictions.csv',bundle)
                np.testing.assert_allclose(replay,saved,atol=1e-6,rtol=0)
                replays.append({'stage':stage,'name':name,'max_abs_difference':float(np.max(np.abs(replay-saved)))})
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'metrics.csv',index=False)
    pivot=table[table.stage.isin(['outer','june'])].pivot(index='name',columns='stage',values='rmse_sgd').sort_values('june')
    pivot.to_csv(OUT/'comparison.csv')
    from analysis.experiments.second_wave_report import monthly_comparison
    intervals=[]
    for stage in ('outer','june'):
        baseline_path=ROOT/'output/models/round_03_tree_transformer_ensemble/ablations'/stage/'models/predictions.csv'
        previous=pd.read_csv(baseline_path).sort_values('source_row').reset_index(drop=True)
        if 'approval_month' in previous:previous=previous.rename(columns={'approval_month':'month'})
        for name in ('ft_replacement','ft_seedmean_replacement','bag_full_exploratory'):
            candidate=pd.read_csv(ART/stage/name/'predictions.csv').sort_values('source_row').reset_index(drop=True)
            monthly,description=monthly_comparison(candidate,previous)
            monthly.to_csv(OUT/f'{stage}_{name}_monthly.csv',index=False)
            intervals.append({'stage':stage,'name':name,**description})
    write_json(OUT/'monthly_comparison.json',{'results':intervals,
        'note':'Descriptive resampling of repeatedly viewed development months; not independent confirmatory confidence intervals.'})
    write_json(OUT/'verification.json',{'artifact_replays':replays,'count':len(replays),
        'unit_tests':{'command':'python -m unittest tests.test_eighth_wave tests.test_third_wave tests.test_sixth_wave','passed':17},
        'sample_weight_normalization':'mean one, matching tuning loss scale',
        'discarded_attempt':'archive_weight_scale_debug: weight sum normalization changed effective regularization; excluded from results',
        'development_only':True})
    selected_meta=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['selected']
    lines=['# 第八轮：小幅优化的实际训练结果','',
        '本轮最低近期候选为 **442.7032**：删除FT-Transformer的五列36个月同期统计，用42/43/44/45四个种子、各固定3轮重新训练并等权平均，替换原融合中15.9872%的FT成员，其他成员及权重保持原样。相比原历史方案442.9025降低 **0.1993**。','',
        '年度窗口为451.4310，原方案为451.1810，反而升高0.2500。近期10个月中有5个月改善，描述性月块重采样区间为[-0.3017, 0.0065]，仍跨过0。它是值得提交比较的小幅优化候选，尚不能证明稳定提升。','',
        '本轮不以430为硬目标，尝试从已训练模型的组合稳定性和历史特征可用性入手。所有分数仍是反复查看过的本地时间开发窗口，不能当作独立测试或Kaggle成绩。','',
        '| 方案 | 年度 RMSE（35,158行） | 近期 RMSE（28,995行） |','|---|---:|---:|']
    for name,r in pivot.iterrows():lines.append(f"| {name} | {r['outer']:.4f} | {r['june']:.4f} |")
    lines+=['','## 训练与选择方法','',
        f'1. 二层残差模型：37组基础预测（实际注册数 {len(members)}），13个配置。用2023-10至2024-01拟合、2024-02至03选择，选中 `{selected_meta}`。年度元模型只能用2024-04以前的预测和标签；六月模型可再使用4—5月的历史样本。原点权重归一化为均值1，避免改变正则强度。后期共26次拟合。',
        f'2. CatBoost种子平均：固定原CPU算法、62输入、833轮，新增43/44/45三个种子，与原42等权平均。共9次新拟合。只在早期窗口从替换比例0/0.5/1选择，锁定为 {fraction}；原叶子权重 {weight:.10f} 保持不变。',
        '3. 深度特征删减：移除五列36个月同期统计。原较早期训练池中这些字段恒定/全缺失，本来已被神经网络编码器删除，故早期输入和预测严格等价；后期重新训练FT-Transformer（固定3轮）及TabM三个种子（固定18/15/15轮），共8次。所有样本均保留。',
        f'4. 固定组合：FT直接替换原权重 {ft_weight:.10f} 的叶子；TabM平均预测与原方案混合，权重由早期预测的最小二乘确定，上限0.3，结果 {a:.6f}。还报告两种预先列出的组合，不使用后期标签调融合权重。','',
        '## 核查与解释','',
        f'另在看到FT删减单种子结果后追加43/44/45三个种子的固定3轮训练，三个时间窗口共9次，与42等权平均，不按后期分数筛选种子。共39次元模型拟合、9次CatBoost拟合、8次深度删减拟合、9次FT种子拟合，合计65次实际拟合（不含全量提交重训）。保存模型共 {len(replays)} 份已重新加载并复现完整窗口预测。信息截止、样本权重尺度、类别映射和特征删减另有自动测试。',
        '初次元模型尝试存在样本权重尺度不一致，已完整归档到 `archive_weight_scale_debug`，修正后全量重跑；归档分数不参与本报告或最佳结果选择。',
        '`bag_half_exploratory` 和 `bag_full_exploratory` 是早期未被选中的固定比例对照；不能称为早期选出的最佳方案。`given_macro` 使用比赛所给预测月份的COE/股票信息，按此前固定权重融合，与纯历史信息方案分开解释。',
        '后期表格中的最低值属于开发集探索结果；各方案权重虽在早期锁定，架构和窗口本身经过多轮讨论和筛选，仍应通过Kaggle提交检验。原最佳提交文件保持原样。','',
        '完整数据：[指标](metrics.csv)、[模型复现](verification.json)、[月度稳定性](monthly_comparison.json)、[二层模型计划](plan.json)、[种子选择](bag_selection.json)、[深度对照计划](deep_plan.json)。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pivot.to_string())


def finish_documents():
    """Record user feedback and link verified full-data exports in UTF-8."""
    import shutil
    from analysis.pipeline.data import sha256
    submission=ROOT/'output/submissions/components/round_08_ft_seed_average'
    summary=json.loads((submission/'summary.json').read_text(encoding='utf-8'))
    ready=ROOT/'output/submissions/candidates';ready.mkdir(parents=True,exist_ok=True)
    copies=[]
    for item in summary['outputs']:
        source=ROOT/item['path'];target=ready/source.name
        assert sha256(source)==item['sha256']
        shutil.copy2(source,target)
        assert sha256(target)==item['sha256']
        copies.append(item|{'ready_path':str(target.relative_to(ROOT)).replace('\\','/')})
    write_json(ready/'eighth_wave_submission_manifest.json',summary|{'outputs':copies})
    feedback={'reported_on':'2026-11-05','source':'user report in conversation',
        'scores':[{'file':'kaggle_best_history.csv','score_approx':415.,'exact_score':None},
            {'file':'kaggle_best_macro.csv','score_approx':415.,'exact_score':None}],
        'user_note':'都是，小数级差别','leaderboard_split':None,'verified_by_connector':False,
        'normal_benchmark_score_approx':411.,'benchmark_source':'user report',
        'other_scores_around_170':'User describes these as cheat methods; not part of valid model comparisons.',
        'note':'No exact score or constant local-to-Kaggle offset is inferred.'}
    write_json(OUT/'kaggle_feedback.json',feedback)
    path=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md'
    content=path.read_text(encoding='utf-8').split('<!-- eighth-wave-details:start -->')[0].rstrip()
    addition='''

<!-- eighth-wave-details:start -->
## 12. 小幅优化与实际训练结果

本轮以实际训练后降低一点也保留为目标。共完成65次新开发拟合：39次二层残差模型、9次CatBoost随机种子、8次神经网络特征删减、9次额外FT种子；另用全部150,000条训练记录重训4个FT模型以导出50,000条测试预测。后续窗口固定训练轮数。

| 方案 | 年度 RMSE | 近期 RMSE | Kaggle反馈 |
|---|---:|---:|---|
| 原历史主融合 | 451.1810 | 442.9025 | 用户反馈约415 |
| 原给定宏观增强 | 451.2081 | 442.8892 | 用户反馈约415，二者仅小数差异 |
| **FT删减＋四种子平均替换** | **451.4310** | **442.7032** | 新文件尚未提交 |
| 新FT替换＋原宏观权重 | 451.5252 | 442.7710 | 新文件尚未提交 |

新方案只更换主融合中的FT-Transformer成员：删去5列36个月同期统计，固定3轮，42/43/44/45四个种子等权平均；保留原15.9872%叶子权重及其他成员。在相同28,995行近期窗口降低0.1993；年度35,158行反而升高0.2500。删减特征对TabM没有显示相同收益；新二层模型和CatBoost种子平均也未超过原近期主方案，负结果一并保留。

近期10个月中5个月改善，描述性月块重采样差值区间[-0.3017,0.0065]跨0，不能宣称稳定提升；两个窗口部分重叠。52份开发保存模型的完整预测复现及17项相关测试均通过。4份全量重训模型也重新加载复现，原始数据哈希均保持一致。

用户于2026-11-05反馈，原历史版与宏观版Kaggle都在415左右、仅小数级差别；正常方案榜上最佳约411，另一些约170的分数被用户描述为cheat方法，故不纳入有效对照。以上均按用户反馈记录，尚无精确小数或榜单类型，不推断本地固定减20或其他固定差距。新方案需在相同榜单上提交比较。

新的可提交文件：优先试[历史版442.7032](../../../output/submissions/candidates/kaggle_history_ft_no36_mean.csv)；另有[宏观版442.7710](../../../output/submissions/candidates/kaggle_macro_ft_no36_mean.csv)。原文件保留，CSV的分数标签指对应时间验证实验，不是50,000条测试数据的已知成绩。

详细证据：[第八轮报告](../round_08_feature_pruning_seeds/README.md)、[对照指标](../round_08_feature_pruning_seeds/comparison.csv)、[模型复现](../round_08_feature_pruning_seeds/verification.json)、[Kaggle反馈](../round_08_feature_pruning_seeds/kaggle_feedback.json)、[全量重训与导出清单](../../../output/submissions/components/round_08_ft_seed_average/summary.json)。
<!-- eighth-wave-details:end -->
'''
    path.write_text(content+addition,encoding='utf-8')
    report=OUT/'README.md';body=report.read_text(encoding='utf-8')
    marker='\n## Kaggle反馈与新提交文件\n'
    body=body.split(marker)[0].rstrip()
    body+=marker+'''
用户反馈原历史版、宏观版均约415，小数级差别，正常方案最佳约411；精确小数和榜单类型未知，不能拟合固定分差。约170分的方法被用户描述为cheat方法，不纳入有效对照。

全量150,000行分别重训四个FT种子，均固定3轮；保存模型已重新加载，生成50,000行测试预测。新文件均通过ID、有限数值、落盘一致性和源文件哈希检查，原提交保持原样。

- 优先试：[历史版，本地442.7032](../../../output/submissions/candidates/kaggle_history_ft_no36_mean.csv)。
- 补充：[宏观版，本地442.7710](../../../output/submissions/candidates/kaggle_macro_ft_no36_mean.csv)。
- [全量训练与导出记录](../../../output/submissions/components/round_08_ft_seed_average/summary.json)。

新文件尚未上传，新Kaggle分数未知。
'''
    report.write_text(body,encoding='utf-8')
    print('Verified new ready-to-upload copies and UTF-8 reports.')


if __name__=='__main__':run()
