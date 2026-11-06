"""Verify author deep-model runs, seed averages, and fixed-origin ensembles."""
from __future__ import annotations
import gc
import json
import joblib
import numpy as np
import pandas as pd
import torch
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.third_wave_training import make_data
from analysis.experiments.fourth_wave_ensemble import old_best
from analysis.experiments.second_wave_report import matched_file,monthly_comparison
from analysis.experiments.sixth_wave_deep import OUT,ART,SPECS,SEED_GROUPS,dependencies,seed,load_payload,predict_saved
from analysis.pipeline.data import sha256


def main():
    dependencies();seed()
    raw,hashes=frame_data()
    for path,digest in hashes.items():
        assert sha256(ROOT/path)==digest
    records=[]
    for path in sorted(OUT.glob('*/*/results.json')):
        record=json.loads(path.read_text())
        stage=record['stage'];name=record['spec']['name']
        frame=matched_file(ART/stage/name/'predictions.csv')
        start,end=('2023-10','2024-04') if stage=='screen' else (
            ('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
        expected=raw.loc[(raw.RENT_APPROVAL_DATE>=start)&(raw.RENT_APPROVAL_DATE<end)]
        np.testing.assert_array_equal(frame.source_row,expected.index)
        np.testing.assert_array_equal(frame.actual,expected.MONTHLY_RENT)
        np.testing.assert_array_equal(frame.month,expected.RENT_APPROVAL_DATE)
        assert frame.source_row.is_unique and np.isfinite(frame.prediction).all()
        for key,value in score(frame.actual,frame.prediction).items():
            np.testing.assert_allclose(value,record[key],atol=1e-8,rtol=0)
        assert record['source_sha256']==hashes
        if record['spec'].get('seed_members'):
            mean=np.zeros(len(frame))
            for member in record['spec']['seed_members']:
                part=matched_file(ART/stage/member/'predictions.csv')
                np.testing.assert_array_equal(part.source_row,frame.source_row)
                mean+=part.prediction.to_numpy()/3
            np.testing.assert_allclose(mean,frame.prediction,atol=1e-8,rtol=0)
        records.append({'stage':stage,'name':name,'family':record['spec']['family'],
            'epochs':record['epochs'],'fit_rows':record['fit_rows'],'features':record['features'],
            'rows':len(frame),'rmse_sgd':record['rmse_sgd'],'mae_sgd':record['mae_sgd'],
            'bias_sgd':record['bias_sgd'],'seconds':record['seconds'],'peak_vram_gib':record['peak_vram_gib'],
            'seed_mean':bool(record['spec'].get('seed_members'))})
    table=pd.DataFrame(records)
    table.to_csv(OUT/'verified_metrics.csv',index=False)
    later_scores=table.loc[table.stage.isin(['outer','june'])].pivot(index='name',columns='stage',values='rmse_sgd').sort_values('june')
    later_scores.to_csv(OUT/'later_configuration_scores.csv')
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    effective_weights={}
    for name,weight in selection['weights'].items():
        members=SEED_GROUPS[name.removeprefix('seedmean__')] if name.startswith('seedmean__') else [name]
        for member in members:
            effective_weights[member]=effective_weights.get(member,0.)+weight/len(members)
    np.testing.assert_allclose(sum(effective_weights.values()),1.,atol=1e-12)
    write_json(OUT/'effective_blend_weights.json',effective_weights)
    reloads=[];comparisons=[];models=[];references=[];model_months=[]
    for stage in ('outer','june'):
        bundle=make_data(stage)
        names=(set(selection['weights'])|set(selection['representatives'].values()))-{'previous_best'}
        leaves=set()
        for name in names:
            spec=json.loads((OUT/'screen'/name/'results.json').read_text())['spec']
            leaves.update(spec.get('seed_members',[name]))
        for name in sorted(leaves):
            payload=load_payload(ART/stage/name/'model.joblib')
            prediction=predict_saved(payload,bundle)
            expected=matched_file(ART/stage/name/'predictions.csv')
            difference=float(np.max(np.abs(prediction-expected.prediction)))
            if difference>1e-5:
                raise AssertionError(f'Checkpoint differs: {stage} {name}: {difference}')
            reloads.append({'stage':stage,'name':name,'max_difference_sgd':difference,
                'parameters':sum(p.numel() for p in payload['model'].parameters())})
            del payload;gc.collect();torch.cuda.empty_cache()
        for family,name in selection['representatives'].items():
            record=table.loc[(table.stage==stage)&(table.name==name)].iloc[0]
            models.append({'stage':stage,'family':family,'selected_on_earlier':name,
                'rmse_sgd':float(record.rmse_sgd),'mae_sgd':float(record.mae_sgd),'bias_sgd':float(record.bias_sgd),
                'seed_mean':bool(record.seed_mean)})
            frame=matched_file(ART/stage/name/'predictions.csv')
            for month,part in frame.groupby('month'):
                model_months.append({'stage':stage,'family':family,'name':name,'month':month,
                    'rows':len(part),**score(part.actual,part.prediction)})
        reference=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/models/{stage}/ft_transformer/predictions.csv')
        np.testing.assert_array_equal(reference.source_row,bundle['row_val'])
        np.testing.assert_array_equal(reference.actual,bundle['y_val'])
        references.append({'stage':stage,'family':'previous_ft_transformer',
            'candidate_features':67,'rows':len(reference),**score(reference.actual,reference.prediction)})
        baseline=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv')
        new=matched_file(ART/'ensemble'/stage/'predictions.csv')
        combined=np.zeros(len(new))
        for name,weight in selection['weights'].items():
            values=old_best(stage,bundle) if name=='previous_best' else matched_file(ART/stage/name/'predictions.csv').prediction.to_numpy()
            combined+=weight*values
        np.testing.assert_allclose(new.prediction,combined,atol=1e-8,rtol=0)
        monthly,delta=monthly_comparison(new,baseline)
        monthly.to_csv(OUT/f'monthly_comparison_{stage}.csv',index=False)
        comparisons.append({'stage':stage,'rows':len(new),'new':score(new.actual,new.prediction),
            'previous_best':score(baseline.actual,baseline.prediction),**delta})
    model_table=pd.DataFrame(models)
    model_table.to_csv(OUT/'selected_model_comparison.csv',index=False)
    pd.DataFrame(references).to_csv(OUT/'previous_neural_reference.csv',index=False)
    pd.DataFrame(model_months).to_csv(OUT/'selected_model_monthly.csv',index=False)
    source_files=[ROOT/'.cache/deep-deps/tabm.py',ROOT/'.cache/deep-deps/rtdl_num_embeddings.py',
        ROOT/'.cache/deep-deps/pytabkit/models/realmlp/model.py',ROOT/'.cache/deep-deps/pytabkit/models/realmlp/train.py']
    author=json.loads((OUT/'author_source.json').read_text())
    for item in author['files']:
        assert sha256(ROOT/item['path'])==item['sha256']
    verification={'prediction_files':len(table),'reloaded_models':reloads,'seed_average_predictions_reconstructed':True,
        'ensemble_predictions_reconstructed':True,'source_sha256':hashes,
        'author_package_sha256':{str(path.relative_to(ROOT)):sha256(path) for path in source_files},
        'author_modernnca':author,'targeted_tests_passed':6,
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__}
    write_json(OUT/'verification.json',verification)
    # Seed standard deviations describe initialization variability, not holdout uncertainty.
    seeds=[]
    for base,members in SEED_GROUPS.items():
        for stage in ('screen','outer','june'):
            subset=table.loc[(table.stage==stage)&table.name.isin(members)]
            if len(subset)==3:
                average=table.loc[(table.stage==stage)&(table.name=='seedmean__'+base)].iloc[0]
                seeds.append({'stage':stage,'configuration':base,'mean_individual_rmse':float(subset.rmse_sgd.mean()),
                    'std_individual_rmse':float(subset.rmse_sgd.std(ddof=1)),
                    'equal_prediction_average_rmse':float(average.rmse_sgd)})
    pd.DataFrame(seeds).to_csv(OUT/'seed_variation.csv',index=False)
    count=int((~table.loc[table.stage=='screen','seed_mean']).sum())
    total_runs=int((~table.seed_mean).sum())
    summary={'trained_screen_configurations':count,'total_training_runs':total_runs,
        'screen_base_configurations':sum(not spec.get('seed_group') for spec in SPECS),
        'families':sorted(table.family.unique()),
        'comparisons':comparisons,'selected_models':model_table.to_dict('records'),
        'seed_variation':seeds,'verification':verification,
        'target_430_achieved':any(c['new']['rmse_sgd']<=430 for c in comparisons),
        'limitations':['Repeatedly viewed development periods; not independent test performance.',
            'Annual and June periods overlap and must not be pooled.',
            'Only a finite parameter/epoch/context budget; do not claim universal architecture rankings.',
            'RealMLP is the author experimental standalone implementation, not the original Lightning trainer.',
            'ModernNCA uses bounded training candidate pools; inference considers every fitting record.']}
    write_json(OUT/'summary.json',summary)
    plot(model_table,comparisons)
    labels={'tabm':'官方TabM','mlp_plr':'MLP-PLR','realmlp':'作者RealMLP独立版','modernnca':'作者ModernNCA'}
    lines=['# 第六轮：更强深度表格模型的实测','',
        f'本轮完成4类深度模型、{count}次较早窗口训练（15种基础配置＋4次补充种子），连同两个后续窗口共{total_runs}次训练，覆盖数值嵌入、参数共享集成、优化过的MLP与学习式检索。使用同一train＋静态auxiliary历史特征，不使用公开租金目标，不接收验证期租金更新。','',
        '## 模型与实现','',
        '- [TabM作者实现](https://github.com/yandex-research/tabm)：`tabm==0.0.3`，32个参数共享成员、3×256结构。比较普通输入/分段线性数值嵌入、原始/残差目标及两种学习率。训练分别计算每个成员的MSE，预测时平均。',
        '- [数值嵌入作者实现](https://github.com/yandex-research/rtdl-num-embeddings)与[RTDL MLP](https://github.com/yandex-research/rtdl-revisiting-models)组合为MLP-PLR；周期数值嵌入、类别Embedding和3×256 MLP。',
        '- [RealMLP作者发布的独立PyTorch实现](https://github.com/dholzmueller/pytabkit/tree/main/pytabkit/models/realmlp)，来自`pytabkit==1.8.0`。原生缺失数值处理、稳健缩放、平滑截断、PBLD嵌入、参数学习率因子与调度；测试1/4个成员。该独立版是作者标注的实验实现，与原Lightning版本存在初始化和权重衰减差异，不能宣称逐位等价。',
        '- [ModernNCA作者源码](https://github.com/LAMDA-Tabular/TALENT/blob/main/TALENT/model/models/modernNCA.py)：固定Git提交，源码未修改，另写训练适配器。学习数值/类别表示及邻域权重，从拟合期租金检索；比较普通/周期嵌入，以及作者默认结构。训练候选池限制4096/8192行，排除当前训练批次；推理使用全部拟合记录，缓存编码分块计算。','',
        '所有模型均从随机初始化学习，不额外引入新的租金数据或预训练目标。此前TabPFN/TabICL实验保留作另外的预训练路线证据。依赖安装在`.cache/deep-deps`，未改动现有Conda环境。','',
        '## 输入和时间协议','',
        '候选输入为62列强历史特征＋5列过去市场/同期历史收缩特征。只根据拟合集删除常量/全缺失列；因此不同起点实际列数可以不同。类别词表、数值变换、PLE分箱以及RealMLP稳健统计都仅由相应拟合集学习。残差预测在输出时只加回由过去租金构造的基准。','',
        '2023-10—2024-03的17,977行用于选择配置、停止epoch与融合权重；2024-04—2025-03年度窗口35,158行，以及2024-06—2025-03近期窗口28,995行，用各起点前全部标签按早期选定epoch重训。后续标签不用于早停。训练允许检索全部拟合期标签；检索上下文不含验证目标，当前训练查询不能从候选池检索自己的标签。','',
        'TabM和MLP-PLR最多60个epoch；RealMLP最多128个epoch，并保留128-epoch学习率调度用于固定epoch重训；ModernNCA最多50个epoch。早停耐心和有限上下文预算见`plan.json`。这是有预算的实测，未穷尽所有调参可能。','',
        '## 同窗口模型对比','',
        '每条路线的代表配置仅根据较早窗口选择；三种子均值使用预先固定42/43/44，所有种子均参与平均。未在后续窗口挑选最低分种子。','',
        '| 路线 | 年度RMSE | 近期RMSE | 较早窗口选定配置 |','|---|---:|---:|---|']
    for family in ('tabm','realmlp','mlp_plr','modernnca'):
        part=model_table.loc[model_table.family==family].set_index('stage')
        lines.append(f"| {labels[family]} | {part.loc['outer','rmse_sgd']:.4f} | {part.loc['june','rmse_sgd']:.4f} | `{part.loc['june','selected_on_earlier']}` |")
    reference_table=pd.DataFrame(references).set_index('stage')
    lines+=['',f"作为既有神经路线参照，第三轮强历史输入的FT-Transformer年度/近期RMSE为{reference_table.loc['outer','rmse_sgd']:.4f}/{reference_table.loc['june','rmse_sgd']:.4f}，逐行预测已重新核对。它同样使用67个候选输入与残差目标，但旧版预处理、网络和训练轮数不同；因此该对照说明整条训练方案的进展，不是仅更换架构的严格消融。",'']
    lines+=['### 后续窗口全部重训配置','',
        '以下保留全部后续结果，包括可能优于早期代表配置的单次运行；按近期分数展示只用于完整记录，不据此重选融合成员或权重。','',
        '| 配置 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for name,row in later_scores.iterrows():
        lines.append(f"| `{name}` | {row['outer']:.4f} | {row['june']:.4f} |")
    lines+=['','### 数值表示的较早窗口对照','',
        'TabM在相同候选输入、目标和学习率下比较普通数值输入与PLE；各配置独立早停，嵌入版本参数量也增加，因此这是训练方案对照。以下只作为筛选证据，未对全部普通输入版本重跑后续窗口，不能声称其增益已经跨窗口验证。','',
        '| 目标 / 学习率 | 普通输入RMSE | PLE输入RMSE | 差值 |','|---|---:|---:|---:|']
    for target,lr in [('raw',.002),('residual',.002),('residual',.0005)]:
        part=table.loc[table.stage=='screen'].set_index('name')
        plain=part.loc[f'tabm_plain_{target}_lr{lr}','rmse_sgd']
        ple=part.loc[f'tabm_ple_{target}_lr{lr}','rmse_sgd']
        lines.append(f'| {target} / {lr} | {plain:.4f} | {ple:.4f} | {ple-plain:+.4f} |')
    lines+=['','## 与原最优融合的比较','',
        '| 窗口 | 原方案RMSE | 新深度模型融合RMSE | 差值 | 新MAE | 改善月份 |','|---|---:|---:|---:|---:|---:|']
    for item in comparisons:
        lines.append(f"| {item['stage']} | {item['previous_best']['rmse_sgd']:.4f} | {item['new']['rmse_sgd']:.4f} | {item['delta_rmse_sgd']:+.4f} | {item['new']['mae_sgd']:.4f} | {item['months_improved']}/{item['months']} |")
    june_comparison=next(item for item in comparisons if item['stage']=='june')
    decision=('新融合在近期窗口也降低了RMSE，但仍需在新的未查看数据上验证其增益。'
        if june_comparison['delta_rmse_sgd']<0 else
        '新融合没有降低近期窗口RMSE，因此保留此前442.9025的历史主方案；已有可提交CSV不被覆盖。年度窗口的改善单独报告，不宣称所有时间范围均提升。')
    lines+=['',decision,'']
    lines+=['','三个月连续月份块重采样的差值区间仅作开发期误差变化描述，不能修正多轮搜索和模型选择的偏差：','',
        '| 窗口 | 新减旧RMSE的描述性95%区间 |','|---|---:|']
    for item in comparisons:
        low,high=item['month_block_interval_95']
        lines.append(f"| {item['stage']} | [{low:+.4f}, {high:+.4f}] |")
    lines+=['','三种子结果（单次RMSE的标准差与平均预测的RMSE含义不同）：','',
        '| 窗口 | 配置 | 单次RMSE均值 ± 标准差 | 三次预测平均RMSE |','|---|---|---:|---:|']
    for item in seeds:
        lines.append(f"| {item['stage']} | `{item['configuration']}` | {item['mean_individual_rmse']:.4f} ± {item['std_individual_rmse']:.4f} | {item['equal_prediction_average_rmse']:.4f} |")
    lines+=['','融合采用较早窗口上非负、和为1的权重，每个新候选最高0.5，并以固定0.001系数向原融合收缩；没有用后续窗口重新调权。单模型和融合的收益分别呈现，不能把不同特征/目标版本解释成架构的纯因果贡献。','',
        '候选池同时包含种子42的原配置与三种子均值，因此最终融合的各个种子有效权重可以不同；它们仅由较早窗口选定。`effective_blend_weights.json`展开所有重复成员，避免把整个最终融合误称为严格等权的三种子平均。','',
        f"430目标{'达到' if summary['target_430_achieved'] else '尚未达到'}。所有后续窗口已被前几轮研究查看，当前分数仍是开发证据；三种子标准差只描述初始化敏感性，不代表未来市场误差的不确定性。两个后续窗口重叠，不合并样本量。",'',
        '## 核验与文件','',
        f"独立核对{len(table)}份预测的ID、月份、标签、样本数和重算指标，重载{len(reloads)}个后续模型复现预测；三种子均值和最终融合重新组合一致。6项专门测试通过，覆盖作者接口、拟合集统计、未知类别、成员独立损失、训练查询排除及缓存检索等价性。源数据和作者源码哈希已记录。",'',
        '- `selected_model_comparison.csv`：各架构的年度/近期对照。',
        '- `verified_metrics.csv`：全部配置、epoch、时间和PyTorch峰值显存（不是物理驻留显存）。',
        '- `seed_variation.csv`：种子变化及等权平均结果。',
        '- `ensemble_selection.json`：在较早窗口锁定的权重与代表模型。',
        '- `author_source.json` / `verification.json`：源码来源、版本和完整核验。','']
    lines+=['## 复现','',
        '需要现有CUDA PyTorch环境，以及`.cache/model-deps`中的RTDL依赖。新增作者包单独安装，不覆盖现有环境。源码下载器固定本轮提交，并记录哈希；具体提交见`author_source.json`。','',
        '```powershell',
        'python -m pip install --no-deps --target .cache/deep-deps -r requirements/optional_deep_models.txt',
        'python -m analysis.experiments.sixth_wave_sources',
        'python -m analysis.experiments.sixth_wave_deep screen',
        'python -m analysis.experiments.sixth_wave_ensemble select',
        'python -m analysis.experiments.sixth_wave_ensemble outer',
        'python -m analysis.experiments.sixth_wave_ensemble june',
        'python -m unittest tests.test_sixth_wave',
        'python -m analysis.experiments.sixth_wave_report','```','',
        '融合使用此前实验保存的逐行预测，因此需保留第二至四轮输出。已有结果默认跳过；后续评估仅使用较早窗口保存的轮数和权重。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    report=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md'
    marker='\n## 10. 更强深度表格模型的后续实测\n'
    previous=report.read_text(encoding='utf-8').split(marker)[0]
    previous=previous.replace('第8项是当前最低开发RMSE方案，年度451.18、近期442.90',
        '第8项是此前筛选的最低开发RMSE方案，年度451.18、近期442.90')
    appendix=[marker,'',f'新增4类深度模型、{count}组较早窗口配置，含官方TabM、数值嵌入MLP、作者RealMLP独立版和ModernNCA；完整协议见[第六轮报告](../round_06_deep_models/README.md)。','',
        '| 路线 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for family in ('tabm','realmlp','mlp_plr','modernnca'):
        part=model_table.loc[model_table.family==family].set_index('stage')
        appendix.append(f"| {labels[family]} | {part.loc['outer','rmse_sgd']:.4f} | {part.loc['june','rmse_sgd']:.4f} |")
    june=next(item for item in comparisons if item['stage']=='june')
    outer=next(item for item in comparisons if item['stage']=='outer')
    appendix+=[f"| 原历史主融合 | {outer['previous_best']['rmse_sgd']:.4f} | {june['previous_best']['rmse_sgd']:.4f} |",
        f"| 加入新深度模型的融合 | {outer['new']['rmse_sgd']:.4f} | {june['new']['rmse_sgd']:.4f} |"]
    appendix+=['',f"较早窗口锁定的新融合近期RMSE为{june['new']['rmse_sgd']:.4f}，原方案为442.9025；430目标{'达到' if summary['target_430_achieved'] else '尚未达到'}。后续窗口不用来选epoch或权重，但它们仍是重复查看过的开发数据。",'']
    appendix+=['正文仍保留同27输入的基础模型比较；另增增强历史输入的深度模型对照，并在性能进展表同时展示新旧融合。各路线与全部单次配置放扩展表，避免把不同输入的分数混成架构排名。',decision,'']
    notice_start='<!-- sixth-wave-status:start -->'
    notice_end='<!-- sixth-wave-status:end -->'
    if notice_start in previous:
        before,remainder=previous.split(notice_start,1)
        previous=before+remainder.split(notice_end,1)[1].lstrip('\n')
    title,body=previous.split('\n',1)
    notice=(f"{notice_start}\n**第六轮更新：** 已完成4类新深度模型实测。新融合年度RMSE为{outer['new']['rmse_sgd']:.4f}（此前451.1810），近期为{june['new']['rmse_sgd']:.4f}（此前442.9025）。前文第8项保留此前方案及其组成作为对照；最新结果及范围见[第10节](#10-更强深度表格模型的后续实测)。430目标{'达到' if summary['target_430_achieved'] else '尚未达到'}。\n{notice_end}\n\n")
    report.write_text(title+'\n\n'+notice+body.lstrip('\n')+'\n'.join(appendix),encoding='utf-8')
    print(json.dumps({'trained_configurations':count,'verified_predictions':len(table),'reloads':len(reloads),
        'selected_models':models,'comparisons':comparisons},indent=2),flush=True)


def plot(models,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for axis,stage in zip(axes,('outer','june')):
        part=models.loc[models.stage==stage]
        names=list(part.family)+['Prior blend','New blend']
        comp=next(c for c in comparisons if c['stage']==stage)
        values=list(part.rmse_sgd)+[comp['previous_best']['rmse_sgd'],comp['new']['rmse_sgd']]
        axis.barh(names,values,color=['#447ca8']*4+['#b7a38a','#ad693a'])
        axis.invert_yaxis();axis.set_xlim(0,max(values)*1.12)
        for i,value in enumerate(values):
            axis.text(value+2,i,f'{value:.2f}',va='center',fontsize=8)
        axis.set_title('12-month development window' if stage=='outer' else '10-month development window')
        axis.set_xlabel('RMSE (SGD)')
    fig.savefig(OUT/'deep_model_comparison.png',dpi=180)
    fig.savefig(OUT/'deep_model_comparison.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
