"""Audit and describe local inference from recent author Transformer models."""
from __future__ import annotations
import json
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.second_wave_report import matched_file,monthly_comparison
from analysis.experiments.seventh_wave_transformers import OUT,ART,SPECS,enable
from analysis.pipeline.data import sha256


def main():
    enable();raw,hashes=frame_data();records=[];adaptation_rows=[]
    manifest=json.loads((OUT/'checkpoint_manifest.json').read_text())
    for item in manifest.values():
        assert sha256(ROOT/item['path'])==item['sha256']==item['author_lfs_sha256']
    tabicl_manifest=json.loads((ROOT/'.cache/foundation-models/tabicl_manifest.json').read_text())
    assert sha256(ROOT/'.cache/foundation-models/tabicl-regressor-v2-20260212.ckpt')==tabicl_manifest['sha256']
    for path in sorted(OUT.glob('*/*/results.json')):
        record=json.loads(path.read_text())
        if record['smoke']:
            continue
        stage=record['stage'];name=record['spec']['name'];frame=matched_file(ART/stage/name/'predictions.csv')
        start,end=('2023-10','2024-04') if stage=='screen' else (('2024-04','2025-04') if stage=='outer' else ('2024-06','2025-04'))
        expected=raw.loc[(raw.RENT_APPROVAL_DATE>=start)&(raw.RENT_APPROVAL_DATE<end)]
        np.testing.assert_array_equal(frame.source_row,expected.index)
        np.testing.assert_array_equal(frame.actual,expected.MONTHLY_RENT)
        np.testing.assert_array_equal(frame.month,expected.RENT_APPROVAL_DATE)
        assert frame.source_row.is_unique and np.isfinite(frame.prediction).all()
        assert record['source_sha256']==hashes
        recipe=joblib.load(ART/stage/name/'recipe.joblib')
        assert recipe['spec']==record['spec'] and len(recipe['columns'])==record['features']
        assert recipe['checkpoint_sha256']==record['checkpoint_sha256']
        assert len(np.unique(recipe['context_indices']))==record['context_available_rows']
        assert np.min(recipe['context_indices'])>=0 and np.max(recipe['context_indices'])<record['source_rows']
        family=record['spec']['family']
        expected_checkpoint=tabicl_manifest['sha256'] if family=='tabicl2' else manifest['tabdpt13' if family.startswith('tabdpt13') else family]['sha256']
        assert expected_checkpoint==record['checkpoint_sha256']
        for key,value in score(frame.actual,frame.prediction).items():
            np.testing.assert_allclose(value,record[key],atol=1e-8,rtol=0)
        if record['spec'].get('adapt'):
            assert sha256(ART/stage/name/'adapted_state.pt')==record['adapted_state_sha256']
            adaptation_rows.append({'stage':stage,'name':name,'epochs':record['training_epochs'],
                'matched_frozen_rmse':record['matching_frozen_control']['rmse_sgd'],
                'adapted_rmse':record['rmse_sgd'],'trainable_parameters':record['trainable_parameters'],
                'supervised_rows_per_epoch':record['supervised_rows_per_epoch']})
        if record['spec']['family'].startswith('nori'):
            reports=record['memory_report']
            reports=reports if isinstance(reports,list) else [reports]
            for report in reports:
                if report is not None:
                    assert not report.get('dropped_context_rows',0)
        records.append({'stage':stage,'name':name,'family':record['spec']['family'],
            'target':record['spec']['target'],'features':record['features'],'rows':len(frame),
            'training_epochs':record.get('training_epochs',0),'trainable_parameters':record.get('trainable_parameters',0),
            'context_available_rows':record['context_available_rows'],
            'context_rows_per_view':record['context_rows_per_view'],'members':record['spec']['members'],
            **score(frame.actual,frame.prediction),'seconds':record['seconds'],'peak_vram_gib':record['peak_vram_gib']})
    table=pd.DataFrame(records);table.to_csv(OUT/'verified_metrics.csv',index=False)
    pd.DataFrame(adaptation_rows).to_csv(OUT/'adaptation_comparison.csv',index=False)
    selection=json.loads((OUT/'ensemble_selection.json').read_text())
    comparison=[];models=[]
    for stage in ('outer','june'):
        new=matched_file(ART/'ensemble'/stage/'predictions.csv')
        old=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/ablations/{stage}/models/predictions.csv')
        sixth=matched_file(ROOT/f'output/models/round_06_deep_models/ensemble/{stage}/predictions.csv')
        ft=matched_file(ROOT/f'output/models/round_03_tree_transformer_ensemble/models/{stage}/ft_transformer/predictions.csv')
        np.testing.assert_array_equal(new.source_row,old.source_row)
        np.testing.assert_array_equal(new.actual,old.actual)
        np.testing.assert_array_equal(new.source_row,ft.source_row)
        np.testing.assert_array_equal(new.actual,ft.actual)
        combined=np.zeros(len(new))
        for name,weight in selection['weights'].items():
            part=old if name=='previous_history' else sixth if name=='previous_deep' else matched_file(ART/stage/name/'predictions.csv')
            np.testing.assert_array_equal(new.source_row,part.source_row)
            combined+=weight*part.prediction.to_numpy(float)
        np.testing.assert_allclose(combined,new.prediction,atol=1e-8,rtol=0)
        monthly,delta=monthly_comparison(new,old);monthly.to_csv(OUT/f'monthly_comparison_{stage}.csv',index=False)
        comparison.append({'stage':stage,'new':score(new.actual,new.prediction),
            'previous_history':score(old.actual,old.prediction),'previous_deep':score(sixth.actual,sixth.prediction),
            'previous_ft':score(ft.actual,ft.prediction),**delta})
        for family,name in selection['representatives'].items():
            row=table.loc[(table.stage==stage)&(table.name==name)].iloc[0].to_dict()
            row['route']=family;models.append(row)
    selected=pd.DataFrame(models);selected.to_csv(OUT/'selected_model_comparison.csv',index=False)
    from analysis.experiments.seventh_wave_adaptation import replay_saved
    replays=[replay_saved(stage) for stage in ('outer','june')]
    diagnostics=json.loads((OUT/'diagnostics.json').read_text())
    verification={'prediction_files':len(table),'all_validation_rows_retained':True,
        'source_sha256':hashes,'checkpoints_verified_against_author_lfs':manifest,
        'existing_tabicl_checkpoint_verified':tabicl_manifest,'adapted_full_window_replays':replays,
        'ensembles_reconstructed':True,'targeted_tests_passed':7,
        'author_package_versions':{name:importlib.metadata.version(name) for name in ['tabdpt','synthefy-nori','synthefy','tabicl','torch','faiss-cpu']},
        'adapter_sha256':{name:sha256(ROOT/'analysis/experiments'/name) for name in ['seventh_wave_sources.py','seventh_wave_transformers.py','seventh_wave_ensemble.py','seventh_wave_adaptation.py']},
        'replay_scope':'Both learned TabDPT adapters reloaded for full later windows. Frozen author TabDPT/Nori APIs passed smoke checks; their final CSVs and recipes audited without claiming every frozen configuration was reloaded for full windows.'}
    write_json(OUT/'verification.json',verification)
    summary={'screen_configurations':int((table.stage=='screen').sum()),'inference_runs':len(table),
        'selected_models':models,'comparisons':comparison,'verification':verification,
        'target_430_achieved':any(item['new']['rmse_sgd']<=430 for item in comparison),
        'access_limited':['User chose to continue public-weight models without activating TabPFN-3.5 login/model-license access.'],
        'limitations':['Repeatedly viewed, overlapping development windows; not independent test/Kaggle scores.',
            'Finite context/view/VRAM budgets; native preprocessing differs by model.',
            'No claim that pretraining data of TabDPT is synthetic-only; authors describe real-data pretraining.']}
    write_json(OUT/'summary.json',summary)
    plot(selected,comparison)
    labels={'tabdpt13':'TabDPT 1.3 冻结','tabdpt13_adapt':'TabDPT 1.3 监督适配','nori':'Nori','tabicl2':'TabICLv2'}
    lines=['# 第七轮：更新的 Transformer 表格模型','',
        f"本轮完成{summary['screen_configurations']}组完整较早窗口配置，连同后续窗口共{len(table)}次完整评估。包含14组冻结上下文推理及1组TabDPT监督适配的早期配置；不把冻结推理次数称为从头训练次数。",'',
        '## 路线与版本','',
        '- [TabDPT 1.3 / Turbo后续版](https://github.com/layer6ai-labs/TabDPT-inference)：作者2026-09发布的v1.3权重，`tabdpt==1.3.2`。预训练Transformer使用源记录租金作为上下文，比较全池随机抽样与逐条最近邻检索。作者说明其预训练使用真实世界表格数据；本项目未额外下载HDB租金标签。',
        '- TabDPT监督适配：另写本项目训练器，更新作者骨干最后两个Transformer块、对应标签编码器与回归头，冻结其余参数及注意力缩放kappa；不是作者原始训练配方。每个epoch将全部拟合记录各作为一次查询，查询自己的标签不进入512条训练上下文。BF16、MSE、AdamW学习率0.00003，最多8轮，早停耐心3轮；包含0轮原权重的匹配对照。后续只按早期固定轮数重训，不用后续标签早停。',
        '- [Nori 1.5](https://github.com/Synthefy/synthefy-nori)：`synthefy-nori==0.22.0`，约20.8M参数，分两阶段编码、特征注意力与样本注意力；另测较小的Nori-6M作为规模对照。作者说明权重由合成数据预训练。',
        '- [TabICLv2](https://github.com/soda-inria/tabicl)：已有2026-02回归权重、`tabicl==2.1.1`；本轮换为62＋5个候选强历史输入及新的历史基准，区别于第二轮较少特征的旧配置。',
        '- [TabPFN-3.5](https://github.com/PriorLabs/TabPFN)：作者访问方式要求用户登录并接受模型许可。用户选择先继续公开权重模型，因此本轮不测试该版本。此前TabPFN结果仍按各自旧版本与输入记录。','',
        '所有预测在本地运行；下载作者权重，未向模型服务发送项目数据。原始租金、相对过去基准的残差，以及原始租金/过去基准的比率，分别独立筛选；后两者在输出时加回/乘回源历史计算的基准。比率分母不使用验证月的实际平均租金。','',
        '## 时间、特征与硬件预算','',
        '较早窗口2023-10—2024-03共17,977条；年度2024-04—2025-03共35,158条；近期2024-06—2025-03共28,995条。每个起点的租金历史冻结，保留每一条验证记录。后续配置、上下文大小、视图数和融合权重均在较早窗口锁定。三个窗口都属于反复查看过的开发数据，后两者重叠。','',
        '候选输入为62列历史/静态特征＋5列过去市场与同期收缩表示，仅按相应拟合上下文删除全缺失/常量列。TabDPT类别用源池词表编码，源池拟合填补与缩放；Nori使用原生类别和数值处理，并禁用HighDimFeatureSelector的fit_on_test。TabICL采用原生类别/数值流水线。','',
        '另对三条路线各加入一组历史成熟上下文对照：仅从时间已满24个月、且24个月同期楼栋历史计数大于0的源记录中选择上下文。筛选不使用租金数值，也不减少验证记录；用于检验冷启动缺失历史与查询输入分布不一致的影响。成熟上下文的可用源池和有效列数可以变化，不能解释成只有模型结构发生变化的消融。','',
        'RTX4060 Laptop 8GB：TabDPT禁用当前Windows构建不支持的Flash-only路径与torch.compile，使用作者公开use_flash=False选项；Nori禁用多流水线同时执行，固定原生分块元素预算，不量化上下文缓存，不允许静默丢弃已选上下文。起初的全查询试运行曾显存不足，改用上述固定分块后重新计算，失败运行不记成绩。','',
        'Nori每视图2048条上下文；Nori1.5使用默认4条预处理流水线，Nori6M限制4条（作者默认8条）。TabDPT用2个视图、2048随机上下文，或每条查询128个近邻；检索配置采用BF16，随机上下文配置采用FP32。其源池仍包含全部拟合期记录。TabICLv2用24,000条随机上下文、2个视图（作者默认8个）。这些预算不能用于宣布模型架构的普遍优劣。源池筛选和随机抽样不读取查询租金。失败及因耗时丢弃的未完成运行见`runtime_adjustments.json`。','',
        '## 后续相同窗口对比','',
        '| 路线 | 年度RMSE | 近期RMSE | 较早窗口选定配置 |','|---|---:|---:|---|']
    for family in ('tabdpt13','tabdpt13_adapt','nori','tabicl2'):
        part=selected.loc[selected.route==family].set_index('stage')
        lines.append(f"| {labels[family]} | {part.loc['outer','rmse_sgd']:.4f} | {part.loc['june','rmse_sgd']:.4f} | `{part.loc['june','name']}` |")
    outer=next(item for item in comparison if item['stage']=='outer');june=next(item for item in comparison if item['stage']=='june')
    for key,label in [('previous_ft','FT-Transformer（此前67列方案）'),('previous_history','原历史主融合'),('previous_deep','第六轮深度融合'),('new','加入新Transformer的融合')]:
        lines.append(f"| {label} | {outer[key]['rmse_sgd']:.4f} | {june[key]['rmse_sgd']:.4f} | — |")
    lines+=['','完整较早窗口成绩（包含负结果；128查询试运行不参与选择）：','',
        '| 配置 | RMSE | MAE | 偏差 | 上下文/视图 | 秒 |','|---|---:|---:|---:|---:|---:|']
    for row in table.loc[table.stage=='screen'].sort_values('rmse_sgd').itertuples():
        lines.append(f'| `{row.name}` | {row.rmse_sgd:.4f} | {row.mae_sgd:.4f} | {row.bias_sgd:+.4f} | {row.context_rows_per_view}/{row.members} | {row.seconds:.1f} |')
    lines+=['','### 匹配的监督适配对照','',
        '相同原始权重、输入、BF16精度、2个视图和2048条推理上下文，比较0轮冻结与较早窗口选定轮数的监督适配。训练时保持骨干推理模式下的上下文归一化，同时对指定参数反向传播。','',
        '| 窗口 | 冻结RMSE | 适配后RMSE | 选定训练轮数 | 每轮监督行数 |','|---|---:|---:|---:|---:|']
    for row in adaptation_rows:
        lines.append(f"| {row['stage']} | {row['matched_frozen_rmse']:.4f} | {row['adapted_rmse']:.4f} | {row['epochs']} | {row['supervised_rows_per_epoch']} |")
    lines+=['','## 为什么这些深度模型没有稳定超过原方案','',
        '以下区分已观察到的现象与原因假设。不能据此认定网络容量无效，也不能将全部差距归因于数据噪声。','',
        '### 历史特征可用性发生变化','',
        '| 近期窗口同期历史 | 拟合源池可用比例 | 查询可用比例 | Nori随机2048上下文可用比例 |','|---|---:|---:|---:|']
    for row in diagnostics['support']:
        if row['stage']=='june':
            lines.append(f"| {row['lag_months']}个月 | {row['source_supported_fraction']:.1%} | {row['query_supported_fraction']:.1%} | {row['nori2048_context_supported_fraction']:.1%} |")
    lines+=['','这表明训练中的冷启动/缺失组合与后续查询明显不同。此前未给神经路线单独控制这项分布差异，是实验设计仍需补足的检查。成熟上下文使TabICLv2早期原始目标从549.7987降至534.0392，但Nori没有相同收益，因而不能将其视为所有差距的唯一原因。','',
        '### 部分误差表现为整体低估','',
        '| 近期模型 | 实际RMSE | 平均偏差 | 去掉均值后的误差标准差（仅分解） |','|---|---:|---:|---:|']
    for row in diagnostics['error_decomposition']:
        if row['stage']=='june':
            lines.append(f"| {row['model']} | {row['rmse_sgd']:.4f} | {row['bias_sgd']:+.4f} | {row['centered_error_std_diagnostic_only']:.4f} |")
    lines+=['','最后一列通过验证标签计算，仅是MSE＝误差方差＋偏差平方的分解，**不是合法校准后的预测成绩**。TabM的−116元偏差和RealMLP的−67元偏差表明，需要检验跨时间水平校准；不能直接用验证集均值去修正提交。','',
        '### 上下文、训练适配与比较基准','',
        '冻结Nori每视图只使用2048条源记录，其有效监督覆盖与使用十万条以上拟合记录的树模型不同。TabDPT监督适配已让每条拟合记录参与训练，但早期训练损失从0.8269降到0.8149时，时间验证分数在首轮后回升；增加训练轮数未稳定解决泛化问题。输入表示、时间变化、预训练任务和本任务的差异仍需逐项消融，不能只归因为模型发布时间或参数量。','',
        '原442.90参照是历史特征＋10个成员的融合。应先比较同信息范围的单模型，再检查最终融合是否获得互补收益；表格中的不同特征支持、上下文大小与预处理预算仍然构成限制。','']
    lines+=['','## 融合与结论边界','',
        '候选包含原历史融合、第六轮融合及本轮完整较早窗口配置。非负和为1，每个新增候选最多0.4，并以0.001系数向原历史融合收缩。后续窗口不重新调权；权重见`ensemble_selection.json`。','',
        f"430目标{'达到' if summary['target_430_achieved'] else '尚未达到'}。近期新融合为{june['new']['rmse_sgd']:.4f}，历史主方案为442.9025。{'近期历史主方案继续保留，未覆盖已有提交文件。' if june['new']['rmse_sgd']>=442.9024826164729 else '新结果可作为待新数据确认的开发候选。'}",'',
        f'已逐行核对{len(table)}份完整预测的ID、月份、标签、样本数和重算指标；本轮新下载权重与官方LFS哈希一致，已有TabICL权重与原下载清单一致，融合逐行重组一致，7项测试通过。两份后续监督适配权重均重载并完整复现全部验证行；其他冻结配置核对了逐行预测和重建配方，不声称全部重载。','',
        '- `verified_metrics.csv`：全部完整配置及上下文/时间/显存预算。',
        '- `selected_model_comparison.csv`：早期选择后的年度与近期对照。',
        '- `checkpoint_manifest.json` / `verification.json`：不可变权重提交、哈希和核验。',
        '- `monthly_comparison_*.csv`：逐月融合误差。','',
        '## 复现','',
        '保留前六轮逐行预测；在既有CUDA环境中按隔离目录安装依赖。','',
        '```powershell','python -m pip install --no-deps --target .cache/transformer-deps -r requirements/optional_pretrained_transformers.txt',
        'python -m analysis.experiments.seventh_wave_sources','python -m analysis.experiments.seventh_wave_transformers screen',
        'python -m analysis.experiments.seventh_wave_ensemble select','python -m analysis.experiments.seventh_wave_ensemble outer',
        'python -m analysis.experiments.seventh_wave_ensemble june','python -m unittest tests.test_seventh_wave',
        'python -m analysis.experiments.seventh_wave_report','```','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    path=ROOT/'output/experiment_runs/00_report_and_final/REPORT.md';marker='\n## 11. 更新的Transformer路线\n'
    before=path.read_text(encoding='utf-8').split(marker)[0]
    appendix=[marker,'','新增TabDPT 1.3、Nori 1.5/6M，并将TabICLv2接入强历史特征；详细协议及全部负结果见[第七轮报告](../round_07_pretrained_transformers/README.md)。','',
        '| 路线 | 年度RMSE | 近期RMSE |','|---|---:|---:|']
    for family in ('tabdpt13','tabdpt13_adapt','nori','tabicl2'):
        part=selected.loc[selected.route==family].set_index('stage')
        appendix.append(f"| {labels[family]} | {part.loc['outer','rmse_sgd']:.4f} | {part.loc['june','rmse_sgd']:.4f} |")
    for key,label in [('previous_ft','FT-Transformer（此前67列方案）'),('previous_history','原历史主融合'),('previous_deep','第六轮融合'),('new','加入新Transformer的融合')]:
        appendix.append(f"| {label} | {outer[key]['rmse_sgd']:.4f} | {june[key]['rmse_sgd']:.4f} |")
    appendix+=['',f"430目标{'达到' if summary['target_430_achieved'] else '尚未达到'}。模型容量、预处理及上下文预算不同；这些是重复查看的本地开发结果，不能直接视作Kaggle分数。TabPFN-3.5因需用户本人登录接受模型许可而未实测。",'']
    status_start='<!-- seventh-wave-status:start -->';status_end='<!-- seventh-wave-status:end -->'
    if status_start in before:
        prefix,remainder=before.split(status_start,1)
        before=prefix+remainder.split(status_end,1)[1].lstrip('\n')
    title,body=before.split('\n',1)
    status=(f"{status_start}\n**第七轮Transformer更新：** 已实测TabDPT 1.3（含监督适配）、Nori 1.5/6M和强历史输入TabICLv2；新融合年度/近期RMSE为{outer['new']['rmse_sgd']:.4f}/{june['new']['rmse_sgd']:.4f}。完整结果与预算见[第11节](#11-更新的transformer路线)。430目标{'达到' if summary['target_430_achieved'] else '尚未达到'}。\n{status_end}\n\n")
    path.write_text(title+'\n\n'+status+body.lstrip('\n')+'\n'.join(appendix),encoding='utf-8')
    print(json.dumps({'verified_predictions':len(table),'comparisons':comparison,'selected_models':models},indent=2),flush=True)


def plot(selected,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for axis,stage in zip(axes,('outer','june')):
        rows=selected.loc[selected.stage==stage]
        comparison=next(item for item in comparisons if item['stage']==stage)
        names=list(rows.family)+['Prior history blend','Prior deep blend','New blend']
        values=list(rows.rmse_sgd)+[comparison[key]['rmse_sgd'] for key in ['previous_history','previous_deep','new']]
        axis.barh(names,values,color=['#447ca8']*len(rows)+['#b7a38a','#ad946e','#b46736'])
        axis.invert_yaxis();axis.set_xlim(0,max(values)*1.15)
        for i,value in enumerate(values):
            axis.text(value+2,i,f'{value:.2f}',va='center',fontsize=8)
        axis.set_title('12-month development window' if stage=='outer' else '10-month development window')
        axis.set_xlabel('RMSE (SGD)')
    fig.savefig(OUT/'transformer_comparison.png',dpi=180);fig.savefig(OUT/'transformer_comparison.svg');plt.close(fig)


if __name__=='__main__':
    main()
