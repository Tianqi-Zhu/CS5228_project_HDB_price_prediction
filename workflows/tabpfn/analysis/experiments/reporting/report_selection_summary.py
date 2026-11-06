"""Build a sourced report shortlist, controlled tables, and independent checks."""
from __future__ import annotations
import gc
import json
import sys
from urllib.parse import quote
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT,frame_data,score,write_json
from analysis.experiments.report_selection_trials import OUT,ART,select_features
from analysis.experiments.third_wave_training import make_data


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def relative(path):
    return str(path.relative_to(ROOT)).replace('\\','/')


def branch_source(branch,path,branches):
    commit=next(b['commit'] for b in branches if b['branch']==branch)
    return f'https://github.com/Tianqi-Zhu/CS5228_project_HDB_price_prediction/blob/{commit}/{quote(path)}'


def registry(branches):
    common=pd.read_csv(ROOT/'output/experiment_runs/comparison_same_27_features/model_metrics.csv').set_index('family')
    rows=[]
    def add(name,model_class,rmse,period,features,role,source,notes):
        rows.append({'model':name,'class':model_class,'reference_rmse_sgd':rmse,
            'period':period,'features_or_information':features,'report_role':role,'source':source,'notes':notes})
    period='2024-04..2025-03; frozen origin'
    add('训练期均值','常量',846.7548764601373,period,'训练标签均值','基准行',
        'output/experiment_runs/baseline_mean_catboost/results.json','不占正文模型名额；证明预测任务的难度。')
    ridge=read('output/experiment_runs/00_report_and_final/ridge/outer/ridge27/results.json')
    add('Ridge 27','正则线性',ridge['rmse_sgd'],period,'同27特征；拟合集填补/缩放/One-hot','正文',
        'output/experiment_runs/00_report_and_final/ridge/outer/ridge27/results.json','本轮补齐公平对照；alpha 仅在2023-10..2024-03选择。旧7列624.88不能放进同27列排名。')
    names={'extra_trees':('ExtraTrees','随机树集成','正文'),
        'lightgbm':('LightGBM','梯度提升树','正文'),'xgboost':('XGBoost','梯度提升树','正文'),
        'catboost':('CatBoost','类别梯度提升树','正文'),
        'embedding_mlp':('Embedding MLP','前馈神经网络','正文'),
        'ft_transformer':('FT-Transformer 96','表格注意力网络','正文'),
        'hist_gradient_boosting':('HistGradientBoosting','直方图梯度提升','附录'),
        'embedding_resnet':('Embedding ResNet','残差前馈网络','附录'),
        'ft_transformer_default':('FT-Transformer 192','表格注意力网络','附录'),
        'tabnet':('TabNet','注意力特征选择网络','附录')}
    for family,(name,kind,role) in names.items():
        add(name,kind,float(common.loc[family,'rmse_sgd']),period,'同27特征；全部拟合行',role,
            f'output/experiment_runs/comparison_same_27_features/{family}/results.json',
            '架构对照允许各自预处理；搜索预算不完全相等，只有一个种子和一个主要年度窗。')
    add('Random Forest','Bagging树',493.7718838822359,period,'旧基础7特征','附录',
        'output/experiment_runs/baseline_classical_models/results.json','与ExtraTrees机制相近；没有必要在正文重复占位；特征不同不直接混排。')
    add('KNN','局部相似检索',539.5806174535578,period,'旧9特征，拟合仅12000行','附录',
        'output/experiment_runs/baseline_spatial_models/metrics.csv','仅采样基准，不能据此推断全量KNN的最佳性能。')
    add('ReLU MLP','sklearn前馈网络',463.97194036186323,'2025-01..2025-03; 8924 rows',
        '原始属性、时间、类别；拟合集IQR清理','附录',
        'output/experiment_runs/baseline_relu_mlp/historical_results.json','三个月窗口；本地复现463.7769。不能与年度/近期10个月排名。')
    add('早期自定义Transformer','神经网络',471.84910011717506,period,'辅助特征＋历史编码/残差','附录',
        'output/experiment_runs/baseline_deep_models/summary/model_metrics.csv','与官方FT接口及后续27特征实验分开标识。')
    add('TabM-style','自定义共享MLP＋秩一适配',561.7053,period,'历史空间特征','附录',
        'output/experiment_runs/baseline_tabm/fold_metrics.csv','不是官方TabM；三折pooled为646.37，树＋6%网络融合年度468.83；同OOF选权属于选择分数。')
    def branch_add(name,kind,rmse,period_,features,role,branch,path,notes):
        add(name,kind,rmse,period_,features,role,branch_source(branch,path,branches),notes)
    branch_add('官方TabM seed42','参数共享神经集成',463.196,'2024-06..2025-03; frozen origin',
        '77列；与TabPFN同残差/历史管线','扩展附录','experiment/tabpfn','reports/tabm_80_20/report.md',
        '官方tabm==0.0.3；seed43/44分别465.285/460.379；三种子均值462.954，SD2.462，未做种子挑选融合。')
    branch_add('TabPFN 3.5 Fast 68列','预训练上下文学习',459.812371,'2024-06..2025-03; frozen origin',
        '68列，全121005上下文','扩展附录','experiment/tabpfn','reports/tabpfn_v3_5_fast_80_20/scores.csv',
        '同管线CatBoost460.782、TabPFN3为465.410；与77列版本不同。')
    branch_add('TabPFN 77列冻结','预训练上下文学习',455.8135701208574,'2024-06..2025-03; frozen origin',
        '77列类别月份＋时间对齐历史＋残差，全上下文','扩展附录','experiment/tabpfn',
        'reports/tabpfn_v3_5_fast_calendar_80_20/scores.csv','无需声称胜过全部树模型；本地62特征树模型更低。')
    branch_add('TabPFN DiagScale/MSE','预训练模型适配',455.07102836,'2024-06..2025-03; frozen origin',
        '同77列；少量参数适配','扩展附录','experiment/tabpfn','reports/tabpfn_components_80_20/scores.csv',
        'MSE-only vs 匹配CRPS+MSE只改善0.051，月度区块区间跨零；也试过Q/K LoRA、head、视图数、采样，不能只选最低值宣称稳定突破。')
    add('TabICLv2','预训练上下文学习',488.17665602095036,period,'47列相对历史＋残差，24000上下文','附录',
        'output/experiment_runs/round_02_anniversary_history/tabicl/outer/relative_residual24k/results.json',
        '预训练模型并非自然优于树；更大上下文受GPU内存约束。')
    branch_add('PCA租金面＋状态空间＋CatBoost','空间时间混合',467.84,period,
        '16×16空间面、5因子、Kalman预测＋静态/过去租金','方法扩展','experiment/dynamic-rental-surface-plan',
        'reports/factored_rental_surface/report.md','同实验全历史CatBoost466.23；残差版三折pooled538.06 vs539.29，但年度更差；面重建/预测误差不是逐条租金RMSE。')
    branch_add('RNN市场编码＋树','序列特征＋树',478.89405617986955,'2024-04..2025-03; monthly updated',
        '月度租金/COE/股票面板；lookback3','方法扩展','codex/rnn-lstm-market-features',
        'output/experiment_runs/rnn_lstm_summary/latest/results.json',
        '基准478.51855；RNN seed42/43/44均未改善。使用真实前月汇总，不能放入固定原点表。')
    branch_add('LSTM市场编码＋树','序列特征＋树',479.2395486603623,'2024-04..2025-03; monthly updated',
        '与RNN同信息条件','方法扩展','codex/rnn-lstm-market-features',
        'output/experiment_runs/rnn_lstm_summary/latest/results.json','不是独立逐条LSTM回归器；序列样本少，选中epoch1的情况多。')
    add('分层历史先验','分组平滑',476.01391497375033,period,'过去分层租金','附录',
        'output/experiment_runs/baseline_deep_models/summary/model_metrics.csv','可作为简单历史基准；不替代同特征模型对照。')
    add('第二轮融合','异质历史树融合',452.93860380419267,period,'27/47/62/87列及标准化目标','性能进展',
        'output/experiment_runs/round_02_anniversary_history/ensemble_outer.json','近期443.8630；更早独立选权16个月压力窗479.6351，不能将重叠窗合并。')
    add('第三轮预先锁定融合','树＋FT融合',451.3097487794427,period,'第二轮＋新模型/特征/专家','正文稳健性对照',
        'output/experiment_runs/round_03_tree_transformer_ensemble/ensemble_outer.json','近期443.0052；早期窗口先选参数权重。与27列主表信息不同，单列在性能进展表。')
    add('最优历史特征融合（第三轮仅新增模型组）','树＋FT融合',451.1810257119484,period,'原融合＋FT＋两种XGBoost','正文第8项：最终最佳开发方案',
        'output/experiment_runs/round_03_tree_transformer_ensemble/ablation_models.json','近期442.9025；查看开发窗后选择分组，明确标为探索性最低值。')
    branch_add('公开租金查表','目标复原',181.2533674382674,'2024-04..2025-03; public target reconstruction',
        '使用公开源中目标月份租金','数据审计附录','experiment/public-rent-lookup','results/summary.json',
        '不在train＋auxiliary的预测范围；173.412是未知目标的假设误差尺度，不是实测Kaggle分数。')
    return pd.DataFrame(rows)


def verify_new():
    source,hashes=frame_data()
    checks=[]
    for path in sorted(OUT.glob('*/*/*/results.json')):
        record=json.loads(path.read_text())
        saved=ART/path.parent.relative_to(OUT)/'predictions.csv'
        if not saved.exists():
            continue
        frame=pd.read_csv(saved).sort_values('source_row')
        actual_source=source.iloc[frame.source_row.to_numpy(int)]
        np.testing.assert_array_equal(frame.actual,actual_source.MONTHLY_RENT)
        np.testing.assert_array_equal(frame.month,actual_source.RENT_APPROVAL_DATE)
        start,end=('2023-10','2024-04') if record['stage']=='screen' else (
            ('2024-04','2025-04') if record['stage']=='outer' else ('2024-06','2025-04'))
        expected=source.index[(source.RENT_APPROVAL_DATE>=start)&(source.RENT_APPROVAL_DATE<end)]
        np.testing.assert_array_equal(frame.source_row,expected)
        assert frame.source_row.is_unique and np.isfinite(frame.prediction).all()
        for key,value in score(frame.actual,frame.prediction).items():
            np.testing.assert_allclose(value,record[key],atol=1e-8,rtol=0)
        assert hashes==record['source_sha256']
        checks.append({'experiment':relative(path.parent),'rows':len(frame),**score(frame.actual,frame.prediction)})
    common_checks=[]
    for directory in sorted((ROOT/'output/experiment_runs/comparison_same_27_features').iterdir()):
        if not directory.is_dir() or not (directory/'predictions.csv').exists():
            continue
        result=json.loads((directory/'results.json').read_text())
        predictions=pd.read_csv(directory/'predictions.csv').sort_values('source_row')
        expected=source.loc[(source.RENT_APPROVAL_DATE>='2024-04')&(source.RENT_APPROVAL_DATE<'2025-04')]
        np.testing.assert_array_equal(predictions.source_row,expected.index)
        np.testing.assert_array_equal(predictions.actual,expected.MONTHLY_RENT)
        np.testing.assert_array_equal(predictions.approval_month,expected.RENT_APPROVAL_DATE)
        for key,value in score(predictions.actual,predictions.prediction).items():
            np.testing.assert_allclose(value,result['scores'][key],atol=1e-8,rtol=0)
        common_checks.append(directory.name)
    reloads=[]
    for stage in ('outer','june'):
        bundle=make_data(stage)
        for family in ('lgb','cat'):
            for level in (19,27,47,62):
                name=f'{family}_f{level}'
                saved=joblib.load(ART/'ladder'/stage/name/'model.joblib')
                _,q=select_features(bundle,level)
                if family=='cat':
                    for c in q.select_dtypes(include=['category','object','string']).columns:
                        q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
                    q=q.fillna(-999.)
                predicted=saved['model'].predict(q)
                expected=pd.read_csv(ART/'ladder'/stage/name/'predictions.csv').prediction
                np.testing.assert_allclose(predicted,expected,atol=1e-7,rtol=0)
                reloads.append({'stage':stage,'name':name,'max_difference_sgd':float(np.max(np.abs(predicted-expected)))})
        saved=joblib.load(ART/'ridge'/stage/'ridge27/model.joblib')
        q=bundle['x_val'].copy()
        for c in q.select_dtypes(include=['category','object','string']).columns:
            q[c]=q[c].astype('string').fillna('__MISSING__').astype(str)
        predicted=saved['model'].predict(q)
        expected=pd.read_csv(ART/'ridge'/stage/'ridge27/predictions.csv').prediction
        np.testing.assert_allclose(predicted,expected,atol=1e-7,rtol=0)
        reloads.append({'stage':stage,'name':'ridge27','max_difference_sgd':float(np.max(np.abs(predicted-expected)))})
        gc.collect()
    optimization={}
    if (OUT/'optimization_selection.json').exists():
        from analysis.experiments.third_wave_cross_ensemble import baseline
        selection=json.loads((OUT/'optimization_selection.json').read_text())
        for stage in ('outer','june'):
            path=ART/'optimized_ensemble'/stage/'predictions.csv'
            if not path.exists():
                continue
            bundle=make_data(stage)
            combined=np.zeros(len(bundle['y_val']))
            for name,weight in selection['weights'].items():
                if name=='third_wave_locked':
                    values=baseline(stage,bundle)
                else:
                    frame=pd.read_csv(ART/'optimization'/stage/name/'predictions.csv').sort_values('source_row')
                    np.testing.assert_array_equal(frame.source_row,bundle['row_val'])
                    values=frame.prediction.to_numpy(float)
                combined+=weight*values
            saved=pd.read_csv(path).sort_values('source_row')
            np.testing.assert_allclose(saved.prediction,combined,atol=1e-8,rtol=0)
            optimization[stage]=score(saved.actual,saved.prediction)
    pd.DataFrame(checks).to_csv(OUT/'new_verified_metrics.csv',index=False)
    verification={'new_prediction_files_verified':len(checks),'model_reloads':reloads,
        'source_sha256':hashes,'optimization_ensembles_reconstructed':optimization,
        'reused_common_feature_predictions_verified':common_checks}
    write_json(OUT/'verification.json',verification)
    return verification


def plots(main,ladder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    labels=['Ridge','ExtraTrees','LightGBM','XGBoost','CatBoost','MLP','FT-Transformer']
    colors=['#93a6ba']+['#3c82b5']*4+['#ca8b45']*2
    axes[0].barh(labels,main.rmse_sgd,color=colors)
    axes[0].invert_yaxis();axes[0].set_xlim(0,560)
    for i,value in enumerate(main.rmse_sgd):
        axes[0].text(value+3,i,f'{value:.2f}',va='center',fontsize=8)
    axes[0].set_title('Architecture comparison: same 27 inputs')
    axes[0].set_xlabel('RMSE (SGD), Apr 2024-Mar 2025')
    for index,stage in enumerate(('outer','june'),start=1):
        for family,label,color in [('lgb','LightGBM','#3c82b5'),('cat','CatBoost GPU','#b26435')]:
            part=ladder.loc[(ladder.stage==stage)&(ladder.family==family)].sort_values('features')
            axes[index].plot(part.features.astype(str),part.rmse_sgd,'o-',label=label,color=color)
        axes[index].set_title('Fixed-round nested features: '+('12 months' if stage=='outer' else '10 months'))
        axes[index].set_xlabel('Number of source features')
        axes[index].set_ylabel('RMSE (SGD)');axes[index].legend(fontsize=8)
        axes[index].grid(alpha=.2)
    fig.savefig(OUT/'report_comparison.png',dpi=180)
    fig.savefig(OUT/'report_comparison.svg')
    plt.close(fig)


def run():
    branches=json.loads((OUT/'branches.json').read_text(encoding='utf-8'))
    catalog=registry(branches)
    catalog.to_csv(OUT/'model_catalog.csv',index=False,encoding='utf-8-sig')
    common=pd.read_csv(ROOT/'output/experiment_runs/comparison_same_27_features/model_metrics.csv').set_index('family')
    family_order=['ridge','extra_trees','lightgbm','xgboost','catboost','embedding_mlp','ft_transformer']
    descriptions={
        'ridge':('Ridge','线性可解释参照；检验非线性交互是否必要'),
        'extra_trees':('ExtraTrees','随机化树集成；与逐步提升树形成机制对照'),
        'lightgbm':('LightGBM','主特征消融模型；快速迭代、刻画非线性'),
        'xgboost':('XGBoost','另一种正则化提升树；强性能对照与融合成员'),
        'catboost':('CatBoost','直接处理类别信息；验证历史特征在另一树模型上的增量'),
        'embedding_mlp':('Embedding MLP','神经网络基线；区分Embedding与注意力结构'),
        'ft_transformer':('FT-Transformer','特征间注意力；检验复杂结构与融合互补性')}
    main=[]
    for family in family_order:
        if family=='ridge':
            result=read('output/experiment_runs/00_report_and_final/ridge/outer/ridge27/results.json')
            metrics={key:result[key] for key in ('rmse_sgd','mae_sgd','bias_sgd')}
            source='output/experiment_runs/00_report_and_final/ridge/outer/ridge27/results.json'
            selected=f"alpha={result['alpha']}"
        else:
            result=common.loc[family]
            metrics={key:float(result[key]) for key in ('rmse_sgd','mae_sgd','bias_sgd')}
            source=f'output/experiment_runs/comparison_same_27_features/{family}/results.json'
            selected=f"iterations/epochs={int(result['iterations'])}"
        name,role=descriptions[family]
        main.append({'family':family,'model':name,'report_purpose':role,'features':27,
            'fitting_rows':114842,'validation_rows':35158,'period':'2024-04..2025-03',
            'selection':selected,'source':source,**metrics})
    main=pd.DataFrame(main)
    main.to_csv(OUT/'standalone_model_comparison.csv',index=False,encoding='utf-8-sig')
    from analysis.experiments.report_best_scheme import build as best_scheme
    best,best_row,best_lines=best_scheme()
    recommended=pd.concat([main,pd.DataFrame([best_row])],ignore_index=True)
    recommended.to_csv(OUT/'recommended_models.csv',index=False,encoding='utf-8-sig')
    ladder=[]
    for path in sorted((OUT/'ladder').glob('*/*/results.json')):
        record=json.loads(path.read_text())
        ladder.append({k:record[k] for k in ('stage','family','features','iterations','rows','rmse_sgd','mae_sgd','bias_sgd')})
    ladder=pd.DataFrame(ladder).sort_values(['stage','family','features'])
    ladder.to_csv(OUT/'controlled_feature_ladder.csv',index=False)
    plots(main,ladder)
    verification=verify_new()
    old=read('output/experiment_runs/round_03_tree_transformer_ensemble/summary.json')
    optimization=verification['optimization_ensembles_reconstructed']
    branch_notes={
        'main':('代码骨架','无可评估模型；不产生分数'),
        'feature/midterm-eda-models':('CatBoost A/B/C、Ridge/RF/LGB，辅助特征和早期深度模型整合','报告的基础属性、时间、空间和原始模型证据'),
        'feature/temporal-auxiliary-analysis':('历史空间树、滞后宏观、分组面板','固定原点历史与按月更新经济协议分开'),
        'feature/neural-tabm-experiment':('自定义TabM-style及历史树融合','网络年度561.71，树＋6%网络468.83；不是官方TabM'),
        'relu-rental-prediction':('sklearn ReLU MLP，嵌套选epoch和拟合集IQR','只有2025Q1：463.97；不与年度表混排'),
        'experiment/dynamic-rental-surface-plan':('动态市场指数、租金面、PCA、AR/状态空间、CatBoost混合','解释市场分解；逐条租金与平滑面误差分开'),
        'codex/rnn-lstm-market-features':('RNN/LSTM市场状态、窗口/截点搜索、三种子','按月更新真实前月信息；整体未超过自己的树基准'),
        'experiment/tabpfn':('TabPFN 3/3.5、上下文/日历/对齐/适配；官方TabM；空间扩展','核心预训练路线；77列冻结455.81，MSE适配455.07；官方TabM463.20'),
        'experiment/public-rent-lookup':('公开目标记录归一化匹配','181.25是目标复原；不符合仅train＋auxiliary预测主线')}
    branch_rows=[]
    for branch in branches:
        name=branch['branch']
        method,note=branch_notes.get(name,('LightGBM/CatBoost/KNN空间基准与控制分析','26列LGB477.20；固定效应/匹配分析是解释证据，不是预测榜单'))
        branch_rows.append({'branch':name,'commit':branch['commit'],'unique_story':method,'report_treatment':note})
    pd.DataFrame(branch_rows).to_csv(OUT/'branch_summary.csv',index=False,encoding='utf-8-sig')
    lines=['# 报告模型筛选与全部分支对照（2026-11-04）','',
        '## 1. 建议采用的模型组合','',
        '**正文采用7个单模型＋第8项“最优历史特征融合（Best ensemble）”，另放一行均值基准。** 7个单模型为 Ridge、ExtraTrees、LightGBM、XGBoost、CatBoost、Embedding MLP、FT-Transformer；第8项是当前最低开发RMSE方案，年度451.18、近期442.90，必须作为正文的最终方案列出。','',
        '先用同27输入比较模型，再固定模型解释特征增量，最后单独展示增强特征与融合的性能进展。不要把不同输入、不同截止时间的分数拼成一张架构排名。',
        '','| 模型 | 同27列年度RMSE | MAE | 偏差（预测−实际） | 在报告中承担的作用 |',
        '|---|---:|---:|---:|---|']
    for row in main.to_dict('records'):
        lines.append(f"| {row['model']} | {row['rmse_sgd']:.2f} | {row['mae_sgd']:.2f} | {row['bias_sgd']:+.2f} | {row['report_purpose']} |")
    lines+=best_lines
    lines+=['','均值基准年度RMSE为846.75。上方同27列单模型表均在114,842条截至2024-03的记录上拟合，预测相同35,158条2024-04至2025-03记录。类别词表、填补与缩放仅在拟合集学习；各模型可采用不同的合法预处理。',
        '','原有同27列对照包含除Ridge外的所有模型。本轮补齐Ridge，alpha在2023-10至2024-03比较1/100/10000后固定。其他模型沿用其相同输入的内部时间选择。由于搜索预算不同、单种子和已查看过验证期，这仍是受控开发比较，不能声称是模型架构的普遍排名。',
        '','**篇幅更紧时**可将ExtraTrees和Embedding MLP移至附录，保留Ridge、LightGBM、XGBoost、CatBoost、FT-Transformer和融合共6项；若报告强调神经结构比较，则保留MLP，把XGBoost的细节压缩。建议完整稿保留7＋1。',
        '','## 2. 特征工程：用固定模型与固定轮数讲清楚','',
        '四层输入严格嵌套：19列房屋/时间/位置/设施；27列增加楼栋×户型、规划区×户型的历史均值、数量、最近均值和陈旧度；47列再加五种粒度的相对市场租金历史；62列再加楼栋×户型在12/24/36个月前±1月的样本量、原始租金、市场比值、离散度和低租金占比。',
        '','本轮补做16次拟合：LightGBM固定353棵、15叶、学习率0.03、L2=10；GPU CatBoost固定535轮、深度6、学习率0.04、L2=10。参数和轮数取自先前较早窗口并在四层特征间完全相同。每个窗口全部验证行保留，验证期间不接收真实租金更新。',
        '','本节固定配置与第1节的模型选择配置不同：例如第1节LightGBM为31叶/108棵，CatBoost为CPU配置；这里为15叶/353棵LightGBM和GPU CatBoost。两节的27列分数不同来自训练配置差异，不是结果冲突。只在各自表内解释模型或特征的影响。',
        '','| 模型 / 窗口 | 19列 | 27列 | 47列 | 62列 | 解释重点 |','|---|---:|---:|---:|---:|---|']
    for stage,label in [('outer','2024-04—2025-03'),('june','2024-06—2025-03')]:
        for family,name in [('lgb','LightGBM'),('cat','CatBoost GPU')]:
            part=ladder.loc[(ladder.stage==stage)&(ladder.family==family)].set_index('features')
            numbers=[f'{part.loc[level,"rmse_sgd"]:.2f}' for level in (19,27,47,62)]
            lines.append(f"| {name} / {label} | {' | '.join(numbers)} | 先看历史增量，再看同期历史增量 |")
    lines+=['','这里每一步的Δ可解释为该固定模型/训练方案下新增特征组的预测增量，仍不是因果效应，也不代表某种续约被识别出来。没有单元/租户ID，只有楼栋和户型。早期月份没有足够的两年前历史，计数为0，统计缺失；远期查询只使用预测起点已观测月份。',
        '','**报告最有说服力的特征结论**：时间与过去租金信息最关键；同户型市场归一化的收益较小且依赖模型；12/24/36个月同期历史提供了更明显的新信息。更多特征并不保证更好：扩大到±3月、局部归一化与尾部统计，在GPU CatBoost深度8上，早期窗口526.08→525.21，却使年度454.77→458.71、近期445.58→449.53。应保留这个负结果说明过拟合与验证的重要性。',
        '','早期基础叙事可引用同固定LightGBM的A→H辅助消融：属性758.24，加入时间526.11，加入Town485.09，楼栋480.20，设施476.97；再加COE479.22、股票480.54，没有稳定收益。经济变量涉及额外可获得性问题，不并入本轮批量固定原点模型。19/27/47/62主表更适合严格解释新增历史特征的作用。',
        '','原有156次逐列/成组重训用于补充解释：三年度汇总删除“时间＋全部历史”使RMSE增加141.33，删除全部历史增加7.94；删除全部设施只增加0.58。相关特征会替代信息，不能把单列重要性相加或把删掉某列当成完全移除了该源信息。见[特征证据](../features_ablation_evidence/README.md)。',
        '','## 3. 性能进展与RMSE目标','',
        '| 方案 | 年度35,158行 | 近期28,995行 | 结果性质 |','|---|---:|---:|---|',
        '| 第一轮搜索融合 | 465.66 | — | 旧开发参考 |',
        '| Frontier融合 | 463.76 | 456.82 | 较早窗口选权后固定 |',
        '| 同期历史第二轮融合 | 452.94 | 443.86 | 较早窗口选权后固定 |',
        '| 第三轮完整融合 | 451.31 | 443.01 | 较早窗口预先锁定 |',
        '| **推荐第8项：最优历史特征融合** | **451.18** | **442.90** | 第三轮仅保留新增模型组；最佳开发方案 |']
    if optimization:
        lines.append(f"| 本轮正则化扩展融合 | {optimization['outer']['rmse_sgd']:.2f} | {optimization['june']['rmse_sgd']:.2f} | 较早窗口保留原完整融合，新增权重均为0 |")
    lines+=['','27列主表回答架构差异；上表允许特征、目标表示和融合变化，回答完整方案能达到什么水平。两张表不能据此计算“某架构提升了多少”。固定27列默认400棵LightGBM的468.49与内部早停108棵的468.87属于不同参数，均须如实标注。',
        '','400—430仍未达到。第三轮完整融合的年度MAE为331.05、近期324.82，均比第二轮的328.44/322.92略差；只新增模型组的近期MAE为323.73。选择以RMSE为主的方案有明确代价，不应写成所有指标都改善。',
        '','本轮额外尝试10组配置：CatBoost深度4/5/7/9、增强L2的深度6/8、CPU Ordered深度4/6，以及两种正则化随机阈值LightGBM。候选先在2023-10至2024-03评估，再用固定0.001收缩和每新模型0.25上限选权。新增候选未获得权重，因此没有重训这些候选的后续窗口，也没有把内部偶然最低值写成新纪录。完整结果保存在`optimization/screen/`。',
        '','年度窗和近期窗相互重叠，不能合并计算更多独立样本。较早起点的16个月压力窗，由单独的起点前选参和选权得到第二轮融合479.64，优于同窗旧XGBoost487.58；它不能被替换成更短窗口的442.90来说明长跨度预测已解决。',
        '','## 4. 所有GitHub分支的审查结论','',
        '已于2026-11-04完成 `git fetch origin --prune`，读取全部10个实际远端分支；`origin/HEAD`仅为main的符号引用，不计作第11个分支。读取Git对象而未切换工作区或覆盖现有修改。分支提交及证据blob存于`branches.json`、`branch_inventory.csv`，小型报告证据去重保存于`branch_evidence/`。','',
        '| 分支 | 提交 | 独有内容 | 报告处理 |','|---|---|---|---|']
    for row in branch_rows:
        lines.append(f"| `{row['branch']}` | `{row['commit'][:8]}` | {row['unique_story']} | {row['report_treatment']} |")
    lines+=['','当前工作区新增的feature_selection、features_ablation_evidence、comparison_model_search、comparison_same_27_features、round_01_history_target_search、round_02_anniversary_history、third_wave等尚未出现在远端当前分支提交中，本报告另行读取本地结果，不把它们伪装成已上传的GitHub成果。所有本地结果路径可从`local_run_index.csv`追溯。',
        '','## 5. 附录保留什么、为什么不全塞进正文','',
        '| 路线 | 代表结果及对照 | 建议 |','|---|---|---|',
        '| 官方TabM | 近期463.20；同77列TabPFN冻结455.81。三种子均值462.95、SD2.46 | 展示现代参数共享神经集成；与旧TabM-style分开 |',
        '| TabPFN | 68列Fast459.81；77列冻结455.81、DiagScale455.21、MSE-only455.07 | 作为预训练路线扩展；主表不是相同27输入，不混排架构 |',
        '| TabPFN适配、采样与空间外部增强 | Q/K LoRA、原生head、多视图、局部上下文均未稳定超过冻结；MSE-only vs匹配loss对照仅−0.051，区间跨零 | 用一段负结果总结；外部收入/土地覆盖等不属于本次train＋auxiliary范围 |',
        '| TabICLv2 | 年度488.18，47列/24000上下文 | 保留基础模型并非自动更优的负结果 |',
        '| HistGradientBoosting | 同27列年度468.02 | 与多个提升树叙事重叠，放完整附表 |',
        '| ResNet / TabNet / FT-192 | 同27列493.88 / 498.53 / 476.25 | 说明更复杂不必更好；FT-192与FT-96不必各占正文一行 |',
        '| ReLU MLP | 2025Q1为463.97，本地复现463.78 | 仅3个月，放附录；正文MLP使用同27输入的483.72 |',
        '| RNN/LSTM市场特征 | 月度更新基准478.52；RNN42为478.89、LSTM42为479.24 | 作为序列表示的负结果，不能冒充固定原点预测比较 |',
        '| PCA动态租金面＋状态空间 | 年度残差混合467.84，同实验全历史CatBoost466.23；pooled改善集中在早期上涨段 | 解释市场分解，明确中间面误差不等于逐条租金误差 |',
        '| KNN / Random Forest | 旧特征KNN采样539.58、RF493.77 | 机制或预算与主表不同，保留归档不重复正文 |',
        '| 公开目标匹配 | 年度已知标签重建181.25；173.41为假设误差尺度 | 仅数据来源审计，不纳入预测排名和目标达成判断 |',
        '','`model_catalog.csv`给出所有以上模型/路线的独立条目及原始证据，`branch_metric_tables.json`与`all_branch_scores.csv`保存分支内数值索引。索引包含分组误差、校准误差、pooled误差和中间目标，不能按全表最小RMSE自动排名。',
        '','## 6. 可直接用于报告的叙事草稿','',
        '> 我们先在统一的27个输入和固定时间验证集上比较线性模型、随机树集成、梯度提升树及神经网络，以区分模型结构的影响。Ridge提供可解释的线性参照，ExtraTrees反映随机化树集成的能力，LightGBM、XGBoost和CatBoost比较不同提升树实现，Embedding MLP与FT-Transformer检验类别嵌入和特征注意力是否带来增益。在有限搜索预算下，树模型的单模型表现整体优于所测试的神经网络配置；这一结果仅适用于当前数据与训练预算。',
        '','> 随后，我们固定LightGBM和CatBoost的参数、训练轮数及样本，逐步加入静态属性、普通历史租金、相对市场历史与同期历史统计。两类模型上的变化共同支持历史租金和同期信息具有预测增量，而更复杂的窗口、空间和宏观表示并未稳定改善结果。最后，我们将不同模型的互补误差用于融合，并将预先锁定方案与评估后消融选出的开发最低值分别报告。',
        '','> 这些实验使用比赛训练数据与所给辅助数据。验证目标属于已多次查看的开发期，不能作为新的独立测试成绩。最终结论以统一时间窗口、全部验证记录上的SGD租金RMSE为准，并同时报告MAE、偏差和长跨度压力窗口，以避免只展示最有利的单个分数。',
        '','## 7. 验证与复现','',
        f"本轮独立核对{verification['new_prediction_files_verified']}份新增逐行预测及{len(verification['reused_common_feature_predictions_verified'])}份已有同27列模型预测，全部验证ID、月份、标签、样本数和重算指标一致；重载{len(verification['model_reloads'])}个Ridge/特征阶梯模型得到一致预测。31项相关单元测试通过。输入哈希与历史实验一致。GitHub结果属于源分支记录，TabPFN/官方TabM没有在本机重新训练；本报告不把归档核对等同于完整重跑。",'',
        '```powershell',
        'python -m analysis.experiments.report_model_catalog',
        'python -m analysis.experiments.report_selection_trials ridge',
        'python -m analysis.experiments.report_selection_trials ladder --stage outer',
        'python -m analysis.experiments.report_selection_trials ladder --stage june',
        'python -m analysis.experiments.report_selection_trials optimize --stage screen',
        'python -m analysis.experiments.report_selection_blend select',
        'python -m analysis.experiments.report_selection_blend outer',
        'python -m analysis.experiments.report_selection_blend june',
        'python -m analysis.experiments.report_selection_summary','```','',
        '图表：[模型与特征对照](report_comparison.png)。机器可读主表：`recommended_models.csv`；特征阶梯：`controlled_feature_ladder.csv`；分支清单：`branch_summary.csv`；模型全目录：`model_catalog.csv`。','']
    report_path=OUT/'REPORT.md'
    continuation=''
    if report_path.exists():
        previous=report_path.read_text(encoding='utf-8')
        marker='\n## 8. 本地RMSE 430目标的后续探索\n'
        if marker in previous:
            continuation=marker+previous.split(marker,1)[1]
    report_path.write_text('\n'.join(lines)+continuation,encoding='utf-8')
    summary={'remote_branches':len(branches),'curated_model_routes':len(catalog),
        'recommended_standalone_models':main.to_dict('records'),
        'recommended_models':json.loads(recommended.to_json(orient='records',force_ascii=False)),
        'recommended_final_scheme':best,
        'feature_ladder':ladder.to_dict('records'),'optimization':optimization,
        'previous_best_development':old['best_development_june'],
        'verification':verification,'report_path':relative(OUT/'REPORT.md')}
    write_json(OUT/'summary.json',summary)
    print(json.dumps({'branches':len(branches),'model_routes':len(catalog),'main_models':len(main),
        'verified_predictions':verification['new_prediction_files_verified'],'reloads':len(verification['model_reloads']),
        'optimization':optimization},indent=2),flush=True)


if __name__=='__main__':
    run()
