"""Independently verify saved frontier predictions and write a concise report."""
from __future__ import annotations
import gc
import json
import platform
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import (ROOT, OUT, ART, DATA, prepare,
    predict_saved, score, write_json, summarize, frame_data)
from analysis.experiments.frontier_compact import prepare_compact
from analysis.pipeline.data import sha256


def main():
    frame, hashes = frame_data()
    checked, restore = [], []
    paths = list((OUT / 'screen').glob('*/results.json')) + list((OUT / 'outer').glob('*/results.json'))
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        stage, name = record['stage'], record['spec']['name']
        artifact = ART / stage / name
        predictions = pd.read_csv(artifact / 'predictions.csv')
        rows = predictions.source_row.to_numpy(int)
        np.testing.assert_array_equal(predictions.actual, frame.iloc[rows]["MONTHLY_RENT"])
        np.testing.assert_array_equal(predictions.month, frame.iloc[rows]['RENT_APPROVAL_DATE'])
        if not predictions.source_row.is_unique or not np.isfinite(predictions.prediction).all():
            raise AssertionError('Invalid saved predictions')
        recomputed = score(predictions.actual, predictions.prediction)
        for key in recomputed:
            np.testing.assert_allclose(recomputed[key], record[key], rtol=0, atol=1e-8)
        if record['source_sha256'] != hashes:
            raise AssertionError('Source fingerprint changed')
        saved = joblib.load(artifact / 'model.joblib')
        if 'model' in saved:
            model = saved['model']
            columns = list(model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_)
            record['n_features'] = len(columns)
            if 'feature_columns' in saved and saved['feature_columns'] != columns:
                saved['feature_columns'] = columns
                joblib.dump(saved, artifact / 'model.joblib', compress=1)
            write_json(path, record)
        checked.append({'stage': stage, 'name': name, 'n': len(predictions), **recomputed})
        del saved
        gc.collect()
    # Reload every new outer component, using its stored fitting feature schema.
    for path in (OUT / 'outer').glob('*/results.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        spec, name = record['spec'], path.parent.name
        if spec.get('family') == 'controlled_lgb':
            bundle = prepare_compact(record['start'], record['end'])
        else:
            bundle = prepare(record['start'], record['end'], spec['snapshot'],
                spec.get('neighbors', False), spec.get('damping', .85))
        saved = joblib.load(ART / 'outer' / name / 'model.joblib')
        predicted = predict_saved(saved, bundle)
        expected = pd.read_csv(ART / 'outer' / name / 'predictions.csv')
        error = float(np.max(np.abs(predicted - expected.prediction.to_numpy())))
        if error > 1e-6:
            raise AssertionError(f'{name} reload mismatch {error}')
        bundle['y_val'] = np.full(len(bundle['y_val']), -999999.)
        without_labels = predict_saved(saved, bundle)
        np.testing.assert_array_equal(without_labels, predicted)
        restore.append({'name': name, 'max_reload_difference_sgd': error,
            'query_label_mutation_changes_predictions': False})
        del bundle, saved
        gc.collect()
    summarize()
    old = pd.read_csv(ROOT / 'output/experiment_runs/comparison_model_search/2024-04/inner_blend/predictions.csv').sort_values('source_row')
    new = pd.read_csv(ART / 'outer/locked_ensemble/predictions.csv').sort_values('source_row')
    np.testing.assert_array_equal(old.source_row, new.source_row)
    np.testing.assert_array_equal(old.actual, new.actual)
    comparison = pd.DataFrame({'month': old.approval_month.to_numpy(), 'actual': old.actual.to_numpy(),
        'old': old.prediction.to_numpy(), 'new': new.prediction.to_numpy()})
    monthly = []
    for month, part in comparison.groupby('month'):
        monthly.append({'month': month, 'n': len(part),
            'old_rmse': score(part.actual, part.old)['rmse_sgd'],
            'new_rmse': score(part.actual, part.new)['rmse_sgd'],
            'old_sse': float(np.square(part.old-part.actual).sum()),
            'new_sse': float(np.square(part.new-part.actual).sum())})
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(OUT / 'ensemble_monthly_comparison.csv', index=False)
    generator = np.random.default_rng(42)
    starts = generator.integers(0, len(monthly)-2, size=(5000, 4))
    indices = (starts[:, :, None] + np.arange(3)).reshape(5000, 12)
    n = monthly.n.to_numpy()[indices].sum(axis=1)
    delta = np.sqrt(monthly.new_sse.to_numpy()[indices].sum(axis=1)/n) - np.sqrt(
        monthly.old_sse.to_numpy()[indices].sum(axis=1)/n)
    old_score, new_score = score(old.actual, old.prediction), score(new.actual, new.prediction)
    control_path = OUT / 'old_only_control.json'
    control = json.loads(control_path.read_text()) if control_path.exists() else None
    if control:
        control_prediction = np.zeros(len(old))
        for family, weight in control['weights'].items():
            component = pd.read_csv(ROOT / 'output/experiment_runs/comparison_model_search/2024-04' / family / 'predictions.csv').sort_values('source_row')
            np.testing.assert_array_equal(component.source_row, old.source_row)
            np.testing.assert_array_equal(component.actual, old.actual)
            control_prediction += weight*component.prediction.to_numpy()
        np.testing.assert_allclose(score(old.actual, control_prediction)['rmse_sgd'], control['outer_scores']['rmse_sgd'], atol=1e-8, rtol=0)
    results = pd.read_csv(OUT / 'metrics.csv')
    outer = results.loc[results.stage.eq('outer')].sort_values('rmse_sgd')
    summary = {'new_configurations_screened': int((results.stage == 'screen').sum()),
        'new_configurations_refit_outer': len(outer), 'old_local_baseline': old_score,
        'locked_ensemble': new_score, 'delta_rmse_sgd': new_score['rmse_sgd'] - old_score['rmse_sgd'],
        'month_block_delta_interval_95': np.quantile(delta, [.025, .975]).tolist(),
        'best_new_single_pipeline': outer.iloc[0].to_dict(),
        'target_430_achieved': bool(min(new_score['rmse_sgd'], float(outer.rmse_sgd.min())) <= 430.),
        'outer_window': ['2024-04', '2025-03'], 'outer_rows': len(new),
        'source_sha256': hashes,
        'limitations': ['These months were examined by earlier project experiments; development evidence, not an untouched final test.',
            'One seed and one earlier six-month selection window; sampling intervals do not cover model-selection or future-market uncertainty.',
            'Full distribution pipelines omit six early warmup months; controlled 27-column additions retain all fitting rows.',
            'Static supplied amenities are snapshots; no future target-month public rents or future realized macro observations are used.']}
    matched_path = OUT / 'matched_june/summary.json'
    matched = json.loads(matched_path.read_text()) if matched_path.exists() else None
    if matched:
        summary['matched_june'] = {key: matched[key] for key in ('scores', 'validation_rows',
            'prior_tabpfn_best_development_rmse_sgd', 'delta_vs_prior_tabpfn_best', 'target_430_achieved')}
    if control:
        summary['old_only_convex_control'] = control['outer_scores']
        summary['delta_vs_old_only_convex_control'] = new_score['rmse_sgd'] - control['outer_scores']['rmse_sgd']
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'verification.json', {'prediction_files_checked': len(checked),
        'checks': checked, 'model_reload_checks': restore, 'sources_unchanged': True,
        'python': platform.python_version(),
        'versions': {name: version(name) for name in ('numpy','pandas','scikit-learn','lightgbm','catboost')},
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'analysis/experiments').glob('frontier_*.py'))}})
    ensemble = json.loads((OUT / 'ensemble_selection.json').read_text())
    lines = ['# 新一轮租金预测探索', '',
        f"本轮筛选 **{summary['new_configurations_screened']} 个新配置**，并按较早窗口的选择对 {len(outer)} 个配置做年度重训。",
        f"原本地融合 RMSE **{old_score['rmse_sgd']:.4f}**；本轮预先锁定融合 **{new_score['rmse_sgd']:.4f}**。",
        f"差值（新−旧）为 **{summary['delta_rmse_sgd']:+.4f} SGD**。目标 400–430：**{'达到' if summary['target_430_achieved'] else '未达到'}**。", '',
        '## 验证协议', '',
        '- 筛选：2023-10 至 2024-03，17,977 行，信息截至 2023-09；用于参数、轮数、特征方案和融合权重选择。',
        '- 年度验证：2024-04 至 2025-03，35,158 行，信息截至 2024-03；所有验证行均保留。',
        '- 新模型外层训练使用内部选定的固定轮数，不在年度窗口早停；融合权重在本轮年度评分之前锁定。',
        '- 所有历史标签只来自查询信息截止点之前；跨度模拟使用 0/5/11 个月延迟。',
        '- 全分布模型有六个月历史预热；受控27列实验保留全部拟合行。因此跨两类的变化不能全归因于架构或单个特征。',
        '- 原项目此前已反复查看年度验证期；本轮仍属于开发证据，不是新独立测试，也不是 Kaggle 分数。', '',
        '## 探索的模型与特征', '',
        '| 路线 | 内容 |', '| --- | --- |',
        '| 分布历史 | 楼栋/街道/规划区/Town/房型层次的均价、标准差、相对同月同户型均价、低/高租金比例、最近均价、6/12月统计 |',
        '| 市场目标分解 | 直接租金、相对历史市场基准的残差、相对市场基准比例；比例模型按基准平方加权，保持SGD平方损失目标 |',
        '| 跨度模拟 | 同一训练行生成0/5/11个月信息延迟，匹配冻结起点的多月预测场景 |',
        '| 空间可比 | 同户型、24个月内的近邻，按位置/面积/租约年份与时间衰减得到16/48/128近邻统计 |',
        '| 分层预测 | 先预测低/普通/高租金概率，再合并各层条件均值；查询租金不参与推理 |',
        '| 稳定性 | 6/12/24月半衰期加权；仅相对价格特征；去掉高基数地址类别；15/31/63叶树 |',
        '| 受控特征添加 | 固定原27列，只加楼栋分布、各层分布、相对历史或可比房源统计 |', '',
        '## 年度验证结果', '', '| 配置 | 特征数 | RMSE | MAE |', '| --- | ---: | ---: | ---: |']
    for row in outer.itertuples():
        lines.append(f'| {row.name} | {row.n_features} | {row.rmse_sgd:.4f} | {row.mae_sgd:.4f} |')
    lines += [f"| 本轮锁定融合 | 多模型 | {new_score['rmse_sgd']:.4f} | {new_score['mae_sgd']:.4f} |",
        f"| 原本地融合 | 多模型 | {old_score['rmse_sgd']:.4f} | {old_score['mae_sgd']:.4f} |", '',
        '## 融合与结果解释', '',
        '融合权重只使用较早的17,977行预测；非负、和为1、单成员最高0.6，不增加常数偏移。完整候选与权重在 `ensemble_selection.json`。', '',
        '| 成员 | 权重 |', '| --- | ---: |']
    lines += [f"| {name} | {member['weight']:.4%} |" for name, member in ensemble['members'].items()]
    interval = summary['month_block_delta_interval_95']
    lines += ['', f"对固定预测做三个月连续区块重采样5,000次，新−旧RMSE差值的描述性95%区间为 [{interval[0]:+.3f}, {interval[1]:+.3f}]。它仅反映这12个月的敏感性。", '',
        '本轮没有发现支持400–430目标的结果。低租金分层、更多历史分布或跨度模拟本身都不保证改善；小幅开发分数差异不能保证未来收益。',
        '原始街道大小写的训练期审计也未出现明显分层信号，未将格式当作新主线。', '',
        '## 文件与复现', '',
        '- `metrics.csv`：全部筛选与年度配置；`summary.json`：主要结论。',
        '- `verification.json`：独立重算、标签/行序/源文件哈希、模型重载和查询标签变更检查。',
        '- `.cache/round_01_history_target_search/`：可再生特征；`output/models/round_01_history_target_search/`：本地模型与逐行预测。',
        '- 原始CSV保持不变，所有本轮新特征均未读取公开源的目标月份租金；本轮未生成或上传比赛提交。', '',
        '```powershell',
        'python -m unittest tests.test_frontier_search -v',
        'python -m analysis.experiments.frontier_search screen',
        'python -m analysis.experiments.frontier_refinement screen',
        'python -m analysis.experiments.frontier_mixture screen',
        'python -m analysis.experiments.frontier_compact screen',
        'python -m analysis.experiments.frontier_ensemble select',
        'python -m analysis.experiments.frontier_ensemble evaluate',
        'python -m analysis.experiments.frontier_report',
        '```', '', '环境使用已有 cs5228 Conda 环境，不需要新增依赖。']
    if control:
        lines += ['', '## 仅旧模型的融合对照', '',
            f"按相同的较早窗口优化、相同非负/权重上限规则，仅使用旧模型得到年度RMSE **{control['outer_scores']['rmse_sgd']:.4f}**。加入新模型的融合为 **{new_score['rmse_sgd']:.4f}**，进一步降低 **{control['outer_scores']['rmse_sgd']-new_score['rmse_sgd']:.4f} SGD**。",
            '这控制了将原来三个成员扩展为更多旧模型的影响，但不能将收益归因于某一个新特征。该补充对照在主年度结果之后添加，权重优化仍只读取较早窗口标签。']
    if matched:
        lines += ['', '## 与此前TabPFN最低分的同起点对照', '',
            f"额外按2024-06起点重训全部12个成员，在相同28,995行上的融合RMSE为 **{matched['scores']['rmse_sgd']:.4f}**；此前TabPFN最低记录为 **{matched['prior_tabpfn_best_development_rmse_sgd']:.4f}**。本轮没有刷新该记录。",
            '两组只用六月以前标签重新选权的融合也未改善；完整结果、成员和复现命令见 [六月对照](matched_june/README.md)。',
            '本轮一年期的改善不能被写作整体最佳模型刷新；两个验证窗口必须分别报告。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
