"""Verify the matched-origin predictions against source rows and branch scores."""
from __future__ import annotations
import io
import json
import subprocess
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import ROOT, OUT, frame_data, score, write_json
from analysis.experiments.frontier_matched import MATCHED, MODELS
from analysis.pipeline.data import sha256


def main():
    source, hashes = frame_data()
    expected = source.loc[(source.RENT_APPROVAL_DATE >= '2024-06') & (source.RENT_APPROVAL_DATE < '2025-04')]
    selection = json.loads((OUT / 'ensemble_selection.json').read_text())
    combined = np.zeros(len(expected))
    checks = []
    for name, component in selection['members'].items():
        predictions = pd.read_csv(MODELS / name / 'predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(predictions.source_row, expected.index)
        np.testing.assert_array_equal(predictions.actual, expected.MONTHLY_RENT)
        np.testing.assert_array_equal(predictions.month, expected.RENT_APPROVAL_DATE)
        if not np.isfinite(predictions.prediction).all():
            raise AssertionError('Nonfinite prediction')
        result = json.loads((MATCHED / name / 'results.json').read_text())
        if result['source_sha256'] != hashes:
            raise AssertionError('Source changed')
        recomputed = score(predictions.actual, predictions.prediction)
        np.testing.assert_allclose(recomputed['rmse_sgd'], result['rmse_sgd'], atol=1e-8, rtol=0)
        combined += component['weight'] * predictions.prediction.to_numpy()
        checks.append({'name': name, 'rows': len(predictions), **recomputed})
    stored = pd.read_csv(MODELS / 'ensemble_predictions.csv').sort_values('source_row')
    np.testing.assert_allclose(stored.prediction, combined, rtol=0, atol=1e-8)
    result = json.loads((MATCHED / 'summary.json').read_text())
    np.testing.assert_allclose(result['scores']['rmse_sgd'], score(expected.MONTHLY_RENT, combined)['rmse_sgd'], atol=1e-8, rtol=0)
    # Verify the numerical baseline against its committed branch artifact.
    branch = 'origin/experiment/tabpfn'
    blob = subprocess.check_output(['git', 'show', branch + ':reports/tabpfn_components_80_20/scores.csv'], cwd=ROOT)
    table = pd.read_csv(io.BytesIO(blob))
    reference = float(table.loc[table.model.eq('diag_mse_only'), 'rmse_sgd'].iloc[0])
    np.testing.assert_allclose(reference, result['prior_tabpfn_best_development_rmse_sgd'], atol=1e-8, rtol=0)
    reference_commit = subprocess.check_output(['git', 'rev-parse', branch], cwd=ROOT, text=True).strip()
    recent = json.loads((MATCHED / 'recent_calibration_results.json').read_text()) if (MATCHED / 'recent_calibration_results.json').exists() else {}
    recent_checks = []
    for name, value in recent.items():
        predictions = pd.read_csv(MODELS / f'{name}_predictions.csv').sort_values('source_row')
        np.testing.assert_array_equal(predictions.source_row, expected.index)
        np.testing.assert_array_equal(predictions.actual, expected.MONTHLY_RENT)
        np.testing.assert_allclose(score(predictions.actual, predictions.prediction)['rmse_sgd'],
            value['scores']['rmse_sgd'], atol=1e-8, rtol=0)
        reconstructed = np.zeros(len(expected))
        for member, weight in value['selection']['weights'].items():
            component = pd.read_csv(MODELS / member / 'predictions.csv').sort_values('source_row')
            reconstructed += weight*component.prediction.to_numpy()
        np.testing.assert_allclose(predictions.prediction, reconstructed, atol=1e-8, rtol=0)
        recent_checks.append({'variant': name, **value['scores']})
    verify = {'rows': len(expected), 'source_labels_and_ids_match': True,
        'component_weights_match_locked_selection': True, 'all_component_scores_independently_recomputed': True,
        'sources_unchanged': True, 'components': checks, 'branch_reference_commit': reference_commit,
        'branch_reference_rmse_sgd': reference, 'recent_calibration_checks': recent_checks,
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'analysis/experiments').glob('frontier_*.py'))}}
    write_json(MATCHED / 'verification.json', verify)
    delta = result['scores']['rmse_sgd'] - reference
    lines = ['# 与TabPFN相同六月起点的对照', '',
        f"固定起点 **2024-06**，训练信息截至 **2024-05**；验证 **2024-06 至 2025-03** 的全部 **{len(expected):,}** 条记录。", '',
        '| 管线 | RMSE SGD | MAE SGD |', '| --- | ---: | ---: |',
        f"| 本轮锁定融合 | **{result['scores']['rmse_sgd']:.4f}** | {result['scores']['mae_sgd']:.4f} |",
        '| 原TabPFN冻结版，分支记录 | 455.8136 | 329.7485 |',
        f'| 原TabPFN MSE-only DiagScale，分支记录 | {reference:.4f} | 330.7898 |']
    lines += [f"| {name} | {value['scores']['rmse_sgd']:.4f} | {value['scores']['mae_sgd']:.4f} |" for name, value in recent.items()]
    lines += ['',
        f"相对原TabPFN最低开发记录的差值（新−旧）为 **{delta:+.4f} SGD**。400–430目标：**{'达到' if result['target_430_achieved'] else '未达到'}**。", '',
        '本次保持此前较早窗口锁定的12个成员、所有轮数和融合权重，不在六月起点的验证标签上重新选择。各成员重新用截止五月的可用训练数据拟合。',
        '数据可用总量为121,005行；部分管线使用历史预热而少用早期拟合行，详见 component_metrics.csv。',
        'TabPFN分数取自已提交分支的 scores.csv，本次没有重跑TabPFN。因此这是相同源数据/起点/验证行的完整管线对照，架构和特征不同。',
        '额外两组融合仅用2023-10至2024-05的24,140条折外预测重新选权：十月至三月从九月底发出，四月至五月从三月底发出；选权函数不读取六月及之后的预测或标签。均匀八个月与三个月半衰期两条规则均在新融合评分前锁定。',
        '本窗口此前已被多个项目实验查看，所有分数均为开发证据，不能当作新独立测试或Kaggle成绩。', '',
        '## 成员结果', '', '| 成员 | 权重 | RMSE SGD |', '| --- | ---: | ---: |']
    for component in result['components']:
        lines.append(f"| {component['name']} | {component['weight']:.2%} | {component['rmse_sgd']:.4f} |")
    lines += ['', '## 复现', '', '```powershell',
        'python -m analysis.experiments.frontier_matched',
        'python -m analysis.experiments.frontier_june_calibration select',
        'python -m analysis.experiments.frontier_june_calibration evaluate',
        'python -m analysis.experiments.frontier_matched_report', '```', '',
        'summary.json 保存汇总，verification.json 保存逐行源标签/月份/ID/分数重算检查及原分支提交SHA。',
        '模型、预处理和预测保存在 output/models/round_01_history_target_search/matched_june/。']
    (MATCHED / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'scores': result['scores'], 'delta_vs_prior_best': delta, 'verification_passed': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
