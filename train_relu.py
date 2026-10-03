"""Train a ReLU rental regressor with nested chronological validation.

Run: conda run -n cs5228-hdb python train_relu.py
Predict again: conda run -n cs5228-hdb python train_relu.py --predict-only
All fitted preprocessing and IQR fences use their fitting partition only.
The 2025-01..03 holdout is evaluated once, after epoch selection on 2024-10..12.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import baseline
from data_quality import baseline_eligible, clean_rows, iqr_bounds, read_csv, remove_outliers

OUT = ROOT / 'outputs/neural_network'
SEED = 42
CATEGORICAL = ['TOWN', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL']
NUMERIC = ['area', 'lease_year', 'lease_age', 'lease_missing', 'month_index', 'month_sin', 'month_cos']


def features(rows):
    """Deterministic features; no statistics are learned here."""
    frame = pd.DataFrame(rows)
    result = frame[CATEGORICAL].replace('', '__missing__').copy()
    dates = pd.to_datetime(frame['RENT_APPROVAL_DATE'], format='%Y-%m', errors='coerce')
    result['area'] = pd.to_numeric(frame['FLOOR_AREA_SQM'], errors='coerce')
    result['lease_year'] = pd.to_numeric(frame['LEASE_COMMENCE_DATE'], errors='coerce')
    result['lease_age'] = dates.dt.year - result['lease_year']
    result['lease_missing'] = result['lease_year'].isna().astype(float)
    result['month_index'] = (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    result['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
    result[NUMERIC] = result[NUMERIC].astype(np.float32)
    return result


def prepare_training(rows):
    eligible = [r for r in rows if baseline_eligible(r)]
    bounds = iqr_bounds(eligible)
    retained = remove_outliers(eligible, bounds)
    if not retained:
        raise ValueError('No rows remain after training cleanup')
    return retained, {'input_rows': len(rows), 'invalid_key_rows': len(rows) - len(eligible),
                      'iqr_removed': len(eligible) - len(retained), 'retained_rows': len(retained),
                      'iqr_bounds': bounds}


def prepare_estimator(rows):
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                              ('scale', StandardScaler())]), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32), CATEGORICAL),
    ], sparse_threshold=1.0)
    x = preprocessing.fit_transform(features(rows)).astype(np.float32)
    y = np.array([float(r['MONTHLY_RENT']) for r in rows], dtype=np.float32)
    mean, std = float(y.mean()), float(y.std())
    if std == 0:
        raise ValueError('Constant target cannot be standardized')
    estimator = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                             learning_rate_init=0.001, alpha=0.01, batch_size=512,
                             shuffle=True, random_state=SEED, max_iter=1,
                             early_stopping=False)
    bundle = {'preprocessing': preprocessing, 'estimator': estimator,
              'target_mean': mean, 'target_std': std, 'numeric': NUMERIC,
              'categorical': CATEGORICAL, 'seed': SEED}
    return bundle, x, (y - mean) / std


def predict(bundle, rows):
    x = bundle['preprocessing'].transform(features(rows)).astype(np.float32)
    return bundle['estimator'].predict(x).astype(float) * bundle['target_std'] + bundle['target_mean']


def scores(actual, predicted):
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': float(r2_score(actual, predicted))}


def choose_epochs(rows, tune, max_epochs, patience):
    retained, cleanup = prepare_training(rows)
    bundle, x, y = prepare_estimator(retained)
    tx = bundle['preprocessing'].transform(features(tune)).astype(np.float32)
    ty = np.array([float(r['MONTHLY_RENT']) for r in tune])
    history, best, best_epoch, stale = [], float('inf'), 0, 0
    for epoch in range(1, max_epochs + 1):
        bundle['estimator'].partial_fit(x, y)
        pred = bundle['estimator'].predict(tx).astype(float) * bundle['target_std'] + bundle['target_mean']
        metric = scores(ty, pred)
        history.append({'epoch': epoch, 'training_loss_scaled': float(bundle['estimator'].loss_), **metric})
        if metric['RMSE'] < best - 0.1:
            best, best_epoch, stale = metric['RMSE'], epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch selection {epoch:3d}: tune MAE={metric['MAE']:.2f}, RMSE={metric['RMSE']:.2f}", flush=True)
        if stale >= patience:
            break
    pd.DataFrame(history).to_csv(OUT / 'epoch_selection.csv', index=False)
    print(f'Selected {best_epoch} epochs on 2024-10..12; best RMSE={best:.2f}', flush=True)
    return best_epoch, history, cleanup


def train_fixed(rows, epochs, label):
    retained, cleanup = prepare_training(rows)
    bundle, x, y = prepare_estimator(retained)
    print(f'{label}: {len(retained):,} fitting rows, {x.shape[1]} encoded features', flush=True)
    for epoch in range(1, epochs + 1):
        bundle['estimator'].partial_fit(x, y)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'{label}: epoch {epoch}/{epochs}', flush=True)
    assert bundle['estimator'].activation == 'relu'
    assert bundle['estimator'].out_activation_ == 'identity'
    bundle['epochs'] = epochs
    return bundle, cleanup


def export_predictions(bundle, test):
    pred = predict(bundle, test)
    if len(pred) != len(test) or not np.isfinite(pred).all():
        raise ValueError('Prediction count or finite-value check failed')
    if (pred <= 0).any():
        raise ValueError('Non-positive rents predicted; inspect model before export')
    template = pd.read_csv(ROOT / 'data/example-submission.csv', dtype={'Id': str})
    if len(template) != len(test):
        raise ValueError('Test/template row count mismatch')
    output = pd.DataFrame({'Id': template['Id'], 'Predicted': pred})
    output.to_csv(OUT / 'submission.csv', index=False, float_format='%.2f')
    preview = pd.DataFrame(test[:10])
    preview.insert(0, 'Id', template['Id'].iloc[:10].values)
    preview['Predicted'] = pred[:10]
    preview.to_csv(OUT / 'prediction_examples.csv', index=False, float_format='%.2f')
    return pred


def make_figures(raw, cleaned, retained, history, validation, predicted, monthly):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    source, final = pd.DataFrame(cleaned), pd.DataFrame(retained)
    for frame in (source, final):
        frame['MONTHLY_RENT'] = pd.to_numeric(frame['MONTHLY_RENT'])
        frame['FLOOR_AREA_SQM'] = pd.to_numeric(frame['FLOOR_AREA_SQM'])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    bins = np.arange(0, 8001, 200)
    axes[0, 0].hist(source['MONTHLY_RENT'], bins=bins, alpha=.5, label='Before IQR removal')
    axes[0, 0].hist(final['MONTHLY_RENT'], bins=bins, alpha=.5, label='After IQR removal')
    axes[0, 0].set(title='Training rent distribution', xlabel='Monthly rent', ylabel='Records')
    axes[0, 0].legend()
    grouped = source.groupby('RENT_APPROVAL_DATE')['MONTHLY_RENT'].median()
    axes[0, 1].plot(pd.to_datetime(grouped.index), grouped.values, color='#197c85')
    axes[0, 1].set(title='Monthly median rent before IQR removal', ylabel='Monthly rent')
    by_type = source.groupby('FLAT_TYPE')['MONTHLY_RENT'].median().sort_values()
    axes[1, 0].barh(by_type.index, by_type.values, color='#197c85')
    axes[1, 0].set(title='Median rent by flat type', xlabel='Monthly rent')
    sample = source.sample(n=min(5000, len(source)), random_state=SEED)
    axes[1, 1].scatter(sample['FLOOR_AREA_SQM'], sample['MONTHLY_RENT'], s=5, alpha=.15)
    axes[1, 1].set(title='Area vs rent (5,000 sampled records)', xlabel='Area (square metres)', ylabel='Monthly rent')
    fig.savefig(OUT / 'eda_overview.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot([r['epoch'] for r in history], [r['RMSE'] for r in history], color='#197c85')
    axes[0].set(title='Epoch selection: 2024 Q4', xlabel='Epoch', ylabel='RMSE')
    actual = np.array([float(r['MONTHLY_RENT']) for r in validation])
    axes[1].scatter(actual, predicted, s=6, alpha=.12)
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[1].plot([low, high], [low, high], '--', color='black', linewidth=1)
    axes[1].set(title='Untouched holdout: 2025 Q1', xlabel='Actual rent', ylabel='Predicted rent')
    axes[2].bar([r['month'] for r in monthly], [r['MAE'] for r in monthly], color='#197c85')
    axes[2].set(title='Holdout error by month', ylabel='MAE')
    fig.savefig(OUT / 'model_evaluation.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        parser.error('Epoch count and patience must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    test, _ = clean_rows(read_csv(ROOT / 'data/test.csv', training=False))
    if args.predict_only:
        # Load only the locally generated, trusted model artifact.
        bundle = joblib.load(OUT / 'relu_model.joblib')
        with threadpool_limits(limits=2):
            pred = export_predictions(bundle, test)
        print(f'Regenerated {len(pred):,} predictions from saved model.')
        return
    raw = read_csv(ROOT / 'data/train.csv')
    cleaned, issues = clean_rows(raw)
    eligible = [r for r in cleaned if baseline_eligible(r)]
    selection_fit = [r for r in eligible if r['RENT_APPROVAL_DATE'] < '2024-10']
    selection_valid = [r for r in eligible if '2024-10' <= r['RENT_APPROVAL_DATE'] < '2025-01']
    outer_fit = [r for r in eligible if r['RENT_APPROVAL_DATE'] < '2025-01']
    holdout = [r for r in eligible if r['RENT_APPROVAL_DATE'] >= '2025-01']
    if not all((selection_fit, selection_valid, outer_fit, holdout)):
        raise ValueError('Expected chronological partitions are empty')
    print(f'Epoch selection: {len(selection_fit):,} earlier rows / {len(selection_valid):,} tuning rows', flush=True)
    print(f'Outer holdout: {len(holdout):,} rows; no IQR removal from evaluation data', flush=True)
    with threadpool_limits(limits=2):
        epochs, history, selection_cleanup = choose_epochs(selection_fit, selection_valid, args.max_epochs, args.patience)
        validation_bundle, validation_cleanup = train_fixed(outer_fit, epochs, 'Holdout model')
        validation_pred = predict(validation_bundle, holdout)
        actual = np.array([float(r['MONTHLY_RENT']) for r in holdout])
        nn_scores = scores(actual, validation_pred)
        baseline_rows, _ = prepare_training(outer_fit)
        baseline_model = baseline.fit(baseline_rows)
        baseline_pred = np.array([baseline.predict(baseline_model, r) for r in holdout])
        baseline_scores = scores(actual, baseline_pred)
        print(f'ReLU holdout: {nn_scores}\nBaseline holdout: {baseline_scores}', flush=True)
        validation_output = pd.DataFrame(holdout)
        validation_output.insert(0, 'csv_row', [i for i, r in enumerate(cleaned, 2)
                                              if baseline_eligible(r) and r['RENT_APPROVAL_DATE'] >= '2025-01'])
        validation_output['Predicted'] = validation_pred
        validation_output['BaselinePredicted'] = baseline_pred
        validation_output['AbsoluteError'] = np.abs(actual - validation_pred)
        validation_output.to_csv(OUT / 'validation_predictions.csv', index=False, float_format='%.2f')
        monthly = []
        for month in sorted({r['RENT_APPROVAL_DATE'] for r in holdout}):
            mask = np.array([r['RENT_APPROVAL_DATE'] == month for r in holdout])
            monthly.append({'month': month, 'rows': int(mask.sum()), **scores(actual[mask], validation_pred[mask])})
        pd.DataFrame(monthly).to_csv(OUT / 'validation_by_month.csv', index=False)
        bundle, final_cleanup = train_fixed(eligible, epochs, 'Final model')
        bundle['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__}
        bundle['training_end'] = max(r['RENT_APPROVAL_DATE'] for r in eligible)
        joblib.dump(bundle, OUT / 'relu_model.joblib', compress=3)
        pred = export_predictions(bundle, test)
        # Ensure the persisted preprocessing/model reproduces all test predictions.
        reloaded = joblib.load(OUT / 'relu_model.joblib')
        np.testing.assert_allclose(predict(reloaded, test), pred, rtol=0, atol=0)
    metrics = {
        'seed': SEED, 'architecture': [128, 64, 1], 'hidden_activation': 'relu', 'output_activation': 'identity',
        'optimizer': 'adam', 'loss': 'squared_error_on_standardized_rent', 'epochs': epochs,
        'numeric_features': NUMERIC, 'categorical_features': CATEGORICAL,
        'excluded_features': ['FURNISHED (constant)', 'FEE (constant)', 'BLOCK (ambiguous without street)'],
        'auxiliary_data_used': False,
        'raw_rows': len(raw), 'cell_issues': len(issues), 'invalid_key_rows': len(raw) - len(eligible),
        'selection_fit': selection_cleanup, 'selection_validation_range': ['2024-10', '2024-12'],
        'holdout_fit': validation_cleanup, 'holdout_range': ['2025-01', '2025-03'],
        'holdout_rows': len(holdout), 'relu': nn_scores, 'baseline': baseline_scores,
        'holdout_by_month': monthly, 'final_fit': final_cleanup, 'test_rows': len(test),
        'test_prediction_summary': {'min': float(pred.min()), 'median': float(np.median(pred)),
                                    'mean': float(pred.mean()), 'max': float(pred.max())},
        'versions': bundle['versions'],
        'input_sha256': {name: hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest()
                         for name in ('train.csv', 'test.csv', 'example-submission.csv')},
        'seconds': time.monotonic() - start,
    }
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    retained, _ = prepare_training(eligible)
    make_figures(raw, cleaned, retained, history, holdout, validation_pred, monthly)
    lines = [
        'ReLU 神经网络租金预测结果',
        f'原始训练数据 {len(raw):,} 行；最终删除 IQR 异常 {final_cleanup["iqr_removed"]:,} 行；使用 {len(retained):,} 行。',
        '统一街道和户型写法；矛盾地契年份设为空；数值缺失用拟合部分中位数填补。',
        '类别使用独热编码，未知类别忽略；数值与目标标准化，预测后还原到原始租金单位。',
        '网络：编码后的输入 → Dense(128) + ReLU → Dense(64) + ReLU → Dense(1) 线性输出。',
        f'优化器 Adam，学习率 0.001，batch size 512；固定 seed 42；选定 {epochs} 个 epoch。',
        '2024-10 至 2024-12 仅用于选择 epoch；2025-01 至 2025-03 作为外层验证，验证记录不删异常值。',
        f'ReLU：MAE {nn_scores["MAE"]:.2f}；RMSE {nn_scores["RMSE"]:.2f}；R² {nn_scores["R2"]:.4f}',
        f'同划分 baseline：MAE {baseline_scores["MAE"]:.2f}；RMSE {baseline_scores["RMSE"]:.2f}；R² {baseline_scores["R2"]:.4f}',
        '最后按选定 epoch 在全量清洗训练集重训，再预测所有测试行。',
        f'输出 submission.csv：{len(test):,} 行，列名 Id,Predicted，与提交模板顺序一致。',
        'test.csv 没有真实租金，因此测试集误差未知；上述指标来自历史验证集，不是测试集成绩。',
        '局限：尚未加入辅助地理数据；未来租金变化未必延续历史规律；一次时间留出不能保证较远未来精度。',
        '原始 CSV 未修改。模型和预处理保存在 relu_model.joblib；metrics.json 记录输入哈希与软件版本。',
    ]
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Completed in {time.monotonic() - start:.1f}s. Outputs: {OUT}', flush=True)


if __name__ == '__main__':
    main()
