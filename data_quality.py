"""Audit and conservatively clean train.csv using Python's standard library.

Run: python3 data_quality.py
Original data is never overwritten. CSV row numbers include the header (first
data row = 2). Rows outside either 1.5-IQR range are removed from training.
Repeated records within the ranges are retained. Targets are never clipped.
"""
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ['RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE',
          'FLAT_MODEL', 'FLOOR_AREA_SQM', 'FURNISHED', 'LEASE_COMMENCE_DATE',
          'FEE', 'MONTHLY_RENT']
TEXT = {'TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FURNISHED'}
NUMERIC = {'FLOOR_AREA_SQM', 'LEASE_COMMENCE_DATE', 'FEE', 'MONTHLY_RENT'}
MISSING = {'', 'na', 'n/a', 'nan', 'null', 'none', '?', '-', 'unknown'}


def read_csv(path, training=True):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        expected = FIELDS if training else FIELDS[:-1]
        if reader.fieldnames != expected:
            raise ValueError(f'Unexpected columns in {path}: {reader.fieldnames}')
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f'Malformed CSV row in {path}; fix column count first')
    return rows


def clean_rows(rows):
    """Apply fixed rules only; return cleaned copies and traceable cell issues."""
    cleaned, issues = [], []
    for line, original in enumerate(rows, 2):
        row = {}

        def issue(field, kind, action, detail, value):
            issues.append(dict(csv_row=line, field=field, issue=kind,
                               original=original[field], cleaned=value,
                               action=action, detail=detail))

        for field, raw in original.items():
            value = raw.strip()
            if value.lower() in MISSING:
                issue(field, 'missing', 'set_empty', 'Blank or missing-value marker', '')
                row[field] = ''
                continue
            if field in TEXT:
                value = ' '.join(value.lower().split())
            if field == 'FLAT_TYPE':
                match = re.fullmatch(r'([1-5])\s*-?\s*room', value)
                if match:
                    value = match[1] + '-room'
                elif value != 'executive':
                    issue(field, 'unrecognized_category', 'retain_review',
                          'Outside observed baseline vocabulary; do not guess replacement', value)
            invalid = None
            if field == 'RENT_APPROVAL_DATE':
                if not re.fullmatch(r'[0-9]{4}-(0[1-9]|1[0-2])', value) or value[:4] == '0000':
                    invalid = 'Expected valid YYYY-MM'
            if field in NUMERIC:
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        invalid = 'Non-finite numeric value'
                    elif field in {'MONTHLY_RENT', 'FLOOR_AREA_SQM'} and number <= 0:
                        invalid = 'Expected positive value'
                    elif field == 'FEE' and number < 0:
                        # Definition of FEE is unavailable: flag, do not assume invalid.
                        issue(field, 'negative_fee', 'retain_review', 'Confirm FEE definition', value)
                    elif field == 'LEASE_COMMENCE_DATE' and (not number.is_integer() or not 1 <= number <= 9999):
                        invalid = 'Expected integer calendar year'
                except ValueError:
                    invalid = 'Cannot parse numeric value'
            if invalid:
                issue(field, 'invalid_value', 'set_empty', invalid, '')
                value = ''
            elif value != raw:
                issue(field, 'format', 'normalize', 'Whitespace/case/flat-type normalization', value)
            row[field] = value
        if row.get('LEASE_COMMENCE_DATE') and row.get('RENT_APPROVAL_DATE'):
            if float(row['LEASE_COMMENCE_DATE']) > int(row['RENT_APPROVAL_DATE'][:4]):
                issue('LEASE_COMMENCE_DATE', 'temporal_inconsistency', 'set_empty',
                      'Lease start year exceeds rental approval year; correct year unknown', '')
                row['LEASE_COMMENCE_DATE'] = ''
        cleaned.append(row)
    return cleaned, issues


def baseline_eligible(row):
    # Unused-feature defects must not remove otherwise usable training labels.
    return all(row.get(k) for k in ('RENT_APPROVAL_DATE', 'TOWN', 'FLAT_TYPE', 'MONTHLY_RENT'))


def quantile(values, p):
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def iqr_bounds(rows):
    """Fit once on the training partition, before any outlier deletion."""
    bounds = {}
    for field in ('FLOOR_AREA_SQM', 'MONTHLY_RENT'):
        values = sorted(float(r[field]) for r in rows if r.get(field))
        if not values:
            raise ValueError(f'No valid values to fit IQR for {field}')
        q1, q3 = quantile(values, .25), quantile(values, .75)
        bounds[field] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    return bounds


def outlier_fields(row, bounds):
    return [field for field, (low, high) in bounds.items()
            if row.get(field) and not low <= float(row[field]) <= high]


def remove_outliers(rows, bounds):
    """Drop the union of flagged rows; endpoints are retained; no iteration."""
    return [row for row in rows if not outlier_fields(row, bounds)]


def audit(raw, clean, issues):
    profile = {}
    for field in FIELDS:
        values = [row[field] for row in raw]
        profile[field] = {
            'blank': sum(not x.strip() for x in values),
            'missing_marker': sum(bool(x.strip()) and x.strip().lower() in MISSING for x in values),
            'unique_raw': len(set(values)),
            'unique_clean_nonempty': len({r[field] for r in clean if r[field]}),
            'changed_cells': sum(a[field] != b[field] for a, b in zip(raw, clean)),
            'empty_after_cleaning': sum(not r[field] for r in clean),
        }
        if field in NUMERIC:
            numbers = sorted(float(r[field]) for r in clean if r[field])
            if numbers:
                profile[field]['numeric'] = {str(p): quantile(numbers, p) for p in (0, .01, .25, .5, .75, .99, 1)}
    duplicate_stats = {}
    for label, rows in [('raw', raw), ('clean', clean)]:
        counts = Counter(tuple(r[k] for k in FIELDS) for r in rows)
        duplicate_stats[label] = dict(groups=sum(n > 1 for n in counts.values()),
                                     rows_in_groups=sum(n for n in counts.values() if n > 1),
                                     excess_rows=sum(n - 1 for n in counts.values()))
    seen = {}
    for line, row in enumerate(clean, 2):
        key = tuple(row[k] for k in FIELDS)
        if key in seen:
            issues.append(dict(csv_row=line, field='ALL', issue='repeated_record',
                               original='', cleaned='', action='retain_review',
                               detail=f'Same cleaned values as CSV row {seen[key]}; no unit identifier'))
        else:
            seen[key] = line
    feature_groups = defaultdict(list)
    for row in clean:
        feature_groups[tuple(row[k] for k in FIELDS[:-1])].append(row['MONTHLY_RENT'])
    conflicts = [v for v in feature_groups.values() if len(set(v)) > 1]
    outliers = {}
    # Whole-train bounds define the exported final training set.
    # Validation training fits separate bounds using only its earlier partition.
    for field in ('FLOOR_AREA_SQM', 'MONTHLY_RENT'):
        stats = profile[field].get('numeric')
        if not stats:
            continue
        q1, q3 = stats['0.25'], stats['0.75']
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flagged = [(line, row) for line, row in enumerate(clean, 2)
                   if row[field] and not low <= float(row[field]) <= high]
        outliers[field] = dict(lower=low, upper=high, count=len(flagged), rule='1.5 IQR; drop row')
        for line, row in flagged:
            issues.append(dict(csv_row=line, field=field, issue='iqr_outlier',
                               original=raw[line - 2][field], cleaned=row[field],
                               action='drop_row', detail=f'Outside [{low}, {high}]; user-selected exclusion rule'))
    return dict(rows=len(raw), profile=profile, duplicates=duplicate_stats,
                same_features_different_rent=dict(groups=len(conflicts), rows=sum(map(len, conflicts))),
                outliers=outliers,
                issue_counts=dict(Counter(x['issue'] for x in issues)),
                changed_rows=sum(a != b for a, b in zip(raw, clean)),
                baseline_excluded_csv_rows=[i for i, r in enumerate(clean, 2) if not baseline_eligible(r)])


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = ROOT / 'data/train.csv'
    raw = read_csv(source)
    clean, issues = clean_rows(raw)
    report = audit(raw, clean, issues)
    bounds = iqr_bounds(clean)
    retained = remove_outliers(clean, bounds)
    removed_lines = [line for line, row in enumerate(clean, 2) if outlier_fields(row, bounds)]
    report['outlier_removal'] = dict(
        removed_rows=len(removed_lines), retained_rows=len(retained),
        overlap_rows=sum(len(outlier_fields(row, bounds)) == 2 for row in clean),
        removed_csv_rows=removed_lines)
    report['profile_scope'] = 'After cell normalization, before outlier row removal'
    report['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = ROOT / 'outputs/data_quality'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'train_clean.csv', retained, FIELDS)
    write_csv(output / 'issues.csv', issues,
              ['csv_row', 'field', 'issue', 'original', 'cleaned', 'action', 'detail'])
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['训练数据质量分析', '数据源：data/train.csv（原文件保留）',
             f'总记录：{len(raw):,}；列数：{len(FIELDS)}；删除异常记录：{len(removed_lines):,}；清洗后记录：{len(retained):,}',
             '定位：issues.csv 的 csv_row 包含表头，第一条数据是第 2 行。', '',
             '以下字段统计、重复统计及数值范围均为单元格规范化后、删除异常行之前。',
             '字段检查（原始空白 / 原始缺失标记 / 修改单元格 / 规范化后空值）：']
    for field, p in report['profile'].items():
        lines.append(f"{field}: {p['blank']} / {p['missing_marker']} / {p['changed_cells']} / {p['empty_after_cleaning']}")
    lines += ['', '处理规则与解释：',
              '1. 统一文字大小写和空白；将 4 room 等统一为 4-room。楼栋编号保留为文本。',
              '2. 空白及 NA/N/A/NaN/null/None/?/-/unknown 识别为缺失；0 不是缺失。',
              '3. 非法日期、非有限/无法解析数字、非正面积/租金、非法地契年份设为空并记日志。',
              '4. 地契年份晚于出租年份属于时间矛盾：无法判断哪个源字段错，将地契年份暂设为空，保留整行和租金。',
              '5. FURNISHED 与 FEE 为常量，保留原列；当前 baseline 不使用它们。',
              '6. 无房屋单元编号，不单独去重；满足 IQR 删除条件的重复行同样删除。',
              '7. 按用户要求，租金或面积任一超出 1.5 IQR 范围即删除整行；边界值保留。阈值仅计算一次。',
              '8. 没有填补或裁剪目标值。baseline 同时排除关键字段缺失和 IQR 异常训练行。',
              '9. 导出和最终训练使用全训练集阈值；时间验证仅用较早训练部分估计阈值并删训练行，验证集和测试集不按 IQR 删行。',
              '10. 本次未对地址真实性和房屋属性逐条外部核验；通过规则不代表所有记录都完全正确。', '',
              f"问题计数（可以重叠）：{report['issue_counts']}",
              f"原始重复：{report['duplicates']['raw']}",
              f"清洗后重复：{report['duplicates']['clean']}",
              'excess_rows 表示每组保留第一条后多出的行数；rows_in_groups 包含每组第一条。',
              f"相同全部输入但不同租金：{report['same_features_different_rent']}；可能是同楼栋不同单元，保留。",
              f"关键字段缺失记录：{len(report['baseline_excluded_csv_rows'])}",
              f"两类异常重叠：{report['outlier_removal']['overlap_rows']} 行；合计删除：{len(removed_lines)} 行。",
              '', '删除前数值范围及异常统计：']
    for field, p in report['profile'].items():
        if 'numeric' in p:
            lines.append(f"{field}: 最小 {p['numeric']['0']}，中位数 {p['numeric']['0.5']}，最大 {p['numeric']['1']}")
    for field, result in report['outliers'].items():
        lines.append(f"{field}: IQR 范围 [{result['lower']}, {result['upper']}]；{result['count']} 行满足删除条件。")
    lines += ['', '运行：python3 data_quality.py 生成本报告和清洗副本；python3 baseline.py 使用相同清洗规则训练。',
              '对照：python3 baseline.py --raw 运行原 baseline；默认运行会删除训练部分的 IQR 异常。',
              '', '具体错误记录：']
    for item in issues:
        if item['issue'] in {'invalid_value', 'temporal_inconsistency', 'unrecognized_category'}:
            lines.append(str(item))
    (output / 'report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nOutputs: {output}')


if __name__ == '__main__':
    main()
