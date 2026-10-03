"""Build reproducible, target-independent features from the provided snapshots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = DATA / 'processed'
REPORT = ROOT / 'reports/spatial_features'
KEYS = ['TOWN', 'BLOCK', 'STREET']
DICTIONARY = []


def clean(series):
    result = series.astype('string').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    return result.mask(result.isin(['nil', '-1', '', 'nan', 'none']))


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def define(name, source, formula, units, explanation):
    DICTIONARY.append(dict(feature=name, source=source, formula=formula, units=units, explanation=explanation))


def locations(frame):
    coords = frame[['LATITUDE', 'LONGITUDE']].apply(pd.to_numeric, errors='coerce')
    coords = coords.where(coords.ge(0)).dropna().drop_duplicates()
    assert coords.LATITUDE.between(1, 2).all() and coords.LONGITUDE.between(103, 105).all()
    return coords.to_numpy()


def distance_features(blocks, points, prefix, counts=True):
    """Haversine in km; each auxiliary coordinate pair counts once."""
    xy = blocks[['LATITUDE', 'LONGITUDE']].to_numpy(dtype=float)
    result = np.full((len(xy), 4), np.nan)
    q = np.radians(points)
    for start in range(0, len(xy), 512):
        p = np.radians(xy[start:start + 512])
        delta = p[:, None, :] - q[None, :, :]
        a = np.sin(delta[:, :, 0] / 2)**2 + np.cos(p[:, None, 0]) * np.cos(q[None, :, 0]) * np.sin(delta[:, :, 1] / 2)**2
        distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        if len(points):
            result[start:start + len(p), 0] = distances.min(axis=1)
            for j, radius in enumerate([.5, 1, 2], 1):
                result[start:start + len(p), j] = (distances <= radius).sum(axis=1)
        result[start:start + len(p)][~np.isfinite(p).all(axis=1)] = np.nan
    name = f'NEAREST_{prefix}_KM'
    blocks[name] = result[:, 0]
    define(name, prefix.lower() + ' snapshot', 'minimum haversine distance; Earth radius 6371.0088 km', 'km', 'Current snapshot, not historical availability at approval date')
    if counts:
        for j, radius in enumerate([500, 1000, 2000], 1):
            name = f'{prefix}_COUNT_{radius}M'
            blocks[name] = result[:, j]
            define(name, prefix.lower() + ' snapshot', f'unique coordinate pairs within inclusive {radius} m haversine radius', 'locations', 'Co-located records counted once; missing block coordinates produce missing count')


def main():
    DICTIONARY.clear()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = [DATA/'train.csv', DATA/'test.csv'] + [DATA/'auxiliary'/f'sg-{name}.csv' for name in ['hdb-block', 'mrt-stations', 'schools', 'shopping-malls']]
    manifest = {'raw_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'splits': {}, 'limitations': ['Amenities represent supplied current snapshots, not approval-date historical availability.', 'Primary schools include MAINLEVEL_CODE containing primary or p1.', 'Location counts deduplicate exact coordinate pairs; distinct sites with slightly differing coordinates remain distinct.', 'No target-derived values, fitted scaling, or imputation.'], 'cbd_reference': {'latitude': 1.2840, 'longitude': 103.8510, 'definition': 'Approximate Raffles Place reference chosen for this analysis, not an authoritative CBD boundary.'}}
    hdb = pd.read_csv(paths[2], dtype={k: 'string' for k in KEYS})
    for k in KEYS:
        hdb[k+'_CLEAN'] = clean(hdb[k])
    join_keys = [k+'_CLEAN' for k in KEYS]
    assert not hdb.duplicated(join_keys).any(), 'Normalized HDB keys must be unique'
    attrs = ['LATITUDE', 'LONGITUDE', 'MAX_FLOOR', 'YEAR_COMPLETED', 'SUBZONE', 'PLANNING_AREA', 'REGION']
    blocks = hdb[join_keys+attrs].copy()
    for k in attrs[:4]:
        blocks[k] = pd.to_numeric(blocks[k], errors='coerce').mask(lambda x: x < 0)
    for k in attrs[4:]:
        blocks[k] = clean(blocks[k])
    for k in attrs:
        define(k, 'sg-hdb-block.csv', 'normalized town/block/street many-to-one join; negative numeric sentinels removed', 'degrees' if k in ['LATITUDE', 'LONGITUDE'] else 'year' if k == 'YEAR_COMPLETED' else 'floors' if k == 'MAX_FLOOR' else 'category', 'Supplied block metadata; categorical whitespace/case/sentinels normalized')
    mrt = pd.read_csv(paths[3])
    schools = pd.read_csv(paths[4])
    malls = pd.read_csv(paths[5])
    sets = {'OPEN_MRT': locations(mrt[clean(mrt.STATUS).eq('open')]), 'PLANNED_MRT': locations(mrt[clean(mrt.STATUS).eq('planned')]), 'MALL': locations(malls), 'PRIMARY_SCHOOL': locations(schools[clean(schools.MAINLEVEL_CODE).str.contains('primary|p1', na=False)])}
    manifest['unique_amenity_locations'] = {k: len(v) for k, v in sets.items()}
    manifest['open_planned_colocated_coordinates'] = len(set(map(tuple, sets['OPEN_MRT'])) & set(map(tuple, sets['PLANNED_MRT'])))
    for name, points in sets.items():
        distance_features(blocks, points, name, name != 'PLANNED_MRT')
    distance_features(blocks, np.array([[1.2840, 103.8510]]), 'CBD', False)
    blocks.rename(columns={'NEAREST_CBD_KM': 'DISTANCE_TO_CBD_KM'}, inplace=True)
    DICTIONARY[-1].update(feature='DISTANCE_TO_CBD_KM', source='chosen approximate Raffles Place reference', explanation='Haversine distance to (1.2840, 103.8510); not distance to CBD boundary')
    define('SOURCE_ROW_ID', 'input row order', 'zero-based original row position', 'row', 'Stable identifier within each split')
    for k in ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FURNISHED']:
        define(k+'_CLEAN', 'original '+k, 'strip, lowercase, collapse whitespace; nil/-1/empty to missing', 'category', 'Original source column retained unchanged')
    next(x for x in DICTIONARY if x['feature'] == 'FLAT_TYPE_CLEAN')['formula'] += '; hyphens replaced with spaces, then whitespace collapsed'
    for name, formula, unit in [('APPROVAL_YEAR', 'approval date year', 'year'), ('APPROVAL_MONTH', 'approval date month', 'month'), ('APPROVAL_MONTH_INDEX', '(year-2021)*12+month-1; 0 = January 2021', 'month'), ('APPROVAL_MONTH_SIN', 'sin(2*pi*(month-1)/12)', 'unitless'), ('APPROVAL_MONTH_COS', 'cos(2*pi*(month-1)/12)', 'unitless'), ('YEARS_SINCE_LEASE_COMMENCEMENT', 'approval year - lease commencement year; negative to missing', 'years'), ('BUILDING_AGE_YEARS', 'approval year - year completed; negative to missing', 'years'), ('LEASE_AGE_NEGATIVE_FLAG', '1 if raw lease age <0 else 0', 'binary'), ('BUILDING_AGE_NEGATIVE_FLAG', '1 if raw building age <0 else 0', 'binary')]:
        define(name, 'approval date and property metadata', formula, unit, 'Year-granularity age; missing dates/metadata propagate to ages')
    for split, path in zip(['train', 'test'], paths[:2]):
        original = pd.read_csv(path)
        frame = original.copy()
        frame['SOURCE_ROW_ID'] = np.arange(len(frame))
        for k in ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE', 'FLAT_MODEL', 'FURNISHED']:
            frame[k+'_CLEAN'] = clean(frame[k])
        frame['FLAT_TYPE_CLEAN'] = frame.FLAT_TYPE_CLEAN.str.replace('-', ' ', regex=False).str.replace(r'\s+', ' ', regex=True)
        frame = frame.merge(blocks, on=join_keys, how='left', validate='many_to_one', sort=False, indicator=True)
        coverage = int(frame['_merge'].eq('both').sum())
        assert coverage == len(original), f'{split}: incomplete block join'
        frame.drop(columns='_merge', inplace=True)
        assert np.array_equal(frame.SOURCE_ROW_ID, np.arange(len(original)))
        assert fingerprint(frame[original.columns]) == fingerprint(original)
        date = pd.to_datetime(frame.RENT_APPROVAL_DATE, format='%Y-%m', errors='raise')
        frame['APPROVAL_YEAR'] = date.dt.year
        frame['APPROVAL_MONTH'] = date.dt.month
        frame['APPROVAL_MONTH_INDEX'] = (date.dt.year - 2021) * 12 + date.dt.month - 1
        frame['APPROVAL_MONTH_SIN'] = np.sin(2*np.pi*(date.dt.month-1)/12)
        frame['APPROVAL_MONTH_COS'] = np.cos(2*np.pi*(date.dt.month-1)/12)
        for age, source, flag in [('YEARS_SINCE_LEASE_COMMENCEMENT', 'LEASE_COMMENCE_DATE', 'LEASE_AGE_NEGATIVE_FLAG'), ('BUILDING_AGE_YEARS', 'YEAR_COMPLETED', 'BUILDING_AGE_NEGATIVE_FLAG')]:
            year = pd.to_numeric(frame[source], errors='coerce').mask(lambda x: x < 0)
            raw_age = frame.APPROVAL_YEAR - year
            frame[flag] = raw_age.lt(0).astype('int8')
            frame[age] = raw_age.mask(raw_age < 0)
        destination = OUT/f'{split}_spatial_features.csv'
        frame.to_csv(destination, index=False)
        reread = pd.read_csv(destination)
        assert fingerprint(reread[original.columns]) == fingerprint(original), 'CSV round-trip changed originals'
        assert reread.SOURCE_ROW_ID.tolist() == list(range(len(original)))
        manifest['splits'][split] = {'rows': len(frame), 'matched_block_rows': coverage, 'join_coverage': coverage/len(frame), 'original_columns_order_hash': fingerprint(original), 'output_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'negative_age_rows': {k: int(frame[k].sum()) for k in ['LEASE_AGE_NEGATIVE_FLAG', 'BUILDING_AGE_NEGATIVE_FLAG']}, 'missing_by_feature': frame.isna().sum().astype(int).to_dict(), 'dtypes': {k: str(v) for k,v in frame.dtypes.items()}}
    pd.DataFrame(DICTIONARY).to_csv(REPORT/'feature_dictionary.csv', index=False)
    manifest['feature_definitions'] = DICTIONARY
    manifest['raw_sha256_after_build'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert manifest['raw_sha256_after_build'] == manifest['raw_sha256']
    manifest['raw_files_unchanged'] = True
    (REPORT/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'splits': {k: {'rows':v['rows'], 'join_coverage':v['join_coverage'], 'negative_age_rows':v['negative_age_rows']} for k,v in manifest['splits'].items()}, 'unique_amenity_locations': manifest['unique_amenity_locations'], 'open_planned_colocated_coordinates': manifest['open_planned_colocated_coordinates']}, indent=2))


if __name__ == '__main__':
    main()
