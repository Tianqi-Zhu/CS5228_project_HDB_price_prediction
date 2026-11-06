"""Explicit sensitivity policy: collapse rows identical on every observed field."""
from __future__ import annotations
import numpy as np
import pandas as pd
from analysis.data import prepare_main,normalized_address


def canonical_training_rows(raw):
    if 'MONTHLY_RENT' not in raw or raw.MONTHLY_RENT.isna().any():
        raise ValueError('Duplicate identity requires observed training rent')
    if not raw.index.is_unique:
        raise ValueError('Stable, unique source row IDs are required')
    prepared=normalized_address(prepare_main(raw))
    canonical=raw.copy()
    for column in ['TOWN','BLOCK','STREET']:
        canonical[column]=prepared[column]
    canonical['FLAT_TYPE']=prepared.FLAT_TYPE_CLEAN
    canonical['FLAT_MODEL']=prepared.FLAT_MODEL_CLEAN
    # Include rent, month and ALL other source attributes; different contracts
    # can still share them, so this is an experimental policy, not proven truth.
    identity=list(canonical)
    duplicate=canonical.duplicated(identity,keep='first')
    grouped=canonical.assign(__source_position=np.arange(len(canonical)))
    positions=grouped.groupby(identity,sort=False,dropna=False,observed=True)['__source_position'].transform('min').to_numpy(int)
    mapping=pd.DataFrame(dict(removed_row_id=canonical.index[duplicate],
        retained_row_id=canonical.index[positions[duplicate]],
        month=canonical.loc[duplicate,'RENT_APPROVAL_DATE'].to_numpy(),
        rent_sgd=canonical.loc[duplicate,'MONTHLY_RENT'].to_numpy()))
    cleaned=canonical.loc[~duplicate].copy()
    if cleaned.duplicated(identity).any(): raise AssertionError('Duplicate removal failed')
    for removed,retained in mapping[['removed_row_id','retained_row_id']].itertuples(index=False,name=None):
        pd.testing.assert_series_equal(canonical.loc[removed],canonical.loc[retained],check_names=False)
    return cleaned,mapping
