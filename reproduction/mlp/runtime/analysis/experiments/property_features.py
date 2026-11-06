"""Deterministic property features; no rent labels or fitted statistics."""
import numpy as np,pandas as pd


def augment(frame):
    f=frame.copy();dates=pd.to_datetime(f.RENT_APPROVAL_DATE,format='%Y-%m')
    lease=pd.to_numeric(f.LEASE_COMMENCE_DATE,errors='coerce')
    lease=lease.where((lease>0)&(lease<=dates.dt.year))
    completed=pd.to_numeric(f.YEAR_COMPLETED,errors='coerce').where(lambda x:x>0)
    f['XNUM_LEASE_YEAR']=lease
    f['XNUM_LEASE_MISSING']=lease.isna().astype(float)
    f['XNUM_LEASE_COMPLETION_GAP']=lease-completed
    f['XNUM_ABS_LEASE_COMPLETION_GAP']=(lease-completed).abs()
    f['XNUM_MONTH_SIN']=np.sin(2*np.pi*dates.dt.month/12)
    f['XNUM_MONTH_COS']=np.cos(2*np.pi*dates.dt.month/12)
    f['XNUM_AREA_AGE']=f.FLOOR_AREA_SQM*f.FLAT_AGE_YEARS/100
    f['XNUM_AREA_MRT']=f.FLOOR_AREA_SQM*f.NEAREST_MRT_KM
    f['XCAT_TOWN_TYPE']=f.TOWN_CLEAN.astype(str)+'|'+f.FLAT_TYPE_CLEAN.astype(str)
    f['XCAT_STREET_TYPE']=f.STREET_ID.astype(str)+'|'+f.FLAT_TYPE_CLEAN.astype(str)
    f['XCAT_BUILDING_TYPE']=f.BUILDING_ID.astype(str)+'|'+f.FLAT_TYPE_CLEAN.astype(str)
    return f
