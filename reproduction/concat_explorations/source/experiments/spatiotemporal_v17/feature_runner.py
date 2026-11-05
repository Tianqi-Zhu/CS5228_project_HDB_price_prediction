"""Build immutable, audited v17 feature banks. No model training."""
import argparse, hashlib, json, os, tempfile, time
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from experiments.fusion_contract_v2 import contract as c2, export, panel
from experiments.fusion_contract_v3 import contract as c3, dbs, coe
from experiments.neural_fusion_v2.contract import StageArrays
from experiments.neural_fusion_v2.preprocessing import ScalerState
from experiments.tabfm_finetune_v13.data import fit_scalar_state, stock_sector_windows, STOCK_NAMES, STOCK_CSV
from experiments.tabfm_finetune_v13.data_full import make_samples, _slice
from .contract import OUT, ROOT
from .features import temporal_features, fit_categories


def sample_ids(frame, indices, count, seed):
    """Proportional stratification with exact largest-remainder allocation; no labels."""
    if count>=len(indices): return np.asarray(indices)
    subset=frame.iloc[indices]; groups=subset.groupby(['MONTH_INDEX','TOWN_CLEAN','FLAT_TYPE_CLEAN'],dropna=False,sort=True).indices
    keys=list(groups); sizes=np.array([len(groups[k]) for k in keys]); expected=sizes*count/len(indices)
    allocations=np.floor(expected).astype(int); remaining=count-allocations.sum()
    order=np.argsort(-(expected-allocations),kind='stable'); allocations[order[:remaining]]+=1
    rng=np.random.default_rng(seed); return np.sort(np.concatenate([rng.choice(np.asarray(indices)[groups[k]],int(n),replace=False) for k,n in zip(keys,allocations) if n]))


def atomic_json(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)); os.replace(tmp,path)


def build(stage='screen',split='orig'):
    start=time.time(); frame=export.load_source(); cutoff={'orig':38,'early':26}[split]
    table=make_samples(frame.MONTH_INDEX.to_numpy(),'selection'); month=frame.MONTH_INDEX.to_numpy()
    table['population']=np.where(month<=cutoff,'fit',np.where(month<=cutoff+12,'validation','excluded'))
    val=table.population.eq('validation'); table.loc[val,'origin_month_index']=cutoff; table.loc[val,'horizon_months']=month[val]-cutoff
    all_fit=np.flatnonzero(table.population.eq('fit')); all_val=np.flatnonzero(val)
    fi=sample_ids(frame,all_fit,30000 if stage=='screen' else len(all_fit),42)
    vi=sample_ids(frame,all_val,5000 if stage=='screen' else len(all_val),43)
    ids=np.sort(np.concatenate([fi,vi])); table=table.iloc[ids].copy(); table['actual_rent_sgd']=frame.MONTHLY_RENT.to_numpy()[ids]
    fit=np.flatnonzero(table.population.eq('fit')); valid=np.flatnonzero(table.population.eq('validation'))
    assert not np.intersect1d(fi,vi).size and np.all(table.iloc[valid].origin_month_index==cutoff)
    p=panel.Panel.build(frame,cutoff); built=export.build_features(frame,table,p); ar=built.arrays; order=built.feature_order
    ai=order['context'].index('anchor_rent_sgd'); ar['context'][:,ai]=np.nan_to_num(ar['context'][:,ai],nan=0.)
    arrays=StageArrays(split=split,stage=stage,**ar,static_order=order['static'],context_order=order['context'],history_order=order['history'],sequence_order=order['sequence'])
    scaler=ScalerState.fit(_slice(arrays,fit),population=f'{split}/{stage}/fit'); tr=scaler.transform(arrays)
    core=np.concatenate([tr['static'],tr['history'],tr['context']],axis=1); core_names=list(scaler.static_columns)+list(scaler.history_columns)+list(scaler.context_columns)
    assert len(core_names)==56 and len(set(core_names))==56 and len(arrays.categorical_columns)==4
    # Exact v15 flat macro definitions, fitted only on selected fit origins.
    origins=np.unique(arrays.origin); vv=[]; mm=[]; daily=dbs.build_dbs_frame(dbs.parse_stock_frame()); monthly=coe.build_coe_monthly(coe.parse_coe_frame())
    for o in origins:
        dv,dm=dbs.scalar_arrays(daily,dbs.asof_position(daily,int(o)),int(o)); cv,cm=coe.scalar_arrays(monthly,int(o))
        vv.append(np.concatenate((dv,cv.ravel()))); mm.append(np.concatenate((dm,cm.ravel())))
    vv=np.asarray(vv); mm=np.asarray(mm,dtype=bool); fwin=np.flatnonzero(np.isin(origins,np.unique(arrays.origin[fit])))
    macro_names=list(c3.DBS_SCALARS)+[f'{cat}_{name}' for cat in c3.COE_CATEGORIES for name in c3.COE_SCALARS]
    ms=fit_scalar_state(vv,mm,fwin,macro_names); macro=np.concatenate((ms.transform(vv,mm),mm.astype(np.float32)),axis=1)[np.searchsorted(origins,arrays.origin)]
    sv,sm,stock_audit=stock_sector_windows(pd.read_csv(STOCK_CSV),origins); ss=fit_scalar_state(sv,sm,fwin,STOCK_NAMES)
    stock=np.concatenate((ss.transform(sv,sm),sm.astype(np.float32)),axis=1)[np.searchsorted(origins,arrays.origin)]
    full=np.concatenate([core,macro,stock],axis=1); full_names=core_names+macro_names+[n+'_observed' for n in macro_names]+STOCK_NAMES+[n+'_observed' for n in STOCK_NAMES]
    assert len(full_names)==184 and len(set(full_names))==184
    rows=frame.iloc[ids].reset_index(drop=True); spatial_cols=['STREET_CLEAN','SUBZONE','PLANNING_AREA','BLOCK_ID']; spatial,vocab=fit_categories(rows,fit,spatial_cols)
    temporal,seq,mask,temporal_names=temporal_features(rows,arrays.origin,cutoff,frame)
    med=np.nanmedian(temporal[fit],axis=0); mean=np.nanmean(temporal[fit],axis=0); std=np.nanstd(temporal[fit],axis=0); std[std<1e-8]=1
    temporal=np.clip((np.where(np.isfinite(temporal),temporal,med)-mean)/std,-8,8).astype(np.float32)
    clean=[]; pruned=[]
    for j,name in enumerate(full_names):
        col=full[fit,j]
        # Retain missing/observed indicators, including constant ones.
        if ('missing' not in name.lower() and 'observed' not in name.lower()) and (np.all(col==col[0]) or any(np.array_equal(col,full[fit,k]) for k in clean)):
            pruned.append(name)
        else: clean.append(j)
    cards=[int(tr['categorical'][fit,i].max())+1 for i in range(4)]; spatial_cards=[len(vocab[n])+1 for n in spatial_cols]
    banks={'F0':(full,full_names,False),'F1':(core,core_names,False),'F2':(full[:,clean],[full_names[i] for i in clean],False),'F3':(core,core_names,True),'F4':(np.concatenate([core,temporal],1),core_names+temporal_names,False),'F5':(np.concatenate([core,temporal],1),core_names+temporal_names,True)}
    common={'y_sgd':arrays.target_sgd,'anchor_ksgd':tr['anchor_ksgd'],'residual':arrays.target_sgd/1000-tr['anchor_ksgd'],'row_id':arrays.row_id,'origin':arrays.origin,'horizon':arrays.horizon,'month':month[ids], 'town':rows.TOWN_CLEAN.fillna('__missing__').astype(str).to_numpy(dtype='U'), 'flat_type':rows.FLAT_TYPE_CLEAN.fillna('__missing__').astype(str).to_numpy(dtype='U'),'rent_sequence':seq,'sequence_mask':mask}
    assert np.allclose((common['residual']+common['anchor_ksgd'])*1000,common['y_sgd'],atol=.001)
    sequence_names=['town_ft_rent_ksgd','town_ft_log_count','block_ft_rent_ksgd','block_ft_log_count']
    base=OUT/split/stage; base.mkdir(parents=True,exist_ok=True)
    state={'scaler':scaler.to_dict(),'encoding':built.encoding,'spatial_vocab':vocab,'temporal_median':med.tolist(),'temporal_mean':mean.tolist(),'temporal_std':std.tolist(),'macro':ms.to_dict(),'stock':ss.to_dict(),'pruned':pruned}
    state_path=base/'fit_state.json'; atomic_json(state_path,state)
    for name,(numeric,names,with_spatial) in banks.items():
        cat=np.concatenate([tr['categorical'],spatial],axis=1) if with_spatial else tr['categorical']
        path=base/name; path.mkdir(exist_ok=True)
        out={}
        for part,idx in [('train',fit),('val',valid)]:
            for k,v in {**common,'x_num':numeric.astype(np.float32),'x_cat':cat}.items(): out[part+'_'+k]=v[idx]
        tmp=path/'arrays.tmp.npz'; np.savez_compressed(tmp,**out); os.replace(tmp,path/'arrays.npz')
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        manifest={'bank_id':name,'split':split,'stage':stage,'numeric_names':names,'categorical_names':list(arrays.categorical_columns)+(spatial_cols if with_spatial else []),'cardinalities':cards+(spatial_cards if with_spatial else []),'sequence_names':sequence_names,'array_sha256':digest(path/'arrays.npz'),'fit_state_sha256':digest(state_path),'source_sha256':export.sha256_file(c2.SOURCE),'counts':{'train':len(fit),'val':len(valid)},'cutoff':cutoff,'history_source':'all legal source labels up to each origin, including unsampled fit rows','pruned':pruned if name=='F2' else [],'sequence_shape':'N x 12 x 4 oldest to newest','elapsed_seconds':time.time()-start,'development_validation':True}
        atomic_json(path/'manifest.json',manifest)
    print(json.dumps({'banks':list(banks),'train':len(fit),'val':len(valid),'seconds':time.time()-start}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['screen','full'],default='screen'); ap.add_argument('--split',choices=['orig','early'],default='orig'); args=ap.parse_args(); build(args.stage,args.split)
