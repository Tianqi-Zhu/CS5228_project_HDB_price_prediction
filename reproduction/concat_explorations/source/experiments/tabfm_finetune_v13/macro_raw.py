"""Original DBS LSTM and COE GRU on raw-phase origin-as-of windows."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from experiments.fusion_contract_v3 import contract, dbs, coe, windows
from experiments.macro_encoders_v3 import scaling, encoders
from experiments.tabfm_finetune_v13.data import sha256


class RawMacroSource(nn.Module):
    def __init__(self, origins, fit_origins, arrays, dbs_scaler, coe_scaler, mode, d_model):
        super().__init__()
        if mode not in ('both','dbs','coe'): raise ValueError('mode must be both, dbs, or coe')
        self.mode = mode
        self.origins = np.asarray(origins,dtype=int).tolist()
        self.fit_origins = np.asarray(fit_origins,dtype=int).tolist()
        self.n_tokens = 2 if mode == 'both' else 1
        self.d_model = int(d_model)
        self.encoders = encoders.build_encoders(token_dim=self.d_model)
        self.dbs_scaler = dbs_scaler.to_dict()
        self.coe_scaler = coe_scaler.to_dict()
        dmask = arrays['dbs_mask'] > 0
        cmask = arrays['coe_mask'] > 0
        seq = arrays['coe_seq'].transpose(0,2,1,3).reshape(len(origins),12,24)
        channel_mask = np.repeat(cmask.transpose(0,2,1),6,axis=2)
        dbs_scaled = dbs_scaler.transform(arrays['dbs_seq'],dmask)
        coe_scaled = coe_scaler.transform(seq,cmask.max(axis=1))
        coe_scaled = np.where(channel_mask,coe_scaled,0).astype(np.float32)
        for name,value in [('dbs_seq',dbs_scaled),('dbs_mask',dmask.astype(np.float32)),('coe_seq',coe_scaled),('coe_mask',cmask.max(axis=1).astype(np.float32))]:
            self.register_buffer(name,torch.from_numpy(np.ascontiguousarray(value)),persistent=True)
        self.register_buffer('origin_values',torch.as_tensor(origins,dtype=torch.long),persistent=True)

    def forward(self,batch):
        origin = batch['origin'].to(device=self.origin_values.device,dtype=torch.long)
        idx = torch.searchsorted(self.origin_values,origin)
        if (idx >= len(self.origins)).any() or not torch.equal(self.origin_values[idx.clamp(max=len(self.origins)-1)],origin):
            raise ValueError('batch contains unknown macro origin')
        unique, inverse = torch.unique(idx,sorted=True,return_inverse=True)
        result = []
        if self.mode in ('both','dbs'):
            token,_ = self.encoders.dbs(self.dbs_seq[unique], self.dbs_mask[unique])
            result.append(token[inverse])
        if self.mode in ('both','coe'):
            token,_ = self.encoders.coe(self.coe_seq[unique], self.coe_mask[unique])
            result.append(token[inverse])
        return torch.stack(result,dim=1)

    def describe(self):
        return {'mode':self.mode,'tokens':self.n_tokens,'d_model':self.d_model,'policy':'origin_asof',
                'origins':self.origins,'scaler_fit_origins':self.fit_origins,
                'dbs_scaler':self.dbs_scaler,'coe_scaler':self.coe_scaler,
                'stock_source_sha256':sha256(contract.STOCK_SOURCE),'coe_source_sha256':sha256(contract.COE_SOURCE),
                'semantics':'original DBS LSTM and COE GRU summary tokens, macro status proxies'}


def make_source(fit, metadata, phase='selection', mode='both', d_model=48):
    if phase != 'selection' or metadata['phase'] != phase: raise ValueError('only selection phase is currently authorized')
    fit_origins = np.unique(fit.origin)
    if len(fit.row_id) != 114842 or metadata['n_fit'] != len(fit.row_id): raise ValueError('expected raw selection fit population')
    origins = np.union1d(fit_origins, np.array([metadata['fit_cutoff']]))
    samples = pd.DataFrame({'origin_month_index':origins,'target_month_index':origins+1})
    daily = dbs.build_dbs_frame(dbs.parse_stock_frame())
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    arrays,_ = windows.build_macro_windows(samples,'origin_asof',daily,monthly)
    fitidx = np.flatnonzero(np.isin(origins,fit_origins))
    coeseq = arrays['coe_seq'].transpose(0,2,1,3).reshape(len(origins),12,24)
    kwargs = dict(split_id='raw',stage=phase,policy='origin_asof',fitted_on='unique origins referenced by raw fit rows')
    ds = scaling.fit_dbs_channels(arrays['dbs_seq'][fitidx],arrays['dbs_mask'][fitidx]>0,**kwargs)
    cs = scaling.fit_coe_channels(coeseq[fitidx],arrays['coe_mask'][fitidx]>0,**kwargs)
    return RawMacroSource(origins,fit_origins,arrays,ds,cs,mode,d_model)
