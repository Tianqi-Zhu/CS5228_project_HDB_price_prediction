"""Matched-input temporal feature Transformers, sequence oldest to newest."""
import torch
from torch import nn
from experiments.feature_transformer_v4.models import FeatureTransformer

class TemporalFT(nn.Module):
    def __init__(self,numeric_names,cardinalities,sequence_channels,mode='concat',d_model=48,layers=3,heads=4,ff_multiplier=2,dropout=.05,sequence_length=12):
        super().__init__()
        if mode not in ('concat','cross'): raise ValueError('mode must be concat or cross')
        self.mode=mode; self.channels=sequence_channels; self.length=sequence_length
        self.base=FeatureTransformer(numeric_names,tuple(cardinalities),d_model=d_model,layers=layers,heads=heads,ff_multiplier=ff_multiplier,dropout=dropout,pooling='cls')
        self.sequence_projection=nn.Linear(sequence_channels*2,d_model)
        self.position=nn.Parameter(torch.randn(1,sequence_length,d_model)*.02)
        self.empty_history=nn.Parameter(torch.zeros(1,1,d_model))
        if mode=='cross':
            self.cross=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True)
            self.cross_norm=nn.LayerNorm(d_model)
    def forward(self,batch):
        seq=batch['rent_sequence']; mask=batch['sequence_mask'].bool()
        if seq.ndim!=3 or seq.shape[1:]!=(self.length,self.channels) or mask.shape!=seq.shape: raise ValueError('invalid temporal input shape')
        valid=mask.any(dim=-1)
        safe=torch.where(mask,seq,torch.zeros_like(seq))
        history=self.sequence_projection(torch.cat((safe,mask.to(seq.dtype)),dim=-1))+self.position
        # Sentinel is visible only for rows with no observations, avoiding fully masked attention.
        empty=~valid.any(dim=1)
        history=torch.cat((history,self.empty_history.expand(seq.size(0),-1,-1)),dim=1)
        padding=torch.cat((~valid,~empty[:,None]),dim=1)
        features=self.base.feature_tokens(batch); query=self.base._query_token(batch,features.dtype)
        if self.mode=='concat':
            tokens=torch.cat((query,features,history),dim=1)
            prefix=torch.zeros(tokens.size(0),1+features.size(1),dtype=torch.bool,device=tokens.device)
            encoded=self.base.encoder(tokens,src_key_padding_mask=torch.cat((prefix,padding),dim=1))
            pooled=self.base.final_norm(encoded)[:,0]
        else:
            encoded=self.base.encoder(torch.cat((query,features),dim=1))
            pooled=self.base.final_norm(encoded)[:,0:1]
            attended,_=self.cross(pooled,history,history,key_padding_mask=padding,need_weights=False)
            pooled=self.cross_norm(pooled+attended)[:,0]
        return self.base.head(pooled).squeeze(-1)
    def describe(self):
        return {'mode':self.mode,'sequence_length':self.length,'sequence_channels':self.channels,'parameters':sum(p.numel() for p in self.parameters()),'macro_enabled':False}
