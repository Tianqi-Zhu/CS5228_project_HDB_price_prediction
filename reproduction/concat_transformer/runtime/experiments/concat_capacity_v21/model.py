"""Only model width/depth vary; inputs, head count and numerical embedding fixed."""
import torch
from torch import nn
from experiments.feature_transformer_v4.models import FeatureTransformer
from experiments.feature_transformer_v4.macro_v4 import MacroOnFeatureTransformer

class Tokens(nn.Module):
 n_tokens=2
 def __init__(self,left,right,width):
  super().__init__();self.d_model=width
  self.register_buffer('left',torch.tensor(left,dtype=torch.long));self.register_buffer('right',torch.tensor(right,dtype=torch.long))
  self.relative=nn.Sequential(nn.Linear(len(left),width),nn.GELU(),nn.Linear(width,width))
  self.anniversary=nn.Sequential(nn.Linear(len(right),width),nn.GELU(),nn.Linear(width,width))
 def forward(self,batch):
  x=batch['numeric'];return torch.stack([self.relative(x[:,self.left]),self.anniversary(x[:,self.right])],dim=1)

def make_model(names,cards,width=128,depth=3):
 left=[i for i,n in enumerate(names) if (n.startswith('HIST_') or 'RATIO' in n or n.startswith('PAST_TYPE')) and not n.startswith('ANN_')]
 right=[i for i,n in enumerate(names) if n.startswith('ANN_') or n.startswith('COHORT24')]
 if not left or not right:raise ValueError('missing history groups')
 base=FeatureTransformer(names,tuple(cards),d_model=width,layers=depth,heads=4,ff_multiplier=2,dropout=.05,pooling='cls')
 return MacroOnFeatureTransformer(base,Tokens(left,right,width),'both')
