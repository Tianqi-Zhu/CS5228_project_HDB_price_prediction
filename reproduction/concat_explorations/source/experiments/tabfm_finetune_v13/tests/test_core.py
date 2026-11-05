import torch
from torch import nn
from experiments.tabfm_finetune_v13.core import prepare_head, masked_targets, query_loss


def test_head_is_fp32_and_backbone_frozen_and_updated():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(3, 4).to(torch.bfloat16)
            self.icl_predictor = nn.Module()
            self.icl_predictor.decoder = nn.Linear(4, 1).to(torch.bfloat16)
        def forward(self, x):
            return self.icl_predictor.decoder(self.backbone(x))
    model = Model()
    prepare_head(model)
    backbone = model.backbone.weight.detach().clone()
    head = model.icl_predictor.decoder.module.weight.detach().clone()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    out = model(torch.ones(1, 3, dtype=torch.bfloat16))
    out.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    opt.step()
    assert torch.equal(backbone, model.backbone.weight)
    assert not torch.equal(head, model.icl_predictor.decoder.module.weight)
    assert model.icl_predictor.decoder.module.weight.dtype == torch.float32


def test_masked_query_labels_and_query_only_loss():
    labels = torch.tensor([[1., 2., 999., -123.]])
    masked = masked_targets(labels, 2)
    assert masked.tolist() == [[1., 2., 0., 0.]]
    pred = torch.tensor([[[100.], [100.], [3.], [4.]]])
    assert query_loss(pred, torch.tensor([[3., 5.]]), 2).item() == 0.5


def test_lora_zero_initialization_and_checkpoint_replay():
    from experiments.tabfm_finetune_v13.core import LoRALinear
    torch.manual_seed(2)
    base = nn.Linear(4, 4).to(torch.bfloat16)
    base.requires_grad_(False)
    layer = LoRALinear(base, rank=2)
    x = torch.randn(3, 4).bfloat16()
    assert torch.equal(layer(x), base(x))
    before = base.weight.detach().clone()
    optimizer = torch.optim.AdamW([layer.a, layer.b], lr=0.01)
    layer(x).float().square().mean().backward()
    assert layer.b.grad.abs().sum() > 0
    optimizer.step()
    assert torch.equal(before, base.weight)
    assert layer.b.detach().abs().sum() > 0
    clone = LoRALinear(nn.Linear(4, 4).bfloat16(), rank=2)
    clone.load_state_dict(layer.state_dict())
    assert torch.equal(layer(x), clone(x))


def test_frozen_prefix_cache_matches_full_model_and_updates_lora():
    from types import SimpleNamespace
    from experiments.tabfm_finetune_v13.core import prepare_lora_head, frozen_prefix, cached_forward
    class AttnBlock(nn.Module):
        def __init__(self):
            super().__init__(); self.attn=nn.Module(); self.attn.q_proj=nn.Linear(4,4); self.attn.v_proj=nn.Linear(4,4)
        def forward(self, x, attn_mask=None, rope=None):
            return self.attn.q_proj(x)+self.attn.v_proj(x)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.icl_predictor=nn.Module(); self.icl_predictor.tf_icl=nn.Module(); self.icl_predictor.tf_icl.blocks=nn.ModuleList([AttnBlock()]); self.icl_predictor.ln=nn.Identity(); self.icl_predictor.decoder=nn.Linear(4,1)
        def forward(self,x,y,train_size,cat_mask=None,d=None):
            b=self.icl_predictor.tf_icl.blocks[-1](x,attn_mask=None,rope=None)
            return self.icl_predictor.decoder(self.icl_predictor.ln(b))
    torch.manual_seed(3); model=prepare_lora_head(Model()); x=torch.randn(1,3,4); y=torch.zeros(1,3); n=torch.tensor([2])
    raw=model(x,y,n); cap=frozen_prefix(model,x,y,n); cached=cached_forward(model,cap)
    assert torch.equal(raw,cached)
    cached[:,2:].square().mean().backward()
    assert model.icl_predictor.tf_icl.blocks[-1].attn.q_proj.b.grad.abs().sum()>0


def test_metadata_canonical_roundtrip():
    from experiments.tabfm_finetune_v13.guarded_resume import canonical_metadata
    data={'stock':{'universe':('O39.SI','U11.SI')}}
    assert canonical_metadata(data)=={'stock':{'universe':['O39.SI','U11.SI']}}


def test_full_cache_cast_only_changes_float_tensors():
    from experiments.tabfm_finetune_v13.full_runner import cast_cap
    src={'args':(torch.ones(2,dtype=torch.bfloat16),torch.tensor([2])), 'kwargs':{'attn_mask':torch.tensor([True])}}
    out=cast_cap(src)
    assert out['args'][0].dtype==torch.float32 and out['args'][1].dtype==torch.int64
    assert out['kwargs']['attn_mask'].dtype==torch.bool and src['args'][0].dtype==torch.bfloat16
