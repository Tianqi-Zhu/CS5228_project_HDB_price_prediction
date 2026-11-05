"""Minimal query-only training primitives for the local TabFM model."""
import torch
from torch import nn


class Float32Head(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module.float()

    def forward(self, x):
        return self.module(x.float())


def prepare_head(model):
    """Freeze all original parameters, then train only the FP32 decoder."""
    model.requires_grad_(False)
    model.icl_predictor.decoder = Float32Head(model.icl_predictor.decoder)
    model.icl_predictor.decoder.requires_grad_(True)
    return model


def masked_targets(labels, train_size):
    labels = labels.clone()
    labels[:, train_size:] = 0
    return labels


def query_loss(prediction, query_targets, train_size):
    return (prediction[:, train_size:, 0].float() - query_targets.float()).square().mean()


class LoRALinear(nn.Module):
    """Frozen projection plus a rank-r FP32 update cast to the base dtype."""
    def __init__(self, base, rank=4, alpha=None):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        update = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        return self.base(x) + (self.scale * update).to(x.dtype)


def prepare_lora_head(model, rank=4):
    prepare_head(model)
    attention = model.icl_predictor.tf_icl.blocks[-1].attn
    attention.q_proj = LoRALinear(attention.q_proj, rank)
    attention.v_proj = LoRALinear(attention.v_proj, rank)
    return model


class _Captured(Exception):
    pass


def frozen_prefix(model, x, y, train_size, cat_mask=None, d=None):
    """Capture last ICL block input; valid only while its prefix stays frozen."""
    last = model.icl_predictor.tf_icl.blocks[-1]
    capture = {}
    def hook(module, args, kwargs):
        capture['args'] = args
        capture['kwargs'] = kwargs
        raise _Captured()
    handle = last.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(x, y, train_size, cat_mask=cat_mask, d=d)
    except _Captured:
        pass
    finally:
        handle.remove()
    if not capture:
        raise RuntimeError('last ICL block was not reached')
    return capture


def cached_forward(model, capture):
    out = model.icl_predictor.tf_icl.blocks[-1](*capture['args'], **capture['kwargs'])
    return model.icl_predictor.decoder(model.icl_predictor.ln(out))
