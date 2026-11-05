"""One synthetic 60-feature local checkpoint gradient feasibility episode."""
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
for sub in ('dependencies', 'source'):
    sys.path.insert(0, str(ROOT / 'reports/tabfm_v10/runtime' / sub))
sys.path.insert(0, str(ROOT))
import torch
from tabfm import tabfm_v1_0_0_pytorch
from experiments.tabfm_finetune_v13.core import prepare_head, masked_targets, query_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    start = time.perf_counter()
    weights = ROOT / 'reports/tabfm_v10/weights/regression'
    assert (weights / 'model.safetensors').is_file()
    model = tabfm_v1_0_0_pytorch.load(model_type='regression', checkpoint_path=str(weights), device='cpu', dtype=torch.bfloat16, use_cache=False)
    load_s = time.perf_counter() - start
    prepare_head(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    frozen = next(p for p in model.parameters() if not p.requires_grad)
    frozen_before = frozen.detach().clone()
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    x = torch.randn(1, 18, 60)
    labels = torch.randn(1, 18)
    y = masked_targets(labels, 16)
    cats = torch.zeros(1, 60, dtype=torch.bool)
    cats[:, -4:] = True
    x[:, :, -4:] = torch.randint(0, 3, (1, 18, 4)).float()
    t = time.perf_counter()
    output = model(x, y, torch.tensor([16]), cat_mask=cats, d=torch.tensor([60]))
    forward_s = time.perf_counter() - t
    loss = query_loss(output, labels[:, 16:], 16)
    t = time.perf_counter()
    loss.backward()
    backward_s = time.perf_counter() - t
    grads = [p.grad for p in trainable]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    t = time.perf_counter()
    optimizer.step()
    optimizer_s = time.perf_counter() - t
    changed = sum(not torch.equal(a, b) for a, b in zip(before, trainable))
    assert changed and torch.equal(frozen_before, frozen)
    result = {'scope': 'FP32 decoder head only; not LoRA', 'synthetic_episode': {'context':16, 'query':2, 'features':60}, 'total_parameters':sum(p.numel() for p in model.parameters()), 'trainable_parameters':sum(p.numel() for p in trainable), 'load_seconds':load_s, 'forward_seconds':forward_s, 'backward_seconds':backward_s, 'optimizer_seconds':optimizer_s, 'loss':loss.item(), 'changed_parameter_tensors':changed, 'frozen_sample_unchanged':True, 'peak_rss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'torch_threads':torch.get_num_threads()}
    path = ROOT / 'reports/tabfm_finetune_v13/probe_head.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
