"""Local real-weight cache parity and LoRA + head gradient probe."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
for sub in ('dependencies', 'source'):
    sys.path.insert(0, str(ROOT / 'reports/tabfm_v10/runtime' / sub))
sys.path.insert(0, str(ROOT))
import torch
from tabfm import tabfm_v1_0_0_pytorch
from experiments.tabfm_finetune_v13.core import prepare_lora_head, frozen_prefix, cached_forward, masked_targets, query_loss

torch.manual_seed(42)
torch.set_num_threads(2)
model = tabfm_v1_0_0_pytorch.load(model_type='regression', checkpoint_path=str(ROOT/'reports/tabfm_v10/weights/regression'), device='cpu', dtype=torch.bfloat16, use_cache=False)
prepare_lora_head(model, rank=4)
x = torch.randn(1, 18, 60); x[:, :, -4:] = torch.randint(0, 3, (1, 18, 4)).float()
labels = torch.randn(1, 18); y = masked_targets(labels, 16)
cat = torch.zeros(1, 60, dtype=torch.bool); cat[:, -4:] = True
size = torch.tensor([16]); d = torch.tensor([60])
t = time.perf_counter()
with torch.no_grad(): raw = model(x, y, size, cat_mask=cat, d=d)
raw_s = time.perf_counter()-t
t = time.perf_counter(); cache = frozen_prefix(model, x, y, size, cat, d); cache_s = time.perf_counter()-t
t = time.perf_counter(); out = cached_forward(model, cache); cached_s = time.perf_counter()-t
assert torch.equal(raw, out), (raw-out).abs().max()
trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
frozen = next(p for p in model.parameters() if not p.requires_grad); oldfrozen = frozen.detach().clone()
old = {n:p.detach().clone() for n,p in trainable.items()}
opt = torch.optim.AdamW(trainable.values(), lr=1e-3)
t = time.perf_counter(); query_loss(out, labels[:, 16:], 16).backward(); backward_s = time.perf_counter()-t
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
assert all(trainable[n].grad.abs().sum() > 0 for n in trainable if n.endswith('.b'))
opt.step()
changed = [n for n,p in trainable.items() if not torch.equal(old[n],p)]
assert all(n in changed for n in trainable if n.endswith('.b'))
assert torch.equal(frozen, oldfrozen)
with torch.no_grad():
    trained = cached_forward(model,cache).clone()
    trained_raw = model(x,y,size,cat_mask=cat,d=d)
assert torch.equal(trained,trained_raw), (trained-trained_raw).abs().max()
state = {n:p.detach().clone() for n,p in trainable.items()}
with torch.no_grad():
    for p in trainable.values(): p.zero_()
    for n,p in trainable.items(): p.copy_(state[n])
    replay = cached_forward(model,cache)
assert torch.equal(trained,replay)
result = {'scope':'rank4 LoRA q/v last ICL block + FP32 decoder','total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in trainable.values()),'lora_parameters':sum(p.numel() for n,p in trainable.items() if '.q_proj.' in n or '.v_proj.' in n),'context':16,'query':2,'features':60,'raw_seconds':raw_s,'prefix_seconds':cache_s,'cached_forward_seconds':cached_s,'backward_seconds':backward_s,'cached_exact_parity':True,'trained_adapter_raw_parity':True,'replay_exact':True,'frozen_sample_unchanged':True,'changed_parameters':changed}
path=ROOT/'reports/tabfm_finetune_v13/probe_lora.json'; path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
