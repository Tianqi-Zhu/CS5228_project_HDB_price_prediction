"""DiagScale adapters for the installed TabPFN-3.5-Fast regressor.

The wrapper follows Ning et al.'s released TabPFN-3 adapter: multiply each
pretrained SoftmaxScalingMLP output by a zero-initialized per-head/per-dimension
``1 + delta``. This local version targets TabPFN 9.2.0's 3.5 architecture.
The reference implementation is Apache-2.0:
https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models
"""

from __future__ import annotations

import torch
from torch import nn
from unittest.mock import patch

from tabpfn import TabPFNRegressor
from tabpfn.architectures.tabpfn_v3_5 import SoftmaxScalingMLP
from tabpfn.finetuning import FinetunedTabPFNRegressor


class DiagScale(nn.Module):
    """Apply a learned diagonal correction after pretrained query scaling."""

    def __init__(self, base: SoftmaxScalingMLP):
        super().__init__()
        self.base = base
        self.delta = nn.Parameter(torch.zeros(base.num_heads, base.head_dim, dtype=torch.float32,
                                              device=next(base.parameters()).device))

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        scaled = self.base(q, n)
        return scaled * (1 + self.delta.to(dtype=scaled.dtype))


def inject_diagscale(model: nn.Module) -> dict[str, int]:
    """Freeze all pretrained weights and wrap every 3.5 attention scale module."""
    if any(isinstance(module, DiagScale) for module in model.modules()):
        raise ValueError("DiagScale is already injected")
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, SoftmaxScalingMLP)]
    if not targets:
        raise ValueError("No TabPFN-3.5 SoftmaxScalingMLP modules found")
    total = sum(parameter.numel() for parameter in model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, module in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, DiagScale(module))
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if len(trainable) != len(targets) or any(not name.endswith(".delta") for name, _ in trainable):
        raise AssertionError("Only DiagScale delta tensors may train")
    return {"scaling_modules": len(targets), "backbone_parameters": total,
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable)}


class DiagScaleTabPFNRegressor(TabPFNRegressor):
    """Load the standard checkpoint first, then insert zero-identity adapters."""

    def _initialize_model_variables(self) -> int:
        byte_size = super()._initialize_model_variables()
        for model in self.models_:
            if not any(isinstance(module, DiagScale) for module in model.modules()):
                self.diagscale_audit_ = inject_diagscale(model)
        return byte_size


class FinetunedDiagScaleRegressor(FinetunedTabPFNRegressor):
    """Use the installed fine-tuning loop with only diagonal adapters trainable."""

    def _create_estimator(self, config):
        return DiagScaleTabPFNRegressor.create_default_for_version(
            version=self.finetune_model_version, **config,
            fit_mode="batched", differentiable_input=False)

    def _fit(self, *args, **kwargs):
        # The installed fine-tuner defaults to fp16 autocast on CUDA. On this
        # regression checkpoint fp16 produced finite loss but NaN adapter
        # gradients; bf16 retains the dynamic range needed for backprop.
        with patch("tabpfn.finetuning.finetuned_base.autocast",
                   lambda enabled: torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)), \
             patch("tabpfn.finetuning.finetuned_base.GradScaler",
                   lambda: torch.amp.GradScaler("cuda", enabled=False)):
            return super()._fit(*args, **kwargs)

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):
        super()._log_epoch_evaluation(epoch, eval_result, mean_train_loss)
        deltas = [m.delta.detach().float() for m in self.finetuned_estimator_.model_.modules()
                  if isinstance(m, DiagScale)]
        total = torch.sqrt(sum(torch.sum(delta.square()) for delta in deltas))
        if not hasattr(self, "diagscale_epochs_"):
            self.diagscale_epochs_ = []
        self.diagscale_epochs_.append({"epoch": int(epoch), "delta_l2": float(total),
                                       "validation_mse": float(eval_result.primary)})
        print(f"DiagScale epoch {epoch}: delta_l2={total.item():.6g}, validation_mse={eval_result.primary:.6g}", flush=True)
