"""Q/K LoRA adapters for the installed TabPFN-3.5-Fast architecture."""

from __future__ import annotations

from unittest.mock import patch

import torch
from torch import nn

from tabpfn import TabPFNRegressor
from tabpfn.finetuning import FinetunedTabPFNRegressor


class LoRALinear(nn.Module):
    """Keep a pretrained projection frozen and add a low-rank weight update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nn.Linear(base.in_features, rank, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** .5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Autocast handles adapter matmuls during training. Cast the update
        # back for ordinary inference when the checkpoint uses another dtype.
        base = self.base(x)
        update = self.lora_b(self.lora_a(x.to(self.lora_a.weight.dtype)))
        return base + update.to(base.dtype) * (self.alpha / self.rank)


def inject_lora_qk(model: nn.Module, *, rank: int = 8, alpha: float = 8.) -> dict:
    """Adapt every paired Q/K attention projection; never adapt V or output."""
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already injected")
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(getattr(module, "q_projection", None), nn.Linear)
               and isinstance(getattr(module, "k_projection", None), nn.Linear)]
    if not targets:
        raise ValueError("No paired attention Q/K linear projections found")
    total = sum(parameter.numel() for parameter in model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, attention in targets:
        attention.q_projection = LoRALinear(attention.q_projection, rank=rank, alpha=alpha)
        attention.k_projection = LoRALinear(attention.k_projection, rank=rank, alpha=alpha)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if len(trainable) != 4 * len(targets) or any(
            not (name.endswith(".lora_a.weight") or name.endswith(".lora_b.weight"))
            for name, _ in trainable):
        raise AssertionError("Only Q/K LoRA matrices may train")
    return {"attention_modules": len(targets), "attention_names": [name for name, _ in targets],
            "rank": rank, "alpha": alpha, "backbone_parameters": total,
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable)}


class LoRAQKTabPFNRegressor(TabPFNRegressor):
    """Load the standard checkpoint, then insert identity-initialized Q/K LoRA."""

    lora_rank = 8
    lora_alpha = 8.

    def _initialize_model_variables(self) -> int:
        byte_size = super()._initialize_model_variables()
        for model in self.models_:
            if not any(isinstance(module, LoRALinear) for module in model.modules()):
                self.lora_audit_ = inject_lora_qk(
                    model, rank=self.lora_rank, alpha=self.lora_alpha)
        return byte_size


class FinetunedLoRAQKRegressor(FinetunedTabPFNRegressor):
    """Use TabPFN's task sampler and optimizer with a frozen backbone."""

    def _create_estimator(self, config):
        return LoRAQKTabPFNRegressor.create_default_for_version(
            version=self.finetune_model_version, **config,
            fit_mode="batched", differentiable_input=False)

    def _fit(self, *args, **kwargs):
        # The installed fine-tuner selects fp16; bf16 avoids nonfinite
        # gradients on this regression checkpoint.
        with patch("tabpfn.finetuning.finetuned_base.autocast",
                   lambda enabled: torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)), \
             patch("tabpfn.finetuning.finetuned_base.GradScaler",
                   lambda: torch.amp.GradScaler("cuda", enabled=False)):
            return super()._fit(*args, **kwargs)

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):
        super()._log_epoch_evaluation(epoch, eval_result, mean_train_loss)
        matrices = [parameter.detach().float() for name, parameter in
                    self.finetuned_estimator_.model_.named_parameters()
                    if name.endswith(".lora_b.weight")]
        norm = torch.sqrt(sum(torch.sum(matrix.square()) for matrix in matrices))
        if not hasattr(self, "lora_epochs_"):
            self.lora_epochs_ = []
        self.lora_epochs_.append({"epoch": int(epoch), "lora_b_l2": float(norm),
                                  "validation_mse": float(eval_result.primary)})
        print(f"Q/K LoRA epoch {epoch}: B_l2={norm.item():.6g}, "
              f"validation_mse={eval_result.primary:.6g}", flush=True)
