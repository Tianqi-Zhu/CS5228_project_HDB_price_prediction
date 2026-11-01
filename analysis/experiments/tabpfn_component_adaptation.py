"""Opt-in component adaptation; existing TabPFN runners remain unchanged."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from tabpfn import TabPFNRegressor
from tabpfn.finetuning import FinetunedTabPFNRegressor
from tabpfn.finetuning.finetuned_base import EvalResult
from tabpfn.finetuning.finetuned_base import _parameters_unused_by_task

from analysis.experiments.diagscale_tabpfn import inject_diagscale
from analysis.experiments.lora_qk_tabpfn import LoRALinear


def configure_component(model, config):
    """Freeze the backbone, then enable exactly the requested component."""
    if getattr(model, "_component_configured", False):
        raise ValueError("Component was already configured")
    kind = config["component"]
    if kind not in ("icl_lora", "head", "diagscale"):
        raise ValueError(f"Unknown component: {kind}")
    for p in model.parameters():
        p.requires_grad_(False)
    targets = []
    if kind == "icl_lora":
        rank = config["rank"]
        blocks = model.icl_blocks
        last = config.get("last_blocks")
        if last is not None and not 1 <= last <= len(blocks):
            raise ValueError("Invalid last-block count")
        start = 0 if last is None else len(blocks) - last
        # Select by architectural stage, not an incidental substring anywhere.
        for i in range(start, len(blocks)):
            attention = blocks[i].icl_attention
            for projection in ("q_projection", "k_projection"):
                base = getattr(attention, projection)
                if not isinstance(base, nn.Linear):
                    raise TypeError("Unexpected ICL projection or repeated injection")
                setattr(attention, projection, LoRALinear(base, rank=rank, alpha=rank))
                targets.append(f"icl_blocks.{i}.icl_attention.{projection}")
    elif kind == "head":
        for name in ("mlp_regression", "output_projection"):
            module = getattr(model.heads, name)
            for p in module.parameters():
                p.requires_grad_(True)
            targets.append(f"heads.{name}")
    else:
        inject_diagscale(model)
        targets = [n for n, p in model.named_parameters() if p.requires_grad]
    for p in _parameters_unused_by_task(model, "regression"):
        p.requires_grad_(False)
    model._component_configured = True
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise AssertionError("No trainable component")
    return {"component": kind, "targets": targets, "trainable_names": names,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def frozen_digest(model):
    """Exact digest of frozen parameters, without retaining a second model."""
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def copy_component(model, state):
    named = dict(model.named_parameters())
    expected = {n for n, p in named.items() if p.requires_grad}
    # Task-specific freezing can remove irrelevant classification deltas.
    allowed = expected | {n for n in named if n.endswith(".delta")}
    if not state or set(state) - allowed or expected - set(state):
        raise ValueError("Saved trainable parameter schema differs from component")
    with torch.no_grad():
        for name, value in state.items():
            if named[name].shape != value.shape or not torch.isfinite(value).all():
                raise ValueError("Invalid component tensor")
            named[name].copy_(value.to(named[name]))


class ComponentRegressor(TabPFNRegressor):
    def __init__(self, *, component_config, adapter_path=None, **kwargs):
        self.component_config = component_config
        self.adapter_path = adapter_path
        super().__init__(**kwargs)

    def _initialize_model_variables(self):
        size = super()._initialize_model_variables()
        for model in self.models_:
            if not getattr(model, "_component_configured", False):
                self.component_audit_ = configure_component(model, self.component_config)
            if self.adapter_path:
                copy_component(model, torch.load(self.adapter_path, map_location="cpu", weights_only=True))
        return size


class FinetunedComponentRegressor(FinetunedTabPFNRegressor):
    """Select checkpoints with weighted one-view deployment RMSE, including epoch 0."""

    def _create_estimator(self, config):
        return ComponentRegressor.create_default_for_version(
            version=self.finetune_model_version, component_config=self.component_config,
            **config, fit_mode="batched", differentiable_input=False)

    def _fit(self, *args, **kwargs):
        with patch("tabpfn.finetuning.finetuned_base.autocast",
                   lambda enabled: torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)), \
             patch("tabpfn.finetuning.finetuned_base.GradScaler",
                   lambda: torch.amp.GradScaler("cuda", enabled=False)):
            return super()._fit(*args, **kwargs)

    def _evaluate_model(self, eval_config, X_train, y_train, X_val, y_val):
        # The upstream loop converts DataFrames to NumPy. Use the exact original
        # typed frames for checkpoint inference, matching standalone deployment.
        if len(X_train) != len(self.calibration_X_train) or len(X_val) != len(self.calibration_X_query):
            raise AssertionError("Checkpoint rows changed")
        X_train, X_val = self.calibration_X_train, self.calibration_X_query
        # Rebuild from the original checkpoint and copy only the learned tensors.
        # This also avoids inherited fine-tuner model/runtime flags affecting
        # inference differently from a standalone reloaded checkpoint.
        cfg = {**self.extra_regressor_kwargs, "device": "cuda", "random_state": 42,
               "n_estimators": 1, "auto_scale_n_estimators": False,
               "fit_mode": "fit_with_cache", "keep_cache_on_device": True,
               "inference_config": {"SUBSAMPLE_SAMPLES": None}}
        state = {n: p.detach().float().cpu().clone() for n, p in self.finetuned_estimator_.model_.named_parameters()
                 if p.requires_grad}
        evaluation_path = Path(self.component_checkpoint_path).with_name("evaluation.pt")
        torch.save(state, evaluation_path)
        evaluator = ComponentRegressor.create_default_for_version(
            version=self.finetune_model_version, component_config=self.component_config,
            adapter_path=str(evaluation_path), **cfg)
        try:
            evaluator.fit(X_train, y_train)
            for model in evaluator.models_:
                named = dict(model.named_parameters())
                for name, tensor in state.items():
                    torch.testing.assert_close(named[name].detach().float().cpu(), tensor, rtol=0, atol=0)
            if evaluator.n_train_samples_ != len(X_train):
                raise AssertionError("Checkpoint context truncated")
            prediction = np.concatenate([
                np.asarray(evaluator.predict(X_val.iloc[i:i + 200], output_type="mean"), float)
                for i in range(0, len(X_val), 200)])
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite checkpoint prediction")
            # Adding the same fixed baseline to prediction/target cancels exactly.
            mse = float(np.average((prediction - y_val) ** 2, weights=self.calibration_weights))
            self.latest_calibration_prediction_ = prediction
            return EvalResult(primary=mse, secondary={"weighted_rmse_sgd": mse ** .5})
        finally:
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):
        model = self.finetuned_estimator_.model_
        if epoch == -1:
            self.frozen_before_ = frozen_digest(model)
            self.component_trace_ = []
        record = {"epoch": int(epoch), "weighted_rmse_sgd": float(eval_result.primary ** .5),
                  "mean_train_loss": None if mean_train_loss is None else float(mean_train_loss)}
        self.component_trace_.append(record)
        if len(self.component_trace_) == 1 or eval_result.primary < self.best_component_mse_ - self.min_delta:
            self.best_component_mse_ = eval_result.primary
            self.best_component_epoch_ = epoch
            state = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(state, self.component_checkpoint_path)
            np.save(Path(self.component_checkpoint_path).with_suffix(".prediction.npy"),
                    self.latest_calibration_prediction_)
        import json
        Path(self.component_trace_path).write_text(json.dumps(self.component_trace_, indent=2), encoding="utf-8")
        print(f"COMPONENT epoch={epoch} deployment_weighted_RMSE={record['weighted_rmse_sgd']:.6f}", flush=True)
