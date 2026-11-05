"""Update-based warmup/cosine scheduling shared by local v15 experiments."""
import math


class WarmupCosine:
    """Set the next update's LR; call step AFTER each successful optimizer step.

    Update zero starts at start_ratio of peak. The final warmup update reaches
    peak, and the final planned update reaches min_lr_ratio. Validation does
    not advance this schedule. Its state belongs in the optimizer checkpoint.
    """

    def __init__(self, optimizer, total_steps, warmup_steps,
                 min_lr_ratio=.01, start_ratio=.01):
        if not isinstance(total_steps, int) or not isinstance(warmup_steps, int):
            raise ValueError('step counts must be integers')
        if not 1 <= warmup_steps < total_steps:
            raise ValueError('require 1 <= warmup_steps < total_steps')
        if not 0 < min_lr_ratio <= 1 or not 0 < start_ratio <= 1:
            raise ValueError('learning rate ratios must lie in (0, 1]')
        self.optimizer = optimizer
        self.config = dict(total_steps=total_steps, warmup_steps=warmup_steps,
                           min_lr_ratio=float(min_lr_ratio), start_ratio=float(start_ratio))
        self.base_lrs = [float(group['lr']) for group in optimizer.param_groups]
        if not self.base_lrs or any(not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs):
            raise ValueError('optimizer peak rates must be positive and finite')
        self.updates_completed = 0
        self._apply()

    def _apply(self):
        conf = self.config
        index = min(self.updates_completed, conf['total_steps'] - 1)
        warmup = conf['warmup_steps']
        if index < warmup:
            ratio = 1. if warmup == 1 else conf['start_ratio'] + (1-conf['start_ratio'])*index/(warmup-1)
        else:
            progress = (index-(warmup-1))/(conf['total_steps']-warmup)
            ratio = conf['min_lr_ratio'] + (1-conf['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
        for group, peak in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = peak * ratio

    def step(self):
        if self.updates_completed >= self.config['total_steps']:
            raise ValueError('schedule has no remaining optimizer updates')
        self.updates_completed += 1
        self._apply()

    def state_dict(self):
        return dict(config=self.config.copy(), base_lrs=self.base_lrs.copy(),
                    updates_completed=self.updates_completed)

    def load_state_dict(self, state):
        if state['config'] != self.config or state['base_lrs'] != self.base_lrs:
            raise ValueError('resume schedule configuration or peak rates differ')
        completed = state['updates_completed']
        if not isinstance(completed, int) or not 0 <= completed <= self.config['total_steps']:
            raise ValueError('invalid completed optimizer update count')
        self.updates_completed = completed
        self._apply()
