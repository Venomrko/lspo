from __future__ import annotations
import math
from typing import List, Optional
import torch

class WarmupCosineSchedule:

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int=500, total_steps: int=15000, min_lr: float=1e-07, base_lr: Optional[float]=None):
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self.base_lrs: List[float] = [group['lr'] for group in optimizer.param_groups] if base_lr is None else [base_lr] * len(optimizer.param_groups)
        self.step_count = 0
        self.set_lr(0)

    def set_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
        else:
            progress = (step - self.warmup_steps + 1) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        current = 0.0
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = self.min_lr + (base - self.min_lr) * scale
            current = group['lr']
        return current

    def step(self) -> float:
        self.step_count += 1
        return self.set_lr(self.step_count)

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

class ConstantSchedule:

    def __init__(self, optimizer, warmup_steps: int=0, total_steps: int=1, min_lr: float=0.0, base_lr: Optional[float]=None):
        self.optimizer = optimizer
        self.step_count = 0
        self.base_lr = base_lr if base_lr is not None else optimizer.param_groups[0]['lr']

    def set_lr(self, step: int) -> float:
        del step
        for group in self.optimizer.param_groups:
            group['lr'] = self.base_lr
        return self.base_lr

    def step(self) -> float:
        self.step_count += 1
        return self.base_lr

    @property
    def lr(self) -> float:
        return self.base_lr

def build_schedule(optimizer, config, total_steps_override: Optional[int]=None):
    optim_cfg = config.optim
    total = total_steps_override or optim_cfg.total_steps
    if optim_cfg.schedule == 'constant':
        return ConstantSchedule(optimizer, warmup_steps=optim_cfg.warmup_steps, total_steps=total, min_lr=optim_cfg.min_lr)
    return WarmupCosineSchedule(optimizer, warmup_steps=optim_cfg.warmup_steps, total_steps=total, min_lr=optim_cfg.min_lr)
