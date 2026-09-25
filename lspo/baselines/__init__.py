from __future__ import annotations
from .common import BaselineTrainer, BaselineState
from .dpo import DPOTrainer, PreferencePair
from .grpo import GroupRollout, GRPOTrainer
from .score import CorrectionTrace, SCoReTrainer
BASELINES = ("dpo", "step_dpo", "grpo", "gdpo", "score")
def build_baseline(name: str, generator, checker, config, logger=None):
    if name == "dpo":
        return DPOTrainer(generator, checker, config, logger=logger)
    if name == "step_dpo":
        return DPOTrainer(generator, checker, config, logger=logger, step_level=True)
    if name == "grpo":
        return GRPOTrainer(generator, checker, config, logger=logger)
    if name == "gdpo":
        return GRPOTrainer(generator, checker, config, logger=logger, group_weighted=True)
    if name == "score":
        return SCoReTrainer(generator, checker, config, logger=logger)
    raise KeyError(f"unknown baseline {name!r}; known: {BASELINES}")
__all__ = [
    "BASELINES",
    "build_baseline",
    "BaselineTrainer",
    "BaselineState",
    "DPOTrainer",
    "PreferencePair",
    "GRPOTrainer",
    "GroupRollout",
    "SCoReTrainer",
    "CorrectionTrace",
]
