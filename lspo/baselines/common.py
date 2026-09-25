from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
from ..train.muon import build_optimizer
from ..train.scheduler import build_schedule
from ..utils.logging import RunLogger

@dataclass
class BaselineState:
    history: List[Dict] = field(default_factory=list)
    step: int = 0

class BaselineTrainer:
    name = 'baseline'

    def __init__(self, generator, checker, config, logger: Optional[RunLogger]=None):
        self.generator = generator
        self.checker = checker
        self.config = config
        self.logger = logger or RunLogger(config.output_dir, f'{config.run_name}_{self.name}', verbose=config.verbose)
        self.state = BaselineState()

    def make_optimizer(self, parameters=None):
        parameters = parameters or [p for p in self.generator.parameters() if p.requires_grad]
        optimizer = build_optimizer(parameters, optimizer=self.config.optim.optimizer, lr=self.config.optim.lr, momentum=self.config.optim.momentum, nesterov=self.config.optim.nesterov, ns_steps=self.config.optim.ns_steps, weight_decay=self.config.optim.weight_decay)
        schedule = build_schedule(optimizer, self.config, total_steps_override=self.config.optim.updates)
        return (optimizer, schedule, parameters)

    @torch.no_grad()
    def sample(self, prompt: str, temperature: Optional[float]=None) -> str:
        return self.generator.generate(prompt, max_new_tokens=self.config.rollout.max_new_tokens, temperature=self.config.rollout.temperature if temperature is None else temperature, top_p=self.config.rollout.top_p).strip()

    def log(self, record: Dict, step: int) -> None:
        self.state.history.append(record)
        self.logger.log(record, step=step)
