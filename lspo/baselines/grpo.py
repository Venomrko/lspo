from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torch
from ..env.verifiers import TaskItem
from ..models.backbone import format_prompt
from ..train.muon import clip_gradients
from .common import BaselineTrainer
@dataclass
class GroupRollout:
    prompt: str
    answers: List[str]
    rewards: List[float]
class GRPOTrainer(BaselineTrainer):
    name = "grpo"
    def __init__(
        self,
        generator,
        checker,
        config,
        logger=None,
        group_size: int = 4,
        beta: float = 0.02,
        group_weighted: bool = False,
    ):
        super().__init__(generator, checker, config, logger)
        self.group_size = max(2, group_size)
        self.beta = beta
        self.group_weighted = group_weighted
        if group_weighted:
            self.name = "gdpo"
        self.reference = copy.deepcopy(generator)
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
    @torch.no_grad()
    def collect_group(self, item: TaskItem) -> GroupRollout:
        prompt = format_prompt(item.question)
        answers: List[str] = []
        rewards: List[float] = []
        for _ in range(self.group_size):
            text = self.sample(prompt, temperature=self.config.rollout.switch_temperature)
            result = self.checker(item, text)
            answers.append(text)
            rewards.append(1.0 if result.verified else float(result.score))
        return GroupRollout(prompt=prompt, answers=answers, rewards=rewards)
    def advantages(self, rewards: Sequence[float]) -> torch.Tensor:
        tensor = torch.tensor(list(rewards), dtype=torch.float32)
        mean = tensor.mean()
        std = tensor.std(unbiased=False)
        standardized = (tensor - mean) / (std + 1e-8)
        if not self.group_weighted:
            return standardized
        weight = float(torch.sigmoid(std * 4.0 - 1.0))
        return standardized * weight
    def loss(self, group: GroupRollout) -> tuple:
        device = next(self.generator.parameters()).device
        advantages = self.advantages(group.rewards).to(device)
        total = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)
        for answer, advantage in zip(group.answers, advantages):
            logp = self.generator.logprob_answer(group.prompt, answer)
            with torch.no_grad():
                ref_logp = self.reference.logprob_answer(group.prompt, answer)
            ratio = torch.exp(logp - logp.detach())
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - 0.2, 1 + 0.2) * advantage
            total = total - torch.min(unclipped, clipped)
            log_ratio = ref_logp - logp
            kl_total = kl_total + (torch.exp(log_ratio) - log_ratio - 1.0)
        loss = (total + self.beta * kl_total) / max(1, len(group.answers))
        diagnostics = {
            "grpo_loss": float(loss.detach().cpu()),
            "reward_mean": float(sum(group.rewards) / len(group.rewards)),
            "reward_std": float(
                torch.tensor(group.rewards).std(unbiased=False).item()
            ),
            "kl": float((kl_total / max(1, len(group.answers))).detach().cpu()),
        }
        return loss, diagnostics
    def train(self, items: Sequence[TaskItem], updates: Optional[int] = None) -> List[Dict]:
        optimizer, schedule, parameters = self.make_optimizer()
        updates = updates or self.config.optim.updates
        for step in range(updates):
            item = items[step % len(items)]
            group = self.collect_group(item)
            loss, diagnostics = self.loss(group)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_gradients(parameters, self.config.optim.grad_clip)
            optimizer.step()
            learning_rate = schedule.step()
            self.log(
                {"method": self.name, "lr": learning_rate, "grad_norm": grad_norm,
                 **diagnostics},
                step=step,
            )
        return self.state.history
