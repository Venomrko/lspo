from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from ..env.edit_distance import span_edit_distance
from ..env.proposals import ProposalKernel
from ..env.verifiers import TaskItem
from ..models.backbone import format_prompt
from ..utils.text import segment_reasoning_spans
from ..train.muon import clip_gradients
from .common import BaselineTrainer
@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    source: str = "verifier"
class DPOTrainer(BaselineTrainer):
    name = "dpo"
    def __init__(self, generator, checker, config, logger=None, beta: float = 0.1,
                 step_level: bool = False):
        super().__init__(generator, checker, config, logger)
        self.beta = beta
        self.step_level = step_level
        if step_level:
            self.name = "step_dpo"
        self.reference = copy.deepcopy(generator)
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
    def build_pairs(
        self, items: Sequence[TaskItem], pairs_per_item: int = 1
    ) -> List[PreferencePair]:
        rng = random.Random(self.config.seed)
        kernel = ProposalKernel(self.generator, self.config.rollout)
        pairs: List[PreferencePair] = []
        for item in items:
            prompt = format_prompt(item.question)
            attempts = [self.sample(prompt) for _ in range(2 * max(1, pairs_per_item))]
            verified = [(text, self.checker(item, text).verified) for text in attempts]
            chosen = [text for text, ok in verified if ok]
            rejected = [text for text, ok in verified if not ok]
            if not chosen or not rejected:
                continue
            for _ in range(pairs_per_item):
                winner = rng.choice(chosen)
                loser = rng.choice(rejected)
                if self.step_level:
                    winner, loser = self._step_level_pair(item, winner, loser, kernel)
                pairs.append(
                    PreferencePair(prompt=prompt, chosen=winner, rejected=loser)
                )
        return pairs
    def _step_level_pair(self, item: TaskItem, winner: str, loser: str,
                         kernel: ProposalKernel) -> Tuple[str, str]:
        spans_winner = segment_reasoning_spans(winner)
        spans_loser = segment_reasoning_spans(loser)
        limit = min(len(spans_winner), len(spans_loser))
        for index in range(limit):
            if span_edit_distance(spans_winner[index], spans_loser[index]) > 0.05:
                return (
                    "\n\n".join(spans_winner[: index + 1]),
                    "\n\n".join(spans_loser[: index + 1]),
                )
        return winner, loser
    def loss(self, pairs: Sequence[PreferencePair], batch_size: int = 4) -> Tuple[torch.Tensor, Dict]:
        if not pairs:
            return torch.zeros((), requires_grad=True), {"pairs": 0.0}
        total = torch.zeros((), device=next(self.generator.parameters()).device)
        count = 0
        for pair in pairs[:batch_size]:
            chosen_logp = self.generator.logprob_answer(pair.prompt, pair.chosen)
            rejected_logp = self.generator.logprob_answer(pair.prompt, pair.rejected)
            with torch.no_grad():
                ref_chosen = self.reference.logprob_answer(pair.prompt, pair.chosen)
                ref_rejected = self.reference.logprob_answer(pair.prompt, pair.rejected)
            margin = self.beta * (
                (chosen_logp - ref_chosen) - (rejected_logp - ref_rejected)
            )
            total = total - F.logsigmoid(margin).mean()
            count += 1
        if count == 0:
            return torch.zeros((), requires_grad=True), {"pairs": 0.0}
        loss = total / count
        with torch.no_grad():
            accuracy = float(
                (
                    self.generator.logprob_answer(pairs[0].prompt, pairs[0].chosen)
                    > self.generator.logprob_answer(pairs[0].prompt, pairs[0].rejected)
                ).float().cpu()
            )
        return loss, {"dpo_loss": float(loss.detach().cpu()), "pair_acc": accuracy}
    def train(self, items: Sequence[TaskItem], updates: Optional[int] = None) -> List[Dict]:
        optimizer, schedule, parameters = self.make_optimizer()
        updates = updates or self.config.optim.updates
        pairs = self.build_pairs(items, pairs_per_item=1)
        self.logger.info(f"[{self.name}] {len(pairs)} preference pairs")
        if not pairs:
            self.logger.info(f"[{self.name}] no pairs available; nothing to train")
            return []
        for step in range(updates):
            batch = [pairs[step % len(pairs)]]
            loss, diagnostics = self.loss(batch, batch_size=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_gradients(parameters, self.config.optim.grad_clip)
            optimizer.step()
            learning_rate = schedule.step()
            self.log(
                {"method": self.name, "loss": float(loss.detach().cpu()),
                 "lr": learning_rate, "grad_norm": grad_norm, **diagnostics},
                step=step,
            )
        return self.state.history
