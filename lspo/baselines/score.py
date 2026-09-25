from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import torch
from ..env.proposals import ProposalKernel
from ..env.verifiers import TaskItem
from ..models.backbone import format_prompt
from ..train.muon import clip_gradients
from .common import BaselineTrainer
@dataclass
class CorrectionTrace:
    prompt: str
    first_attempt: str
    target: str
    first_correct: bool
    target_correct: bool
    stage: int
class SCoReTrainer(BaselineTrainer):
    name = "score"
    def __init__(self, generator, checker, config, logger=None):
        super().__init__(generator, checker, config, logger)
        self.kernel = ProposalKernel(generator, config.rollout)
    def build_traces(self, items: Sequence[TaskItem]) -> List[CorrectionTrace]:
        traces: List[CorrectionTrace] = []
        for item in items:
            prompt = format_prompt(item.question)
            first = self.sample(prompt)
            first_result = self.checker(item, first)
            revision = self.kernel.reflect(item.question, first).text.strip()
            revision_result = self.checker(item, revision)
            if revision_result.verified:
                traces.append(
                    CorrectionTrace(
                        prompt=prompt, first_attempt=first, target=revision,
                        first_correct=bool(first_result.verified),
                        target_correct=True, stage=1,
                    )
                )
            if first_result.verified:
                traces.append(
                    CorrectionTrace(
                        prompt=prompt, first_attempt=first, target=first,
                        first_correct=True, target_correct=True, stage=2,
                    )
                )
        return traces
    def _loss(self, trace: CorrectionTrace) -> torch.Tensor:
        context = (
            f"{trace.prompt}\n\nPrevious attempt:\n{trace.first_attempt}\n\n"
            "Write the corrected answer, ending with \"Answer: <answer>\"."
        )
        logp = self.generator.logprob_answer(context, trace.target)
        length = max(1, self.generator.token_count(trace.target))
        return -(logp / length).mean()
    def train(self, items: Sequence[TaskItem], updates: Optional[int] = None) -> List[Dict]:
        optimizer, schedule, parameters = self.make_optimizer()
        updates = updates or self.config.optim.updates
        traces = self.build_traces(items)
        self.logger.info(f"[score] {len(traces)} revision traces")
        if not traces:
            self.logger.info("[score] no verified traces; nothing to train")
            return []
        half = max(1, updates // 2)
        for step in range(updates):
            stage = 1 if step < half else 2
            pool = [t for t in traces if stage == 2 or t.stage == 1] or traces
            trace = pool[step % len(pool)]
            loss = self._loss(trace)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_gradients(parameters, self.config.optim.grad_clip)
            optimizer.step()
            learning_rate = schedule.step()
            self.log(
                {"method": self.name, "stage": stage,
                 "loss": float(loss.detach().cpu()), "lr": learning_rate,
                 "grad_norm": grad_norm},
                step=step,
            )
        return self.state.history
