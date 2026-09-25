from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from ..env.edit_distance import edit_distance
from ..env.proposals import ProposalKernel, apply_bounded_patch
from ..env.verifiers import StateRecord, TaskItem, build_preference_pairs
from ..models.backbone import REVISE_TEMPLATE, SWITCH_TEMPLATE
from ..models.energy import pairwise_logistic_loss, pairwise_margin_loss
from ..models.scoring import ScoringBranch
from ..models.transition import Action
from ..utils.logging import RunLogger

@dataclass
class SampledState:
    prompt: str
    task_id: int
    task_family: str
    text: str
    step: int
    cum_edit: float
    verified: bool
    source: str = 'rollout'
    hs: Tuple[torch.Tensor, ...] = ()
    actions: Tuple[int, ...] = ()
    direct_text: str = ''

    def as_record(self) -> StateRecord:
        return StateRecord(prompt=self.prompt, question='', text=self.text, task_id=self.task_id, task_family=self.task_family, verified=self.verified, step=self.step, edit_cost=self.cum_edit, source=self.source)

def _perturb(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) < 4:
        return text
    index = rng.randrange(len(tokens))
    mode = rng.random()
    if mode < 0.34:
        tokens[index] = str(rng.randint(0, 9))
    elif mode < 0.67:
        del tokens[index]
    else:
        tokens.insert(index, tokens[index])
    return ' '.join(tokens)

def _restate(text: str) -> str:
    return text + '\n\nLet me restate the same conclusion once more for clarity.'

@torch.no_grad()
def sample_states(generator, scoring: ScoringBranch, kernel: ProposalKernel, checker, items: Sequence[TaskItem], family_to_index, config, rng: Optional[random.Random]=None) -> List[SampledState]:
    rng = rng or random.Random(config.seed)
    per_prompt = max(1, config.energy_fit.states_per_prompt)
    batch_size = max(1, getattr(config.optim, 'rollout_batch_size', 8) or 8)
    states: List[SampledState] = []
    for start in range(0, len(items), batch_size):
        chunk = list(items[start:start + batch_size])
        prompts = [_format(item.question) for item in chunk]
        task_ids = [family_to_index(item.family) for item in chunk]
        direct_texts = [text.strip() or ' ' for text in generator.generate_batch(prompts, max_new_tokens=config.rollout.max_new_tokens, temperature=config.rollout.temperature, top_p=config.rollout.top_p)]
        current = list(direct_texts)
        cumulative = [0.0] * len(chunk)
        chain_actions: List[List[int]] = [[] for _ in chunk]
        chain_h: List[List[torch.Tensor]] = [[scoring.pooled_text(prompt, text)] for prompt, text in zip(prompts, direct_texts)]

        def record(index: int, text: str, step: int, source: str='rollout', verified: Optional[bool]=None) -> None:
            if verified is None:
                verified = bool(checker(chunk[index], text).verified)
            states.append(SampledState(prompt=prompts[index], task_id=task_ids[index], task_family=chunk[index].family, text=text, step=step, cum_edit=cumulative[index], verified=verified, source=source, hs=tuple(chain_h[index]), actions=tuple(chain_actions[index]), direct_text=direct_texts[index]))
        for index, text in enumerate(direct_texts):
            record(index, text, 0)
        for step in range(per_prompt - 1):
            actions = [rng.choice([Action.REVISE, Action.SWITCH]) for _ in chunk]
            revise_rows = [i for i, a in enumerate(actions) if a == Action.REVISE]
            switch_rows = [i for i, a in enumerate(actions) if a == Action.SWITCH]
            next_texts = list(current)
            if revise_rows:
                proposals = generator.generate_batch([REVISE_TEMPLATE.format(question=chunk[i].question, answer=current[i]) for i in revise_rows], max_new_tokens=config.rollout.proposal_max_new_tokens, temperature=config.rollout.temperature, top_p=config.rollout.top_p)
                for row, text in zip(revise_rows, proposals):
                    patched, _spans = apply_bounded_patch(current[row], text.strip(), max_patches=config.rollout.revise_max_patches, max_span_chars=config.rollout.revise_max_span_chars)
                    next_texts[row] = patched.strip() or current[row]
            if switch_rows:
                proposals = generator.generate_batch([SWITCH_TEMPLATE.format(question=chunk[i].question) for i in switch_rows], max_new_tokens=config.rollout.proposal_max_new_tokens, temperature=config.rollout.switch_temperature, top_p=config.rollout.top_p)
                for row, text in zip(switch_rows, proposals):
                    candidate = text.strip()
                    next_texts[row] = candidate if candidate and candidate != current[row].strip() else current[row]
            for index in range(len(chunk)):
                cumulative[index] += edit_distance(current[index], next_texts[index], config.rollout)
                chain_actions[index].append(int(actions[index]))
                chain_h[index].append(scoring.pooled_text(prompts[index], next_texts[index]))
            current = next_texts
            for index, text in enumerate(current):
                record(index, text, step + 1)
        for index in range(len(chunk)):
            parent = current[index]
            for source, text in (('noisy', _perturb(parent, rng)), ('longer_no_gain', _restate(parent))):
                if text == parent:
                    continue
                hidden = scoring.pooled_text(prompts[index], text)
                states.append(SampledState(prompt=prompts[index], task_id=task_ids[index], task_family=chunk[index].family, text=text, step=len(chain_actions[index]) + 1, cum_edit=cumulative[index] + edit_distance(parent, text, config.rollout), verified=bool(checker(chunk[index], text).verified), source=source, hs=tuple(chain_h[index]) + (hidden,), actions=tuple(chain_actions[index]) + (int(Action.REVISE),), direct_text=direct_texts[index]))
    return states

def _format(question: str) -> str:
    from ..models.backbone import format_prompt
    return format_prompt(question)

def recompute_coordinates(scoring: ScoringBranch, state: SampledState, task_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    z = scoring.lift_projection.init_coordinate(state.hs[0])
    for t, action in enumerate(state.actions):
        z = scoring.transition(h_t=state.hs[t], z_t=z, h_next=state.hs[t + 1], action=int(action))
    return (state.hs[len(state.actions)], z)

