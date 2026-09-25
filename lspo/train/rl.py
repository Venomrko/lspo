from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from ..env.rollout import Trajectory, Transition
from ..models.scoring import ScoringBranch
from ..models.transition import ACTION_NAMES
from ..models.value import clipped_value_loss
from ..utils.logging import RunLogger

@dataclass
class UpdateBatch:
    trajectories: List[Trajectory]
    transitions: List[Transition]
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor

    def __len__(self) -> int:
        return len(self.transitions)

@torch.no_grad()
def annotate_values(value_head, scoring: ScoringBranch, transitions: Sequence[Transition], config) -> None:
    if not transitions:
        return
    device = transitions[0].h.device
    h = torch.stack([t.h for t in transitions]).to(device)
    z = torch.stack([t.z for t in transitions]).to(device)
    tasks = torch.stack([scoring.task_vector(t.task_id, device=device) for t in transitions])
    values = value_head(h, z, tasks)
    for transition, value in zip(transitions, values):
        transition.value = float(value.detach().cpu())

def compute_gae(trajectories: Sequence[Trajectory], config, normalize: Optional[bool]=None) -> Tuple[List[float], List[float]]:
    objective = config.objective
    advantages: List[float] = []
    returns: List[float] = []
    for trajectory in trajectories:
        transitions = trajectory.transitions
        count = len(transitions)
        trajectory_advantages = [0.0] * count
        running = 0.0
        for index in reversed(range(count)):
            transition = transitions[index]
            value_t = transition.value
            value_next = 0.0 if transition.done else transitions[index + 1].value
            delta = transition.reward + objective.gamma * (0.0 if transition.done else 1.0) * value_next - value_t
            running = delta + objective.gamma * objective.gae_lambda * (0.0 if transition.done else 1.0) * running
            trajectory_advantages[index] = running
        advantages.extend(trajectory_advantages)
        returns.extend([adv + t.value for adv, t in zip(trajectory_advantages, transitions)])
    should_normalize = objective.advantage_normalize if normalize is None else normalize
    if should_normalize and advantages:
        tensor = torch.tensor(advantages, dtype=torch.float32)
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
            stats = torch.tensor([tensor.sum(), tensor.square().sum(), tensor.numel()], device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            mean = float(stats[0] / stats[2])
            variance = max(0.0, float(stats[1] / stats[2]) - mean * mean)
            tensor = (tensor - mean) / (variance ** 0.5 + 1e-08)
        elif tensor.numel() > 1:
            tensor = (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-08)
        advantages = tensor.tolist()
    return (advantages, returns)

def build_update_batch(trajectories: Sequence[Trajectory], config) -> UpdateBatch:
    transitions: List[Transition] = []
    for trajectory in trajectories:
        transitions.extend(trajectory.transitions)
    advantages, returns = compute_gae(trajectories, config)
    return UpdateBatch(trajectories=list(trajectories), transitions=transitions, advantages=torch.tensor(advantages or [0.0], dtype=torch.float32), returns=torch.tensor(returns or [0.0], dtype=torch.float32), old_values=torch.tensor([t.value for t in transitions] or [0.0], dtype=torch.float32))

def policy_representations(generator, scoring: ScoringBranch, transitions: Sequence[Transition], config, with_grad: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not transitions:
        empty = torch.zeros(0)
        return (empty, empty, empty)
    device = transitions[0].h.device
    z = torch.stack([t.z for t in transitions]).to(device)
    if config.model.policy_state_source == 'shared':
        if with_grad:
            h = generator.pooled_texts_batch([t.prompt for t in transitions], [t.state_text for t in transitions], pooling=config.model.text_pooling)
        else:
            stacked = [t.h_gen if t.h_gen is not None else t.h for t in transitions]
            h = torch.stack(stacked).to(device)
    else:
        h = torch.stack([t.h for t in transitions]).to(device)
    task = torch.stack([scoring.task_vector(t.task_id, device=device) for t in transitions])
    return (h, z, task)

def controller_loss(generator, scoring: ScoringBranch, policy, reference_policy, batch: UpdateBatch, config, logger: Optional[RunLogger]=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not batch.transitions:
        zero = torch.zeros((), requires_grad=True)
        return (zero, {})
    objective = config.objective
    h, z, task = policy_representations(generator, scoring, batch.transitions, config, with_grad=True)
    logits = policy(h, z, task)
    log_probs = torch.log_softmax(logits, dim=-1)
    actions = torch.tensor([t.action for t in batch.transitions], dtype=torch.long, device=logits.device)
    selected_logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        reference_logits = reference_policy(torch.stack([t.h for t in batch.transitions]).to(h.device), z.detach(), task.detach())
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    advantages = batch.advantages.to(logits.device)
    behaviour_logp = torch.tensor([t.logp_behavior for t in batch.transitions], dtype=torch.float32, device=logits.device)
    ratio = torch.exp(selected_logp - behaviour_logp)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - objective.clip_eps, 1.0 + objective.clip_eps) * advantages
    surrogate = torch.min(unclipped, clipped)
    probabilities = log_probs.exp()
    kl = (probabilities * (log_probs - reference_log_probs)).sum(dim=-1)
    policy_loss = -surrogate.mean()
    kl_penalty = objective.beta * kl.mean()
    loss = policy_loss + kl_penalty
    with torch.no_grad():
        diagnostics = {'policy_loss': float(policy_loss.detach().cpu()), 'kl': float(kl.mean().detach().cpu()), 'kl_max': float(kl.max().detach().cpu()), 'ratio_mean': float(ratio.mean().detach().cpu()), 'ratio_max': float(ratio.max().detach().cpu()), 'clip_frac': float(((ratio - 1.0).abs() > objective.clip_eps).float().mean().cpu()), 'entropy': float((-(probabilities * log_probs).sum(dim=-1)).mean().detach().cpu()), 'advantage_mean': float(advantages.mean().cpu()), 'advantage_std': float(advantages.std(unbiased=False).cpu() if advantages.numel() > 1 else 0.0)}
        for index, name in enumerate(ACTION_NAMES):
            diagnostics[f'action_{name}_pct'] = float(100.0 * (actions == index).float().mean().cpu())
    del logger
    return (loss, diagnostics)

def value_head_loss(value_head, scoring: ScoringBranch, batch: UpdateBatch, config) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not batch.transitions:
        return (torch.zeros((), requires_grad=True), {})
    device = next(value_head.parameters()).device
    h = torch.stack([t.h for t in batch.transitions]).to(device)
    z = torch.stack([t.z for t in batch.transitions]).to(device)
    task = torch.stack([scoring.task_vector(t.task_id, device=device) for t in batch.transitions])
    values = value_head(h, z, task)
    returns = batch.returns.to(device)
    old_values = batch.old_values.to(device)
    loss = clipped_value_loss(values, old_values, returns, config.objective.value_clip)
    with torch.no_grad():
        explained = 1.0 - float(((returns - values).var(unbiased=False) / (returns.var(unbiased=False) + 1e-08)).cpu())
    del scoring
    return (loss, {'value_loss': float(loss.detach().cpu()), 'value_explained_var': explained})

def make_policy_optimizer_inputs(generator, policy, value_head, config) -> List[torch.nn.Parameter]:
    parameters: List[torch.nn.Parameter] = []
    parameters.extend((p for p in generator.parameters() if p.requires_grad))
    parameters.extend((p for p in policy.parameters() if p.requires_grad))
    parameters.extend((p for p in value_head.parameters() if p.requires_grad))
    del config
    return parameters
