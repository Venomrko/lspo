from __future__ import annotations
import copy
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import MLP
from .transition import ACTION_NAMES, Action
class ActionPolicy(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_z: int = 8,
        d_task: int = 64,
        hidden: int = 256,
        depth: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        self.net = MLP(
            d_in=d_text + d_z + d_task,
            d_out=len(ACTION_NAMES),
            hidden=hidden,
            depth=depth,
            activation=activation,
        )
    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([h, z, task_embedding], dim=-1))
    def distribution(self, h, z, task_embedding) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(h, z, task_embedding))
    def log_prob(self, h, z, task_embedding, actions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(h, z, task_embedding)
        return F.log_softmax(logits, dim=-1).gather(
            -1, actions.to(torch.long).unsqueeze(-1)
        ).squeeze(-1)
    def entropy(self, h, z, task_embedding) -> torch.Tensor:
        logits = self.forward(h, z, task_embedding)
        return torch.distributions.Categorical(logits=logits).entropy()
def build_reference_policy(policy: ActionPolicy) -> ActionPolicy:
    reference = copy.deepcopy(policy)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference
def sample_actions(
    policy: ActionPolicy,
    h: torch.Tensor,
    z: torch.Tensor,
    task_embedding: torch.Tensor,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    greedy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = policy(h, z, task_embedding)
    if greedy:
        actions = logits.argmax(dim=-1)
    else:
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    log_prob = F.log_softmax(logits, dim=-1).gather(
        -1, actions.unsqueeze(-1)
    ).squeeze(-1)
    return actions, log_prob
def action_name(action: int) -> str:
    return ACTION_NAMES[int(action)]
def is_text_changing(action: int) -> bool:
    return Action(int(action)).is_text_changing
