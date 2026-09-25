from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import MLP
class ValueHead(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_z: int = 8,
        d_task: int = 64,
        hidden: int = 512,
        layers: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        self.net = MLP(
            d_in=d_text + d_z + d_task,
            d_out=1,
            hidden=hidden,
            depth=layers,
            activation=activation,
        )
    def forward(self, h: torch.Tensor, z: torch.Tensor, task_embedding: torch.Tensor):
        return self.net(torch.cat([h, z, task_embedding], dim=-1)).squeeze(-1)
def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip: float = 0.2,
) -> torch.Tensor:
    unclipped = (values - returns) ** 2
    clipped_values = old_values + (values - old_values).clamp(-clip, clip)
    clipped = (clipped_values - returns) ** 2
    return torch.max(unclipped, clipped).mean()
