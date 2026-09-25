from __future__ import annotations
from enum import IntEnum
import torch
import torch.nn as nn
from .blocks import ResidualBlock, get_activation
class Action(IntEnum):
    REVISE = 0
    SWITCH = 1
    STOP = 2
    @property
    def is_text_changing(self) -> bool:
        return self is not Action.STOP
ACTION_NAMES = ("revise", "switch", "stop")
class ActionEmbedding(nn.Module):
    def __init__(self, dim: int = 128, n_actions: int = len(ACTION_NAMES)):
        super().__init__()
        self.embedding = nn.Embedding(n_actions, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.embedding.weight.requires_grad_(False)
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.embedding(actions)
class TransitionBlock(nn.Module):
    def __init__(
        self,
        d_h: int,
        d_z: int = 8,
        action_dim: int = 128,
        hidden: int = 1024,
        blocks: int = 2,
        activation: str = "silu",
    ):
        super().__init__()
        self.d_z = d_z
        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_h + d_z + action_dim),
            nn.Linear(d_h + d_z + action_dim, hidden),
            get_activation(activation),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, activation) for _ in range(blocks)]
        )
        self.output_proj = nn.Linear(hidden, d_z)
        self.output_norm = nn.LayerNorm(d_z)
    def forward(
        self,
        delta_h: torch.Tensor,
        b_t: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([delta_h, b_t, action_embedding], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)
    def step(
        self,
        h_t: torch.Tensor,
        h_next: torch.Tensor,
        b_t: torch.Tensor,
        action: torch.Tensor,
        action_embedding: nn.Module,
    ) -> torch.Tensor:
        embedding = action_embedding(action)
        if b_t.dim() == 1 and embedding.dim() == 2:
            embedding = embedding.squeeze(0)
        delta = self.forward(h_next - h_t, b_t, embedding)
        return self.output_norm(b_t + delta)
