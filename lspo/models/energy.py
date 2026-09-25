from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import MLP
class TaskEmbedding(nn.Module):
    def __init__(self, n_tasks: int, dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(n_tasks, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(task_ids)
class EnergyHead(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_z: int = 8,
        d_task: int = 64,
        hidden: int = 1024,
        layers: int = 3,
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
    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([h, z, task_embedding], dim=-1)
        return self.net(x).squeeze(-1)
class SurfaceEnergyHead(nn.Module):
    def __init__(
        self,
        d_text: int,
        d_task: int = 64,
        hidden: int = 1024,
        layers: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        self.net = MLP(
            d_in=d_text + d_task,
            d_out=1,
            hidden=hidden,
            depth=layers,
            activation=activation,
        )
    def forward(self, h, z, task_embedding) -> torch.Tensor:
        del z
        return self.net(torch.cat([h, task_embedding], dim=-1)).squeeze(-1)
def pairwise_margin_loss(
    energy_winner: torch.Tensor,
    energy_loser: torch.Tensor,
    margin: float = 0.20,
    reduction: str = "mean",
) -> torch.Tensor:
    loss = torch.clamp(margin - (energy_loser - energy_winner), min=0.0)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
def pairwise_logistic_loss(
    energy_winner: torch.Tensor,
    energy_loser: torch.Tensor,
    margin: float = 0.20,
    reduction: str = "mean",
) -> torch.Tensor:
    loss = torch.nn.functional.softplus(margin + energy_winner - energy_loser)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
def energy_ordering_accuracy(
    energy_winner: torch.Tensor, energy_loser: torch.Tensor
) -> torch.Tensor:
    return (energy_winner < energy_loser).float().mean()
