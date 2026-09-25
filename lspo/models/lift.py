from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import MLP
class LiftProjection(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_z: int = 8,
        hidden: int = 1024,
        layers: int = 2,
        activation: str = "gelu",
        input_layernorm: bool = True,
    ):
        super().__init__()
        self.d_z = d_z
        self.net = MLP(
            d_in=d_in,
            d_out=d_z,
            hidden=hidden,
            depth=layers,
            activation=activation,
            input_layernorm=input_layernorm,
        )
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
    def init_coordinate(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h)
class IdentityLift(nn.Module):
    def __init__(self, d_z: int = 8):
        super().__init__()
        self.d_z = d_z
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(h.shape[:-1] + (self.d_z,), dtype=h.dtype, device=h.device)
    def init_coordinate(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h)
class RandomLift(nn.Module):
    def __init__(self, d_in: int, d_z: int = 8, mode: str = "random", seed: int = 0):
        super().__init__()
        self.d_z = d_z
        self.mode = mode
        generator = torch.Generator().manual_seed(seed)
        if mode in {"random_projection", "frozen_projection"}:
            weight = torch.randn(d_z, d_in, generator=generator) / (d_in ** 0.5)
            self.register_buffer("weight", weight)
        elif mode == "random":
            self.register_buffer(
                "weight", torch.randn(d_z, generator=generator)
            )
        else:
            raise ValueError(f"unknown RandomLift mode {mode!r}")
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode in {"random_projection", "frozen_projection"}:
            return h @ self.weight.t()
        return self.weight.unsqueeze(0).expand(*h.shape[:-1], self.d_z)
    def init_coordinate(self, h: torch.Tensor) -> torch.Tensor:
        return self.forward(h)
