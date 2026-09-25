from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
_ACTIVATIONS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
}
def get_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()
class MLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int,
        depth: int = 2,
        activation: str = "silu",
        input_layernorm: bool = False,
        output_layernorm: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("MLP depth must be >= 1")
        layers: List[nn.Module] = []
        if input_layernorm:
            layers.append(nn.LayerNorm(d_in))
        current = d_in
        for _ in range(depth - 1):
            layers.append(nn.Linear(current, hidden))
            layers.append(get_activation(activation))
            current = hidden
        layers.append(nn.Linear(current, d_out))
        if output_layernorm:
            layers.append(nn.LayerNorm(d_out))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str = "silu"):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = get_activation(activation)
        self.fc2 = nn.Linear(width, width)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + h
def maybe_layernorm(dim: int, enabled: bool) -> Optional[nn.Module]:
    return nn.LayerNorm(dim) if enabled else None
