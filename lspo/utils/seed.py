from __future__ import annotations
import os
import random
from typing import Optional
import numpy as np
import torch
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
def resolve_dtype(name: str, device: Optional[torch.device] = None) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype = table.get(name.lower(), torch.float32)
    if device is not None and device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype
def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )
