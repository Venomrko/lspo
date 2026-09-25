from __future__ import annotations
import contextlib
import os
from dataclasses import dataclass
from typing import Optional
import torch
_ACCELERATORS = ("cuda", "mps")
def cuda_available() -> bool:
    return torch.cuda.is_available()
def resolve_device(requested: str = "auto") -> torch.device:
    if requested in ("auto", "", None):
        for candidate in _ACCELERATORS:
            if _available(candidate):
                return torch.device(candidate)
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not cuda_available():
        raise RuntimeError(
            "device='cuda' was requested but no CUDA device is visible.  Installed "
            f"torch is {torch.__version__} (CUDA build: {torch.version.cuda}).  A "
            "'+cpu' build cannot use the GPU; reinstall from the CUDA index, e.g. "
            "`pip install torch --index-url https://download.pytorch.org/whl/cu124`. "
            "Use device='auto' or 'cpu' to run without a GPU."
        )
    return device
def _available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    return False
def is_accelerator(device: torch.device) -> bool:
    return device.type in _ACCELERATORS
def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name in ("auto", "", None):
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    table = {
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
    }
    if name.lower() not in table:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(table)}")
    return table[name.lower()]
def bf16_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_bf16_supported()
@contextlib.contextmanager
def autocast(device: torch.device, dtype: Optional[torch.dtype], enabled: bool = True):
    if not enabled or dtype is None or device.type == "cpu":
        yield
        return
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield
def needs_grad_scaler(dtype: Optional[torch.dtype]) -> bool:
    return dtype == torch.float16
def make_grad_scaler(device: torch.device, dtype: Optional[torch.dtype], enabled: bool = True):
    use = enabled and needs_grad_scaler(dtype) and device.type == "cuda"
    return torch.amp.GradScaler("cuda", enabled=use)
@dataclass
class DeviceReport:
    device: str
    name: str = ""
    total_gb: float = 0.0
    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    compute_dtype: str = "float32"
    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "device_name": self.name,
            "device_total_gb": round(self.total_gb, 2),
            "peak_allocated_gb": round(self.peak_allocated_gb, 3),
            "peak_reserved_gb": round(self.peak_reserved_gb, 3),
            "compute_dtype": self.compute_dtype,
        }
def device_report(device: torch.device, dtype: Optional[torch.dtype] = None) -> DeviceReport:
    report = DeviceReport(
        device=str(device),
        compute_dtype=str(dtype).replace("torch.", "") if dtype else "float32",
    )
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        report.name = properties.name
        report.total_gb = properties.total_memory / 2 ** 30
        report.peak_allocated_gb = torch.cuda.max_memory_allocated(index) / 2 ** 30
        report.peak_reserved_gb = torch.cuda.max_memory_reserved(index) / 2 ** 30
    return report
def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
def configure_backends(device: torch.device, allow_tf32: bool = True) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def best_attention_implementation(device: torch.device, requested: str = "auto") -> str:
    if requested not in ("auto", "", None):
        return requested
    if device.type == "cuda":
        try:
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            return "sdpa"
    return "sdpa"
def describe_environment() -> dict:
    payload = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }
    if torch.cuda.is_available():
        payload["names"] = [
            torch.cuda.get_device_properties(i).name
            for i in range(torch.cuda.device_count())
        ]
    return payload
