from __future__ import annotations
import math
from typing import Iterable, List, Optional
import torch
from torch.optim.optimizer import Optimizer
try:
    from torch.distributed.tensor import DTensor
    _HAS_DTENSOR = True
except ImportError:
    DTensor = None
    _HAS_DTENSOR = False

def is_sharded(tensor: torch.Tensor) -> bool:
    return _HAS_DTENSOR and isinstance(tensor, DTensor)

def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if is_sharded(tensor):
        return tensor.full_tensor()
    return tensor

def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if is_sharded(tensor):
        return tensor.to_local()
    return tensor

def gather_like(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if not is_sharded(like):
        return value
    distributed = DTensor.from_local(value, like.device_mesh, like.placements, shape=like.shape, stride=like.stride())
    return distributed.full_tensor()

def scatter_like(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if not is_sharded(like):
        return value
    from torch.distributed.tensor import distribute_tensor
    return distribute_tensor(value, like.device_mesh, like.placements)

def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int=5, eps: float=1e-07) -> torch.Tensor:
    a, b, c = (3.4445, -4.775, 2.0315)
    x = grad.to(torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(grad.dtype)

class Muon(Optimizer):

    def __init__(self, params: Iterable, lr: float=2e-05, momentum: float=0.95, nesterov: bool=True, ns_steps: int=5, weight_decay: float=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        with torch.enable_grad():
            loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                state = self.state[parameter]
                self._matrix_step(parameter, grad, state, group, lr, momentum)
        return loss

    def _matrix_step(self, parameter, grad, state, group, lr, momentum) -> None:
        local_grad = local_tensor(grad)
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(local_grad)
        buffer = state['momentum_buffer']
        buffer.mul_(momentum).add_(local_grad, alpha=1 - momentum)
        if group['nesterov']:
            local_update = local_grad.add(buffer, alpha=momentum)
        else:
            local_update = buffer
        update = gather_like(local_update, parameter)
        original_shape = update.shape
        update = update.reshape(1, -1) if update.ndim < 2 else update.reshape(update.shape[0], -1)
        rows, cols = update.shape
        scale = max(1.0, rows / max(1, cols)) ** 0.5
        orthogonal = zeropower_via_newtonschulz5(update, group['ns_steps'])
        local_orthogonal = local_tensor(scatter_like(orthogonal.reshape(original_shape), parameter))
        local_parameter = local_tensor(parameter)
        if group['weight_decay']:
            local_parameter.mul_(1.0 - lr * group['weight_decay'])
        local_parameter.add_(local_orthogonal, alpha=-lr * scale)

def build_optimizer(parameters: List[torch.nn.Parameter], optimizer: str='muon', lr: float=2e-05, momentum: float=0.95, nesterov: bool=True, ns_steps: int=5, weight_decay: float=0.1) -> torch.optim.Optimizer:
    parameters = [p for p in parameters if p.requires_grad]
    if optimizer != 'muon':
        raise ValueError('This implementation uses Muon for every parameter group')
    return Muon(parameters, lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)

@torch.no_grad()
def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float=1.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    if any((is_sharded(p.grad) for p in params)):
        import torch.distributed as dist
        total = torch.zeros((), device=local_tensor(params[0].grad).device)
        for parameter in params:
            local = local_tensor(parameter.grad)
            contribution = local.detach().float().pow(2).sum()
            if not is_sharded(parameter.grad) and dist.is_initialized():
                contribution = contribution / dist.get_world_size()
            total = total + contribution
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        norm = float(total.sqrt().item())
        coefficient = min(1.0, max_norm / (norm + 1e-06))
        for parameter in params:
            parameter.grad.mul_(coefficient)
        return norm
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))
