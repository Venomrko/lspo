from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from ..env.rollout import Trajectory
from ..models.scoring import ScoringBranch
from ..utils.logging import RunLogger
@dataclass
class InternalizationPair:
    prompt: str
    target: str
    source_energy: float
    origin: str
    improved: bool
    task_id: int = 0
def build_internalization_pairs(
    trajectories: Sequence[Trajectory],
) -> List[InternalizationPair]:
    pairs: List[InternalizationPair] = []
    for trajectory in trajectories:
        if trajectory.selection is None:
            continue
        pairs.append(
            InternalizationPair(
                prompt=trajectory.prompt,
                target=trajectory.selection.text,
                source_energy=trajectory.selection.energy,
                origin=trajectory.selection.origin,
                improved=trajectory.selection.origin != "direct",
                task_id=trajectory.task_id,
            )
        )
    return pairs
def internalization_loss(
    generator,
    pairs: Sequence[InternalizationPair],
    config,
    logger: Optional[RunLogger] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    empty = torch.zeros((), requires_grad=True)
    if not pairs or not config.objective.use_internalization:
        return empty, {"si_loss": 0.0, "si_pairs": 0.0}
    log_probs: List[torch.Tensor] = []
    per_token: List[float] = []
    lengths: List[int] = []
    for pair in pairs:
        value = generator.logprob_answer(pair.prompt, pair.target)
        length = max(1, generator.token_count(pair.target))
        log_probs.append(value.squeeze())
        lengths.append(length)
        per_token.append(float(value.detach().cpu()) / length)
    stacked = torch.stack(log_probs)
    reduction = getattr(config.objective, "si_reduction", "mean")
    if reduction == "sum":
        loss = -stacked.mean()
    else:
        weights = torch.tensor(
            [1.0 / length for length in lengths],
            dtype=stacked.dtype,
            device=stacked.device,
        )
        loss = -(stacked * weights).mean()
    diagnostics = {
        "si_loss": float(loss.detach().cpu()),
        "si_logprob": float(stacked.detach().mean().cpu()),
        "si_logprob_per_token": float(sum(per_token) / max(1, len(per_token))),
        "si_pairs": float(len(pairs)),
        "si_target_tokens": float(sum(lengths) / max(1, len(lengths))),
        "si_improved_frac": float(
            sum(1 for p in pairs if p.improved) / max(1, len(pairs))
        ),
    }
    del logger
    return loss, diagnostics
def internalization_gap(
    generator,
    scoring: ScoringBranch,
    pairs: Sequence[InternalizationPair],
    device: Optional[torch.device] = None,
) -> float:
    if not pairs:
        return 0.0
    device = device or next(scoring.parameters()).device
    gaps: List[float] = []
    with torch.no_grad():
        for pair in pairs:
            direct = generator.generate(
                pair.prompt, max_new_tokens=64, temperature=0.0, top_p=1.0
            ).strip()
            h_direct = scoring.pooled_text(pair.prompt, direct)
            z_direct = scoring.lift_projection.init_coordinate(h_direct)
            task = scoring.task_vector(pair.task_id, device=device)
            energy_direct = float(scoring.energy(h_direct, z_direct, task).cpu())
            h_target = scoring.pooled_text(pair.prompt, pair.target)
            z_target = scoring.lift_projection.init_coordinate(h_target)
            energy_target = float(scoring.energy(h_target, z_target, task).cpu())
            gaps.append(energy_direct - energy_target)
    return float(sum(gaps) / len(gaps))
