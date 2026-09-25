from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ..models.scoring import LiftedState
@dataclass
class Candidate:
    text: str
    state: LiftedState
    energy: float
    verified: Optional[bool] = None
    origin: str = "direct"
    step: int = 0
    def key(self) -> Tuple[float, int]:
        return (self.energy, self.step)
@dataclass
class CandidateSet:
    prompt: str
    task_id: int
    k_max: int = 7
    candidates: List[Candidate] = field(default_factory=list)
    def add(self, text: str, state: LiftedState, energy: float,
            origin: str = "revise", step: int = 0) -> None:
        self.candidates.append(
            Candidate(text=text, state=state, energy=energy, origin=origin, step=step)
        )
        if len(self.candidates) > self.k_max:
            direct = [c for c in self.candidates if c.origin == "direct"]
            others = sorted(
                (c for c in self.candidates if c.origin != "direct"),
                key=lambda c: c.energy,
            )
            keep = max(0, self.k_max - len(direct))
            self.candidates = direct[:1] + others[:keep]
    def __len__(self) -> int:
        return len(self.candidates)
    def energies(self) -> List[float]:
        return [c.energy for c in self.candidates]
    def best(self) -> Candidate:
        return min(self.candidates, key=lambda c: c.energy)
    def select(
        self,
        use_verifier: bool = True,
        tie_epsilon: float = 1e-6,
    ) -> Candidate:
        if not self.candidates:
            raise ValueError("empty candidate set")
        ordered = sorted(self.candidates, key=lambda c: c.energy)
        best = ordered[0]
        if not use_verifier:
            return best
        tied = [c for c in ordered if c.energy <= best.energy + tie_epsilon]
        verified = [c for c in tied if c.verified]
        if verified:
            return min(verified, key=lambda c: c.energy)
        return best
@dataclass
class SelectionOutcome:
    text: str
    energy: float
    candidate_count: int
    energy_span: float
    origin: str
    step: int
def select_transfer_target(
    candidate_set: CandidateSet,
    use_verifier: bool = True,
) -> SelectionOutcome:
    chosen = candidate_set.select(use_verifier=use_verifier)
    energies = candidate_set.energies()
    return SelectionOutcome(
        text=chosen.text,
        energy=chosen.energy,
        candidate_count=len(candidate_set),
        energy_span=(max(energies) - min(energies)) if energies else 0.0,
        origin=chosen.origin,
        step=chosen.step,
    )
