from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from .backbone import BackboneLM, build_frozen_encoder
from .energy import EnergyHead, SurfaceEnergyHead, TaskEmbedding
from .lift import IdentityLift, LiftProjection
from .transition import ACTION_NAMES, Action, ActionEmbedding, TransitionBlock
@dataclass
class LiftedState:
    text: str
    z: torch.Tensor
    h: torch.Tensor
    energy: Optional[float] = None
    step: int = 0
    h_gen: Optional[torch.Tensor] = None
    def detached(self) -> "LiftedState":
        return LiftedState(
            text=self.text,
            z=self.z.detach(),
            h=self.h.detach(),
            energy=self.energy,
            step=self.step,
            h_gen=None if self.h_gen is None else self.h_gen.detach(),
        )
class ScoringBranch(nn.Module):
    def __init__(self, config, encoder: BackboneLM):
        super().__init__()
        self.config = config
        model_cfg = config.model
        self.text_encoder = build_frozen_encoder(config, encoder)
        d_text = self.text_encoder.d_model
        lifted = config.variant not in {"no_lift", "no_lift_mlp"}
        if lifted:
            self.lift_projection: nn.Module = LiftProjection(
                d_in=d_text,
                d_z=model_cfg.d_z,
                hidden=model_cfg.lift_hidden,
                layers=model_cfg.lift_layers,
                activation=model_cfg.lift_activation,
                input_layernorm=model_cfg.lift_input_layernorm,
            )
        elif config.variant == "no_lift_mlp":
            self.lift_projection = LiftProjection(
                d_in=d_text,
                d_z=model_cfg.d_z,
                hidden=model_cfg.lift_hidden,
                layers=model_cfg.lift_layers,
                activation=model_cfg.lift_activation,
                input_layernorm=model_cfg.lift_input_layernorm,
            )
        else:
            self.lift_projection = IdentityLift(d_z=model_cfg.d_z)
        self.action_embedding = ActionEmbedding(dim=model_cfg.action_embed_dim)
        self.transition_block = TransitionBlock(
            d_h=d_text,
            d_z=model_cfg.d_z,
            action_dim=model_cfg.action_embed_dim,
            hidden=model_cfg.transition_hidden,
            blocks=model_cfg.transition_blocks,
            activation=model_cfg.transition_activation,
        )
        self.task_embedding = TaskEmbedding(
            n_tasks=len(model_cfg.task_families), dim=model_cfg.d_task
        )
        if lifted:
            self.energy_head: nn.Module = EnergyHead(
                d_text=d_text,
                d_z=model_cfg.d_z,
                d_task=model_cfg.d_task,
                hidden=model_cfg.energy_hidden,
                layers=model_cfg.energy_layers,
                activation=model_cfg.energy_activation,
            )
        else:
            self.energy_head = SurfaceEnergyHead(
                d_text=d_text,
                d_task=model_cfg.d_task,
                hidden=model_cfg.energy_hidden,
                layers=model_cfg.energy_layers,
                activation=model_cfg.energy_activation,
            )
        self.d_text = d_text
        self.d_z = model_cfg.d_z
        self.pooling = model_cfg.text_pooling
    def pooled_text(self, prompt: str, answer: str) -> torch.Tensor:
        return self.text_encoder.pooled(prompt, answer, pooling=self.pooling)
    def pooled_texts_batch(
        self, prompts: Sequence[str], answers: Sequence[str]
    ) -> torch.Tensor:
        return self.text_encoder.pooled_texts_batch(
            prompts, answers, pooling=self.pooling
        )
    def task_vector(self, task_id: int, device: Optional[torch.device] = None) -> torch.Tensor:
        index = torch.tensor([int(task_id)], device=device)
        return self.task_embedding(index)[0]
    def task_embeddings(
        self, task_ids: Sequence[int], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        index = torch.as_tensor(list(task_ids), dtype=torch.long, device=device)
        return self.task_embedding(index)
    def init_state(self, prompt: str, answer: str, task_id: int = 0) -> LiftedState:
        h = self.pooled_text(prompt, answer)
        z = self.lift_projection.init_coordinate(h)
        return LiftedState(text=answer, z=z, h=h, energy=None, step=0)
    def base_for_action(
        self,
        action: int,
        z_t: torch.Tensor,
        h_next: torch.Tensor,
    ) -> torch.Tensor:
        if int(action) == Action.SWITCH and self.config.rollout.switch_resets_coordinate:
            return self.lift_projection.init_coordinate(h_next)
        return z_t
    def transition(
        self,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        h_next: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        b_t = self.base_for_action(action, z_t, h_next)
        action_tensor = torch.tensor([int(action)], device=h_t.device, dtype=torch.long)
        return self.transition_block.step(
            h_t=h_t, h_next=h_next, b_t=b_t, action=action_tensor,
            action_embedding=self.action_embedding,
        )
    def transition_batch(
        self,
        h_t: torch.Tensor,
        z_t: torch.Tensor,
        h_next: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        actions = torch.as_tensor(actions, device=h_t.device, dtype=torch.long)
        if self.config.rollout.switch_resets_coordinate:
            is_switch = (actions == int(Action.SWITCH)).unsqueeze(-1)
            b_t = torch.where(
                is_switch, self.lift_projection.init_coordinate(h_next), z_t
            )
        else:
            b_t = z_t
        embedding = self.action_embedding(actions)
        delta = self.transition_block.forward(h_next - h_t, b_t, embedding)
        return self.transition_block.output_norm(b_t + delta)
    def energy(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        task_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.energy_head(h, z, task_embedding)
    @torch.no_grad()
    def score_states(
        self,
        states: Sequence[LiftedState],
        task_id: int = 0,
    ) -> List[float]:
        if not states:
            return []
        device = states[0].h.device
        h = torch.stack([s.h for s in states])
        z = torch.stack([s.z for s in states])
        task = self.task_vector(task_id, device=device).expand(h.shape[0], -1)
        return self.energy(h, z, task).detach().cpu().tolist()
    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
    def frozen_parameter_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]
    def param_groups_for_energy_fit(self):
        trainable = self.config.energy_fit.freezes
        groups: List[Dict] = []
        for name, parameter in self.named_parameters():
            if name.startswith("text_encoder"):
                continue
            groups.append({"name": name, "params": [parameter]})
        del trainable
        return groups
    def as_deployable(self) -> Dict[str, nn.Module]:
        return {
            "text_encoder": self.text_encoder,
            "lift_projection": self.lift_projection,
            "transition_block": self.transition_block,
            "energy_head": self.energy_head,
            "action_embedding": self.action_embedding,
        }
def action_names() -> Tuple[str, ...]:
    return ACTION_NAMES
class ScoreFunction:
    def __init__(self, branch: ScoringBranch, task_id: int = 0):
        self.branch = branch
        self.task_id = task_id
    @torch.no_grad()
    def __call__(self, state: LiftedState) -> float:
        task = self.branch.task_vector(self.task_id, device=state.h.device)
        value = self.branch.energy(state.h, state.z, task)
        return float(value.detach().cpu())
    def copy(self) -> "ScoreFunction":
        return ScoreFunction(self.branch, self.task_id)
def deep_copy_branch(branch: ScoringBranch) -> ScoringBranch:
    clone = copy.deepcopy(branch)
    clone.freeze_all()
    return clone
