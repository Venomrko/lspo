from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from ..models.backbone import REVISE_TEMPLATE, SWITCH_TEMPLATE
from ..models.scoring import LiftedState, ScoringBranch
from ..models.transition import Action
from .candidates import CandidateSet, SelectionOutcome, select_transfer_target
from .edit_distance import edit_distance
from .proposals import ProposalKernel, apply_bounded_patch
from .verifiers import TaskItem, VerificationResult

def reward_for_transition(energy_t: float, energy_next: float, text_t: str, text_next: str, action: int, config) -> Tuple[float, float]:
    if Action(int(action)) == Action.STOP:
        return (0.0, 0.0)
    distance = edit_distance(text_t, text_next, config.rollout)
    cost = config.objective.lambda_edit * distance + config.objective.lambda_step
    if not config.objective.use_progress_reward:
        return (-cost, distance)
    return (energy_t - energy_next - cost, distance)

def reward_terms(energy_t: torch.Tensor, energy_next: torch.Tensor, distances: torch.Tensor, config) -> torch.Tensor:
    cost = config.objective.lambda_edit * distances + config.objective.lambda_step
    if not config.objective.use_progress_reward:
        return -cost
    return energy_t - energy_next - cost

def trajectory_return(energies: Sequence[float], distances: Sequence[float], config) -> float:
    n_moves = len(distances)
    if n_moves == 0:
        return 0.0
    energy_term = energies[0] - energies[-1] if config.objective.use_progress_reward else 0.0
    edit_term = config.objective.lambda_edit * float(sum(distances))
    step_term = config.objective.lambda_step * n_moves
    return energy_term - edit_term - step_term

@dataclass
class Transition:
    step: int
    action: int
    action_name: str
    state_text: str
    next_text: str
    reward: float
    edit_distance: float
    logp_behavior: float
    logp_reference: float
    done: bool
    h: torch.Tensor
    z: torch.Tensor
    h_next: torch.Tensor
    z_next: torch.Tensor
    h_gen: Optional[torch.Tensor]
    h_gen_next: Optional[torch.Tensor]
    energy_t: float
    energy_next: float
    value: float = 0.0
    prompt: str = ''
    task_id: int = 0

    @property
    def is_stop(self) -> bool:
        return Action(int(self.action)) == Action.STOP

@dataclass
class Trajectory:
    item: TaskItem
    prompt: str
    task_id: int
    direct_text: str
    transitions: List[Transition] = field(default_factory=list)
    candidate_set: Optional[CandidateSet] = None
    selection: Optional[SelectionOutcome] = None
    initial_verification: Optional[VerificationResult] = None
    terminated_by_stop: bool = False
    reached_horizon: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def executed_moves(self) -> int:
        return sum((1 for t in self.transitions if not t.is_stop))

    @property
    def total_edit(self) -> float:
        return float(sum((t.edit_distance for t in self.transitions)))

    def energies_executed(self) -> List[float]:
        if not self.transitions:
            return []
        values = [self.transitions[0].energy_t]
        for transition in self.transitions:
            if not transition.is_stop:
                values.append(transition.energy_next)
        return values

    def distances(self) -> List[float]:
        return [t.edit_distance for t in self.transitions if not t.is_stop]

def apply_terminal_reward(trajectory: Trajectory, config) -> None:
    if not trajectory.transitions:
        return
    total = trajectory_return(trajectory.energies_executed(), trajectory.distances(), config)
    last = len(trajectory.transitions) - 1
    for index, transition in enumerate(trajectory.transitions):
        transition.reward = total if index == last else 0.0

class RolloutCollector:

    def __init__(self, generator, scoring: ScoringBranch, policy, reference_policy, kernel: ProposalKernel, checker, config):
        self.generator = generator
        self.scoring = scoring
        self.policy = policy
        self.reference_policy = reference_policy
        self.kernel = kernel
        self.checker = checker
        self.config = config
        self.rollout = config.rollout

    def _device(self) -> torch.device:
        return next(self.scoring.parameters()).device

    def _energies(self, h: torch.Tensor, z: torch.Tensor, task_embedding: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            energy = self.scoring.energy(h, z, task_embedding)
        noise_std = float(getattr(self.config.objective, 'energy_noise_std', 0.0) or 0.0)
        if noise_std > 0:
            energy = energy + torch.randn_like(energy) * noise_std
        return energy

    def _action_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        allowed = tuple(getattr(self.rollout, 'allowed_actions', ('revise', 'switch', 'stop')))
        if len(allowed) >= 3:
            return None
        mask = torch.full((3,), float('-inf'), device=device)
        for index, name in enumerate(('revise', 'switch', 'stop')):
            if name in allowed:
                mask[index] = 0.0
        return mask

    def _policy_states_batch(self, prompts: Sequence[str], texts: Sequence[str]) -> Optional[torch.Tensor]:
        if self.config.model.policy_state_source != 'shared':
            return None
        with torch.no_grad():
            return self.generator.pooled_texts_batch(list(prompts), list(texts))

    def collect(self, item: TaskItem, prompt: str, task_id: int) -> Trajectory:
        return self.collect_batch([item], [prompt], [task_id])[0]

    def collect_batch(self, items: Sequence[TaskItem], prompts: Sequence[str], task_ids: Sequence[int]) -> List[Trajectory]:
        cfg = self.config
        device = self._device()
        batch = len(items)
        if batch == 0:
            return []
        task_emb = self.scoring.task_embeddings(list(task_ids), device=device)
        direct_texts = [text.strip() for text in self.generator.generate_batch(list(prompts), max_new_tokens=cfg.rollout.max_new_tokens, temperature=cfg.rollout.temperature, top_p=cfg.rollout.top_p)]
        direct_texts = [text if text else ' ' for text in direct_texts]
        trajectories: List[Trajectory] = []
        for item, prompt, task_id, text in zip(items, prompts, task_ids, direct_texts):
            trajectory = Trajectory(item=item, prompt=prompt, task_id=task_id, direct_text=text, prompt_tokens=self.generator.token_count(prompt), output_tokens=self.generator.token_count(text))
            trajectory.initial_verification = self.checker(item, text)
            trajectories.append(trajectory)
        h = self.scoring.pooled_texts_batch(list(prompts), direct_texts)
        z = self.scoring.lift_projection.init_coordinate(h)
        energies = self._energies(h, z, task_emb)
        texts = list(direct_texts)
        energy_list = energies.tolist()
        for index, trajectory in enumerate(trajectories):
            candidate_set = CandidateSet(prompt=prompts[index], task_id=task_ids[index], k_max=cfg.rollout.k_candidates)
            state = LiftedState(text=texts[index], z=z[index].detach(), h=h[index].detach(), energy=energy_list[index], step=0)
            candidate_set.add(texts[index], state, energy_list[index], origin='direct', step=0)
            trajectory.candidate_set = candidate_set
        h_gen = self._policy_states_batch(prompts, texts)
        active: List[int] = list(range(batch))
        for step in range(cfg.rollout.t_max):
            if not active:
                break
            index = torch.as_tensor(active, dtype=torch.long, device=device)
            active_prompts = [prompts[i] for i in active]
            active_texts = [texts[i] for i in active]
            active_task = task_emb[index]
            h_cur, z_cur = (h[index], z[index])
            e_cur = energies[index]
            h_pol = h_gen[index] if h_gen is not None else h_cur
            with torch.no_grad():
                logits = self.policy(h_pol, z_cur, active_task)
                mask = self._action_mask(logits.device)
                masked = logits + mask if mask is not None else logits
                log_probs = torch.log_softmax(masked, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)
                logp_behavior = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                ref_logits = self.reference_policy(h_cur, z_cur, active_task)
                if mask is not None:
                    ref_logits = ref_logits + mask
                logp_reference = torch.log_softmax(ref_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            action_list = actions.tolist()
            behavior_list = logp_behavior.tolist()
            reference_list = logp_reference.tolist()
            e_cur_list = e_cur.tolist()
            stop_rows = [r for r, a in enumerate(action_list) if Action(a) == Action.STOP]
            moving_rows = [r for r, a in enumerate(action_list) if Action(a) != Action.STOP]
            for r in stop_rows:
                i = active[r]
                trajectory = trajectories[i]
                trajectory.transitions.append(Transition(step=step, action=int(action_list[r]), action_name='stop', state_text=texts[i], next_text=texts[i], reward=0.0, edit_distance=0.0, logp_behavior=behavior_list[r], logp_reference=reference_list[r], done=True, h=h[i].detach(), z=z[i].detach(), h_next=h[i].detach(), z_next=z[i].detach(), h_gen=None if h_gen is None else h_gen[i].detach(), h_gen_next=None if h_gen is None else h_gen[i].detach(), energy_t=e_cur_list[r], energy_next=e_cur_list[r], prompt=prompts[i], task_id=task_ids[i]))
                trajectory.terminated_by_stop = True
            active = [active[r] for r in moving_rows]
            if not active:
                break
            move_actions = [action_list[r] for r in moving_rows]
            move_texts = [texts[i] for i in active]
            move_prompts = [prompts[i] for i in active]
            move_index = torch.as_tensor(active, dtype=torch.long, device=device)
            move_task = task_emb[move_index]
            next_texts = self._propose_texts(items, active, move_actions, move_texts)
            h_sel = h[move_index]
            z_sel = z[move_index]
            h_next = self.scoring.pooled_texts_batch(move_prompts, next_texts)
            actions_tensor = torch.as_tensor(move_actions, dtype=torch.long, device=device)
            z_next = self.scoring.transition_batch(h_t=h_sel, z_t=z_sel, h_next=h_next, actions=actions_tensor)
            e_next = self._energies(h_next, z_next, move_task)
            h_gen_next = self._policy_states_batch(move_prompts, next_texts)
            moving_index = torch.as_tensor(moving_rows, dtype=torch.long, device=device)
            distances = [edit_distance(move_texts[r], next_texts[r], cfg.rollout) for r in range(len(active))]
            distance_tensor = torch.as_tensor(distances, dtype=e_next.dtype, device=device)
            rewards = reward_terms(e_cur[moving_index], e_next, distance_tensor, cfg)
            reward_list = rewards.tolist()
            e_next_list = e_next.tolist()
            done = step == cfg.rollout.t_max - 1
            for r, i in enumerate(active):
                trajectory = trajectories[i]
                trajectory.transitions.append(Transition(step=step, action=int(move_actions[r]), action_name='revise' if Action(move_actions[r]) == Action.REVISE else 'switch', state_text=move_texts[r], next_text=next_texts[r], reward=reward_list[r], edit_distance=distances[r], logp_behavior=behavior_list[moving_rows[r]], logp_reference=reference_list[moving_rows[r]], done=done, h=h_sel[r].detach(), z=z_sel[r].detach(), h_next=h_next[r].detach(), z_next=z_next[r].detach(), h_gen=None if h_gen is None else h_gen[i].detach(), h_gen_next=None if h_gen_next is None else h_gen_next[r].detach(), energy_t=e_cur_list[moving_rows[r]], energy_next=e_next_list[r], prompt=move_prompts[r], task_id=task_ids[i]))
                state = LiftedState(text=next_texts[r], z=z_next[r].detach(), h=h_next[r].detach(), energy=e_next_list[r], step=step + 1)
                trajectory.candidate_set.add(next_texts[r], state, e_next_list[r], origin=trajectory.transitions[-1].action_name, step=step + 1)
                trajectory.output_tokens += self.generator.token_count(next_texts[r])
                if done:
                    trajectory.reached_horizon = True
            texts = list(texts)
            for r, i in enumerate(active):
                texts[i] = next_texts[r]
            h = h.clone()
            z = z.clone()
            energies = energies.clone()
            h[move_index] = h_next.detach()
            z[move_index] = z_next.detach()
            energies[move_index] = e_next.detach()
            if h_gen is not None and h_gen_next is not None:
                h_gen = h_gen.clone()
                h_gen[move_index] = h_gen_next.detach()
            if done:
                active = []
        for trajectory in trajectories:
            self._label_candidates(trajectory.candidate_set, trajectory.item)
            selector = getattr(cfg.objective, 'target_selector', 'energy')
            if selector == 'verifier':
                trajectory.selection = self._select_by_verifier(trajectory.candidate_set)
            else:
                trajectory.selection = select_transfer_target(trajectory.candidate_set, use_verifier=True)
            if cfg.objective.terminal_reward:
                apply_terminal_reward(trajectory, cfg)
        return trajectories

    def _propose_texts(self, items: Sequence[TaskItem], active: Sequence[int], move_actions: Sequence[int], move_texts: Sequence[str]) -> List[str]:
        cfg = self.rollout
        next_texts: List[Optional[str]] = [None] * len(active)
        revise_rows: List[int] = []
        revise_prompts: List[str] = []
        switch_rows: List[int] = []
        switch_prompts: List[str] = []
        switch_candidates = max(1, cfg.switch_candidates)
        for r, (i, action) in enumerate(zip(active, move_actions)):
            if Action(int(action)) == Action.REVISE:
                revise_rows.append(r)
                revise_prompts.append(REVISE_TEMPLATE.format(question=items[i].question, answer=move_texts[r]))
            else:
                for _ in range(switch_candidates):
                    switch_rows.append(r)
                    switch_prompts.append(SWITCH_TEMPLATE.format(question=items[i].question))
        if revise_prompts:
            raw = self.generator.generate_batch(revise_prompts, max_new_tokens=cfg.proposal_max_new_tokens, temperature=cfg.temperature, top_p=cfg.top_p)
            for r, text in zip(revise_rows, raw):
                patched, _spans = apply_bounded_patch(move_texts[r], text.strip(), max_patches=cfg.revise_max_patches, max_span_chars=cfg.revise_max_span_chars)
                next_texts[r] = patched
        if switch_prompts:
            raw = self.generator.generate_batch(switch_prompts, max_new_tokens=cfg.proposal_max_new_tokens, temperature=cfg.switch_temperature, top_p=cfg.top_p)
            grouped: Dict[int, List[str]] = {}
            for r, text in zip(switch_rows, raw):
                grouped.setdefault(r, []).append(text.strip())
            for r, options in grouped.items():
                chosen = move_texts[r]
                for option in options:
                    if option and option != move_texts[r].strip():
                        chosen = option
                        break
                else:
                    chosen = options[-1] if options else move_texts[r]
                next_texts[r] = chosen
        return [text if text and text.strip() else move_texts[r] for r, text in enumerate(next_texts)]

    def _label_candidates(self, candidate_set: Optional[CandidateSet], item: TaskItem) -> None:
        if candidate_set is None:
            return
        for candidate in candidate_set.candidates:
            if candidate.verified is None:
                candidate.verified = bool(self.checker(item, candidate.text))

    def _select_by_verifier(self, candidate_set: CandidateSet) -> SelectionOutcome:
        verified = [c for c in candidate_set.candidates if c.verified]
        if verified:
            chosen = min(verified, key=lambda c: (c.step, c.energy))
        else:
            chosen = candidate_set.best()
        energies = candidate_set.energies()
        return SelectionOutcome(text=chosen.text, energy=chosen.energy, candidate_count=len(candidate_set), energy_span=max(energies) - min(energies) if energies else 0.0, origin=chosen.origin, step=chosen.step)

def make_prompt_fn(generator):
    from ..models.backbone import format_prompt
    del generator
    return format_prompt

def summarize_action_statistics(trajectories: Sequence[Trajectory]) -> Dict[str, float]:
    total = sum((len(t.transitions) for t in trajectories))
    if total == 0:
        return {}
    counts = {'revise': 0, 'switch': 0, 'stop': 0}
    decreases = {'revise': 0, 'switch': 0}
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            counts[transition.action_name] = counts.get(transition.action_name, 0) + 1
            if not transition.is_stop:
                if transition.energy_next < transition.energy_t:
                    decreases[transition.action_name] += 1
    reached = sum((1 for t in trajectories if t.reached_horizon))
    return {'revise_pct': 100.0 * counts['revise'] / total, 'switch_pct': 100.0 * counts['switch'] / total, 'stop_pct': 100.0 * counts['stop'] / total, 'reach_tmax_pct': 100.0 * reached / max(1, len(trajectories)), 'revise_energy_decrease_pct': 100.0 * decreases['revise'] / counts['revise'] if counts['revise'] else 0.0, 'switch_energy_decrease_pct': 100.0 * decreases['switch'] / counts['switch'] if counts['switch'] else 0.0}
