from __future__ import annotations
import statistics
import hashlib
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from ..env.verifiers import TaskItem
from ..models.backbone import format_prompt
from ..utils.logging import RunLogger
from .metrics import EnergyOrderingMetrics, TransitionMetrics, aggregate_accuracy, energy_ordering_metrics, pass_at_1, pass_at_k, span_transition_metrics, transition_metrics

@dataclass
class GenerationRecord:
    item_id: str
    suite: str
    text: str
    correct: bool
    latency_s: float
    output_tokens: int
    prompt_tokens: int
    energy: Optional[float] = None
    executed: bool = True

@dataclass
class SuiteResult:
    suite: str
    n: int
    pass_at_1: float
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    mean_latency_s: float = 0.0
    mean_output_tokens: float = 0.0
    mean_energy: Optional[float] = None
    records: List[GenerationRecord] = field(default_factory=list)
    truncation_rate: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        payload = {'n': float(self.n), 'pass@1': self.pass_at_1, 'latency_s': self.mean_latency_s, 'output_tokens': self.mean_output_tokens, 'truncation_rate': self.truncation_rate}
        for k, value in self.pass_at_k.items():
            if k != 1:
                payload[f'pass@{k}'] = value
        if self.mean_energy is not None:
            payload['direct_energy'] = self.mean_energy
        return payload

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _sample_seed(config, item, sample):
    key = f'{config.seed}:{item.family}:{item.item_id}:{sample}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')

def _generate(generator, prompt, config, seed):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        _sync()
        started = time.perf_counter()
        text = generator.generate(prompt, max_new_tokens=config.eval.max_new_tokens, temperature=config.rollout.temperature, top_p=config.rollout.top_p).strip()
        _sync()
        return (text, time.perf_counter() - started)

@torch.no_grad()
def evaluate_direct(generator, checker, scoring, items, config, family_to_index, samples_per_item=1, energy_head=False):
    records, flags_all, primary = ([], [], [])
    generator.eval()
    if items:
        for _ in range(config.eval.warmup_requests):
            _generate(generator, format_prompt(items[0].question), config, config.seed)
    for item in items:
        prompt = format_prompt(item.question)
        flags = []
        for sample in range(max(1, samples_per_item)):
            text, elapsed = _generate(generator, prompt, config, _sample_seed(config, item, sample))
            result = checker(item, text)
            energy = None
            if energy_head:
                h = scoring.pooled_text(prompt, text)
                z = scoring.lift_projection.init_coordinate(h)
                task = scoring.task_vector(family_to_index(item.family), device=h.device)
                energy = float(scoring.energy(h, z, task).cpu())
            record = GenerationRecord(item.item_id, item.family, text, bool(result.verified), elapsed, generator.token_count(text), generator.token_count(prompt), energy, result.executed)
            records.append(record)
            flags.append(bool(result.verified))
            if sample == 0:
                primary.append(record)
        flags_all.append(flags)
    result = SuiteResult(suite=items[0].family if items else 'empty', n=len(items), pass_at_1=pass_at_1([f[0] for f in flags_all]), pass_at_k={k: pass_at_k(flags_all, k) for k in config.eval.pass_k if k <= samples_per_item}, mean_latency_s=statistics.fmean((r.latency_s for r in primary)) if primary else 0.0, mean_output_tokens=statistics.fmean((r.output_tokens for r in primary)) if primary else 0.0, records=records)
    result.truncation_rate = sum((r.output_tokens >= config.eval.max_new_tokens for r in primary)) / max(1, len(primary))
    return result

@torch.no_grad()
def evaluate_reflection(generator, checker, scoring, items, config, family_to_index, passes=None):
    from ..models.backbone import REFLECT_TEMPLATE
    records = []
    generator.eval()
    if items:
        for _ in range(config.eval.warmup_requests):
            _generate(generator, format_prompt(items[0].question), config, config.seed)
    for item in items:
        prompt = format_prompt(item.question)
        current, elapsed, tokens = ('', 0.0, 0)
        for stage in range(passes or config.eval.reflection_passes):
            request = prompt if stage == 0 else REFLECT_TEMPLATE.format(question=item.question, answer=current)
            current, duration = _generate(generator, request, config, _sample_seed(config, item, stage))
            elapsed += duration
            tokens += generator.token_count(current)
        result = checker(item, current)
        records.append(GenerationRecord(item.item_id, item.family, current, bool(result.verified), elapsed, tokens, generator.token_count(prompt), executed=result.executed))
    return SuiteResult(suite=items[0].family if items else 'empty', n=len(items), pass_at_1=pass_at_1([r.correct for r in records]), mean_latency_s=statistics.fmean((r.latency_s for r in records)) if records else 0.0, mean_output_tokens=statistics.fmean((r.output_tokens for r in records)) if records else 0.0, records=records)

@torch.no_grad()
def evaluate_selection_budget(generator, checker, scoring, items: Sequence[TaskItem], config, family_to_index, budgets: Optional[Sequence[int]]=None, logger: Optional[RunLogger]=None) -> Dict[int, Dict[str, float]]:
    budgets = budgets or config.eval.selection_budgets
    device = next(scoring.parameters()).device
    results: Dict[int, Dict[str, float]] = {}
    pools: Dict[int, List[Tuple[str, float, bool]]] = {int(b): [] for b in budgets}
    for item in items:
        prompt = format_prompt(item.question)

        def score(text: str) -> float:
            h = scoring.pooled_text(prompt, text)
            z = scoring.lift_projection.init_coordinate(h)
            task = scoring.task_vector(family_to_index(item.family), device=device)
            return float(scoring.energy(h, z, task).detach().cpu())
        direct = generator.generate(prompt, max_new_tokens=config.rollout.max_new_tokens, temperature=config.rollout.temperature, top_p=config.rollout.top_p).strip()
        direct_correct = bool(checker(item, direct).verified)
        direct_energy = score(direct)
        extras: List[Tuple[str, bool]] = []
        max_budget = max((int(b) for b in budgets))
        for _ in range(max_budget):
            text = generator.generate(prompt, max_new_tokens=config.rollout.max_new_tokens, temperature=config.rollout.temperature, top_p=config.rollout.top_p).strip()
            extras.append((text, bool(checker(item, text).verified)))
        for budget in budgets:
            budget = int(budget)
            pool = [(direct, direct_energy, direct_correct)]
            for text, correct in extras[:budget]:
                pool.append((text, score(text), correct))
            best = min(pool, key=lambda entry: entry[1])
            pools[budget].append((best[0], best[1], best[2]))
    baseline = statistics.fmean([e[1] for e in pools[int(budgets[0])]])
    baseline_correct = pass_at_1([e[2] for e in pools[int(budgets[0])]])
    for budget in budgets:
        budget = int(budget)
        entries = pools[budget]
        mean_energy = statistics.fmean([e[1] for e in entries])
        accuracy = pass_at_1([e[2] for e in entries])
        results[budget] = {'mean_energy': mean_energy, 'energy_drop_pct': 100.0 * (baseline - mean_energy) / abs(baseline) if baseline else 0.0, 'accuracy': 100.0 * accuracy, 'delta_pp': 100.0 * (accuracy - baseline_correct)}
        if logger:
            logger.log({'stage': 'selection_budget', 'k_sel': budget, **results[budget]})
    return results

def evaluate_transitions(before: Sequence[bool], after: Sequence[bool]) -> TransitionMetrics:
    return transition_metrics(before, after)

def evaluate_span_transitions(before_valid: Sequence[bool], after_valid: Sequence[bool]) -> Dict[str, float]:
    return span_transition_metrics(before_valid, after_valid)

@torch.no_grad()
def evaluate_energy_ordering(generator, checker, scoring, items, config, family_to_index, candidates_per_item=4, logger=None, collector=None, calibration=None):
    if collector is None:
        raise ValueError('Ordering audits require the trained rollout collector')
    energies, labels, groups = ([], [], [])
    for item in items:
        trajectory = collector.collect(item, format_prompt(item.question), family_to_index(item.family))
        for candidate in trajectory.candidate_set.candidates:
            energies.append(candidate.energy)
            labels.append(int(bool(checker(item, candidate.text).verified)))
            groups.append(f'{item.family}:{item.item_id}')
    return energy_ordering_metrics(energies, labels, groups, n_bins=config.eval.energy_bins, calibration=calibration)

def run_evaluation(trainer, suites: Dict[str, Sequence[TaskItem]], config, logger: Optional[RunLogger]=None, with_reflection: bool=True) -> Dict[str, Dict[str, float]]:
    from ..models.backbone import format_prompt
    family_to_index = trainer.family_to_index
    results: Dict[str, Dict[str, float]] = {}
    aggregates: List[float] = []
    for name, items in suites.items():
        if not items:
            continue
        suite_result = evaluate_direct(trainer.generator, trainer.checker, trainer.scoring, items, config, family_to_index, samples_per_item=max(config.eval.pass_k))
        payload = suite_result.as_dict()
        results[name] = payload
        aggregates.append(suite_result.pass_at_1)
        if logger:
            logger.log({'stage': 'eval', 'suite': name, **payload})
    if aggregates:
        results['average'] = {f'pass@{k}': aggregate_accuracy({name: value[f'pass@{k}'] for name, value in results.items() if f'pass@{k}' in value}) for k in config.eval.pass_k}
    if with_reflection:
        for name, items in suites.items():
            if not items:
                continue
            reflection = evaluate_reflection(trainer.generator, trainer.checker, trainer.scoring, items, config, family_to_index)
            results.setdefault(name, {}).update({f'reflection_{k}': v for k, v in reflection.as_dict().items()})
    return results

def summarize_cost(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    latencies = [value['latency_s'] for key, value in results.items() if key != 'average' and 'latency_s' in value]
    tokens = [value['output_tokens'] for key, value in results.items() if key != 'average' and 'output_tokens' in value]
    return {'latency_s': statistics.fmean(latencies) if latencies else 0.0, 'output_tokens': statistics.fmean(tokens) if tokens else 0.0}
