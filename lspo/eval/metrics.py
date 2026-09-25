from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

def pass_at_k(correct: Sequence[Sequence[bool]], k: int) -> float:
    estimates: List[float] = []
    for samples in correct:
        n = len(samples)
        if n < k:
            continue
        c = int(sum((bool(s) for s in samples)))
        if c == 0:
            estimates.append(0.0)
            continue
        if n - c < k:
            estimates.append(1.0)
            continue
        log_ratio = sum((math.log(n - c - i) - math.log(n - i) for i in range(k)))
        estimates.append(1.0 - math.exp(log_ratio))
    return float(np.mean(estimates)) if estimates else float('nan')

def pass_at_1(correct: Sequence[bool]) -> float:
    return float(np.mean([1.0 if c else 0.0 for c in correct])) if len(correct) else float('nan')

@dataclass
class TransitionMetrics:
    n: int
    repair_pct: float
    damage_pct: float
    net_gain: float
    correct_to_correct: int
    incorrect_to_incorrect: int

    def as_dict(self) -> Dict[str, float]:
        return {'n': float(self.n), 'repair_pct': self.repair_pct, 'damage_pct': self.damage_pct, 'net_gain': self.net_gain, 'c_to_c': float(self.correct_to_correct), 'w_to_w': float(self.incorrect_to_incorrect)}

def transition_metrics(before: Sequence[bool], after: Sequence[bool]) -> TransitionMetrics:
    if len(before) != len(after):
        raise ValueError('before/after must be paired')
    n = len(before)
    if n == 0:
        return TransitionMetrics(0, 0.0, 0.0, 0.0, 0, 0)
    repairs = sum((1 for b, a in zip(before, after) if not b and a))
    damage = sum((1 for b, a in zip(before, after) if b and (not a)))
    c2c = sum((1 for b, a in zip(before, after) if b and a))
    w2w = sum((1 for b, a in zip(before, after) if not b and (not a)))
    repair_pct = 100.0 * repairs / n
    damage_pct = 100.0 * damage / n
    return TransitionMetrics(n=n, repair_pct=repair_pct, damage_pct=damage_pct, net_gain=repair_pct - damage_pct, correct_to_correct=c2c, incorrect_to_incorrect=w2w)

def span_transition_metrics(before_valid: Sequence[bool], after_valid: Sequence[bool]) -> Dict[str, float]:
    if len(before_valid) != len(after_valid):
        raise ValueError('span labels must be paired')
    total = len(before_valid)
    if total == 0:
        return {k: 0.0 for k in ('initial_pV', 'V_to_V', 'I_to_V', 'V_to_I', 'net_span_gain')}
    initially_valid = [i for i, v in enumerate(before_valid) if v]
    initially_invalid = [i for i, v in enumerate(before_valid) if not v]
    preserved = sum((1 for i in initially_valid if after_valid[i]))
    repaired = sum((1 for i in initially_invalid if after_valid[i]))
    damaged = sum((1 for i in initially_valid if not after_valid[i]))
    p_v = len(initially_valid) / total
    preservation = preserved / len(initially_valid) if initially_valid else 0.0
    repair = repaired / len(initially_invalid) if initially_invalid else 0.0
    damage = damaged / len(initially_valid) if initially_valid else 0.0
    return {'initial_pV': 100.0 * p_v, 'V_to_V': 100.0 * preservation, 'I_to_V': 100.0 * repair, 'V_to_I': 100.0 * damage, 'net_span_gain': 100.0 * ((1.0 - p_v) * repair - p_v * damage)}

def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        ranks[order[index:end + 1]] = average
        index = end + 1
    return ranks

def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    positives = scores_array[labels_array == 1]
    negatives = scores_array[labels_array == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float('nan')
    all_scores = np.concatenate([positives, negatives])
    ranks = _ranks(all_scores)
    rank_sum = ranks[:len(positives)].sum()
    return float((rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives)))

def spearman_correlation(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(scores) < 3:
        return float('nan')
    ranks_scores = _ranks(np.asarray(scores, dtype=np.float64))
    ranks_labels = _ranks(np.asarray(labels, dtype=np.float64))
    if ranks_scores.std() == 0 or ranks_labels.std() == 0:
        return float('nan')
    return float(np.corrcoef(ranks_scores, ranks_labels)[0, 1])

def kendall_tau_b(scores: Sequence[float], labels: Sequence[int]) -> float:
    n = len(scores)
    if n < 3:
        return float('nan')
    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.float64)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = scores_array[i] - scores_array[j]
            dy = labels_array[i] - labels_array[j]
            if dx == 0 and dy == 0:
                continue
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float((concordant - discordant) / denominator) if denominator else float('nan')

def pairwise_preference_accuracy(scores: Sequence[float], labels: Sequence[int], groups: Optional[Sequence]=None) -> float:
    if groups is None:
        groups = [0] * len(scores)
    buckets: Dict[object, List[int]] = {}
    for index, group in enumerate(groups):
        buckets.setdefault(group, []).append(index)
    agreements: List[float] = []
    for indices in buckets.values():
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                i, j = (indices[a_pos], indices[b_pos])
                if labels[i] == labels[j]:
                    continue
                correct, incorrect = (i, j) if labels[i] == 1 else (j, i)
                agreements.append(1.0 if scores[correct] > scores[incorrect] else 0.0)
    return float(np.mean(agreements)) if agreements else float('nan')

def monotone_bins(scores: Sequence[float], labels: Sequence[int], n_bins: int=5) -> int:
    if len(scores) < n_bins * 2:
        return 0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    labels_array = np.asarray(labels, dtype=np.float64)[order]
    chunks = np.array_split(labels_array, n_bins)
    means = [float(chunk.mean()) if len(chunk) else 0.0 for chunk in chunks]
    preserved = 1
    for a, b in zip(means, means[1:]):
        if b <= a + 1e-09:
            preserved += 1
        else:
            break
    return preserved

def _isotonic(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values)
    sorted_targets = targets[order]
    weights = np.ones(n, dtype=np.float64)
    level_weights: List[float] = []
    level_totals: List[float] = []
    for target, weight in zip(sorted_targets, weights):
        level_weights.append(float(weight))
        level_totals.append(float(target) * float(weight))
        while len(level_totals) > 1 and level_totals[-2] / level_weights[-2] > level_totals[-1] / level_weights[-1]:
            total = level_totals.pop() + level_totals.pop()
            weight_sum = level_weights.pop() + level_weights.pop()
            level_totals.append(total)
            level_weights.append(weight_sum)
    fitted = np.empty(n, dtype=np.float64)
    cursor = 0
    for level_total, level_weight in zip(level_totals, level_weights):
        size = int(level_weight)
        fitted[cursor:cursor + size] = level_total / level_weight
        cursor += size
    inverse = np.empty(n, dtype=np.float64)
    inverse[order] = fitted
    return inverse

def expected_calibration_error(fit_scores: Sequence[float], fit_labels: Sequence[int], test_scores: Sequence[float], test_labels: Sequence[int], n_bins: int=10) -> float:
    if len(fit_scores) < 8 or len(test_scores) < 4:
        return float('nan')
    calibration = _isotonic(np.asarray(fit_scores, dtype=np.float64), np.asarray(fit_labels, dtype=np.float64))
    order = np.argsort(np.asarray(fit_scores, dtype=np.float64))
    grid = np.asarray(fit_scores, dtype=np.float64)[order]
    mapped = calibration[order]
    test = np.asarray(test_scores, dtype=np.float64)
    indices = np.searchsorted(grid, test).clip(0, len(grid) - 1)
    predicted = mapped[indices]
    labels_array = np.asarray(test_labels, dtype=np.float64)
    order_test = np.argsort(-predicted)
    predicted_sorted = predicted[order_test]
    labels_sorted = labels_array[order_test]
    chunks_pred = np.array_split(predicted_sorted, n_bins)
    chunks_true = np.array_split(labels_sorted, n_bins)
    ece = 0.0
    for chunk_pred, chunk_true in zip(chunks_pred, chunks_true):
        if len(chunk_pred) == 0:
            continue
        ece += len(chunk_pred) / len(predicted) * abs(float(chunk_pred.mean()) - float(chunk_true.mean()))
    return float(ece)

@dataclass
class EnergyOrderingMetrics:
    auc: float
    spearman: float
    kendall: float
    pairwise: float
    bins: int
    ece: float
    n: int

    def as_dict(self, prefix: str='energy_') -> Dict[str, float]:
        return {f'{prefix}auc': self.auc, f'{prefix}spearman': self.spearman, f'{prefix}kendall': self.kendall, f'{prefix}pairwise': self.pairwise, f'{prefix}bins': float(self.bins), f'{prefix}ece': self.ece, f'{prefix}n': float(self.n)}

def energy_ordering_metrics(energies: Sequence[float], labels: Sequence[int], groups: Optional[Sequence]=None, n_bins: int=5, calibration_split: float=0.5, calibration: Optional[Tuple[Sequence[float], Sequence[int]]]=None) -> EnergyOrderingMetrics:
    utilities = [-float(e) for e in energies]
    n = len(utilities)
    if n == 0:
        return EnergyOrderingMetrics(*(float('nan'),) * 4, 0, float('nan'), 0)
    cut = max(1, int(n * calibration_split)) if n >= 8 else n
    fit_utilities, test_utilities = (utilities[:cut], utilities[cut:])
    fit_labels, test_labels = (list(labels)[:cut], list(labels)[cut:])
    if not test_utilities:
        test_utilities, test_labels = (fit_utilities, fit_labels)
    if calibration is not None:
        fit_utilities = [-float(e) for e in calibration[0]]
        fit_labels = list(calibration[1])
        test_utilities, test_labels = (utilities, list(labels))
    return EnergyOrderingMetrics(auc=roc_auc(utilities, labels), spearman=spearman_correlation(utilities, labels), kendall=kendall_tau_b(utilities, labels), pairwise=pairwise_preference_accuracy(utilities, labels, groups), bins=monotone_bins(utilities, labels, n_bins=n_bins), ece=expected_calibration_error(fit_utilities, fit_labels, test_utilities, test_labels), n=n)

def aggregate_accuracy(per_suite: Dict[str, float]) -> float:
    values = [v for v in per_suite.values() if not math.isnan(v)]
    return float(np.mean(values)) if values else float('nan')

def sample_std(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2:
        return 0.0
    return float(array.std(ddof=1))

def bootstrap_std(values: Sequence[float], samples: int=1000, seed: int=0) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(array, size=array.size, replace=True).mean()) for _ in range(samples)]
    return float(np.std(means, ddof=1))

def paired_bootstrap_interval(a: Sequence[float], b: Sequence[float], samples: int=2000, seed: int=0) -> Tuple[float, float, float]:
    a_array = np.asarray(list(a), dtype=np.float64)
    b_array = np.asarray(list(b), dtype=np.float64)
    if a_array.size != b_array.size or a_array.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    difference = a_array - b_array
    rng = np.random.default_rng(seed)
    draws = [float(rng.choice(difference, size=difference.size, replace=True).mean()) for _ in range(samples)]
    return (float(difference.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
