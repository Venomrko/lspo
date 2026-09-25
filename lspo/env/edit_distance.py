from __future__ import annotations
import difflib
from typing import List, Sequence
from ..utils.text import normalize_for_metric, segment_reasoning_spans, tokenize
_MAX_DP_TOKENS = 4000
def _levenshtein(a: Sequence[str], b: Sequence[str], cap: int = _MAX_DP_TOKENS) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > cap or len(b) > cap:
        return abs(len(a) - len(b)) + sum(
            1 for x, y in zip(a, b) if x != y
        )
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (token_a != token_b)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]
def _changed_span_distance(tokens_a: List[str], tokens_b: List[str]) -> tuple[float, int]:
    matcher = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
    distance = 0.0
    denominator = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        span_a, span_b = tokens_a[i1:i2], tokens_b[j1:j2]
        distance += _levenshtein(span_a, span_b)
        denominator += max(len(span_a), len(span_b))
    return distance, denominator
def span_edit_distance(
    before: str,
    after: str,
    cap: float = 1.0,
    normalize_math: bool = True,
    normalize_code: bool = True,
    use_spans: bool = True,
    normalization: str = "answer",
) -> float:
    if before == after:
        return 0.0
    norm_a = normalize_for_metric(before, math=normalize_math, code=normalize_code)
    norm_b = normalize_for_metric(after, math=normalize_math, code=normalize_code)
    if norm_a == norm_b:
        return 0.0
    tokens_a = tokenize(norm_a)
    tokens_b = tokenize(norm_b)
    answer_denominator = max(len(tokens_a), len(tokens_b))
    distance, changed_denominator, widest_span = 0.0, 0, 0
    if use_spans:
        spans_a = segment_reasoning_spans(norm_a)
        spans_b = segment_reasoning_spans(norm_b)
        if len(spans_a) == len(spans_b):
            for span_a, span_b in zip(spans_a, spans_b):
                if span_a == span_b:
                    continue
                span_tokens_a, span_tokens_b = tokenize(span_a), tokenize(span_b)
                span_distance, span_denominator = _changed_span_distance(
                    span_tokens_a, span_tokens_b
                )
                distance += span_distance
                changed_denominator += span_denominator
                widest_span = max(widest_span, span_denominator)
        else:
            distance, changed_denominator = _changed_span_distance(tokens_a, tokens_b)
            widest_span = changed_denominator
    else:
        distance, changed_denominator = _changed_span_distance(tokens_a, tokens_b)
        widest_span = changed_denominator
    if normalization == "answer":
        denominator = answer_denominator
    elif normalization == "changed_span":
        denominator = changed_denominator
    elif normalization == "patch_span":
        denominator = widest_span
    else:
        raise ValueError(f"unknown edit normalization {normalization!r}")
    if denominator == 0:
        return 0.0
    return min(cap, distance / denominator)
def plain_edit_distance(before: str, after: str, cap: float = 1.0) -> float:
    if before == after:
        return 0.0
    tokens_a = tokenize(normalize_for_metric(before))
    tokens_b = tokenize(normalize_for_metric(after))
    denominator = max(len(tokens_a), len(tokens_b))
    if denominator == 0:
        return 0.0
    return min(cap, _levenshtein(tokens_a, tokens_b) / denominator)
def edit_distance(before: str, after: str, config=None) -> float:
    if config is None:
        return span_edit_distance(before, after)
    mode = getattr(config, "edit_distance", "span_levenshtein")
    cap = getattr(config, "edit_cap", 1.0)
    normalization = getattr(config, "edit_normalization", "answer")
    if mode == "plain_levenshtein":
        return plain_edit_distance(before, after, cap=cap)
    return span_edit_distance(
        before,
        after,
        cap=cap,
        normalize_math=getattr(config, "normalize_math_tokens", True),
        normalize_code=getattr(config, "normalize_code_tokens", True),
        normalization=normalization,
    )
