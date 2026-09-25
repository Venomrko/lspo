from __future__ import annotations
import difflib
from dataclasses import dataclass
from typing import List, Tuple
from ..models.backbone import REFLECT_TEMPLATE, REVISE_TEMPLATE, SWITCH_TEMPLATE
from ..utils.text import tokenize
@dataclass
class Proposal:
    text: str
    raw_text: str
    action_name: str
    patched_spans: int = 0
    truncated: bool = False
def _changed_regions(old: str, new: str) -> List[Tuple[int, int]]:
    old_tokens, new_tokens = tokenize(old), tokenize(new)
    matcher = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    regions: List[Tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        regions.append((j1, j2))
    return regions
def apply_bounded_patch(
    old: str,
    new: str,
    max_patches: int = 1,
    max_span_chars: int = 900,
) -> Tuple[str, int]:
    if old == new:
        return old, 0
    regions = _changed_regions(old, new)
    if not regions:
        return old, 0
    old_tokens, new_tokens = tokenize(old), tokenize(new)
    def region_size(region: Tuple[int, int]) -> int:
        start, end = region
        return sum(len(token) for token in new_tokens[start:end])
    ranked = sorted(regions, key=region_size, reverse=True)[: max(1, max_patches)]
    ranked = sorted(ranked, key=lambda r: r[0])
    kept = set()
    for start, end in ranked:
        kept.update(range(start, end))
    out: List[str] = []
    for index, token in enumerate(new_tokens):
        if index in kept:
            out.append(token)
    patched = "".join(out).strip()
    if not patched:
        return old, 0
    if len(patched) > max_span_chars:
        clipped = patched[:max_span_chars]
        cut = max(clipped.rfind("\n"), clipped.rfind(" "))
        patched = clipped[: cut if cut > 0 else max_span_chars].rstrip()
    if patched == old.strip():
        return old, 0
    return patched, len(ranked)
class ProposalKernel:
    def __init__(self, generator, rollout_config):
        self.generator = generator
        self.config = rollout_config
    def revise(self, question: str, current: str) -> Proposal:
        prompt = REVISE_TEMPLATE.format(question=question, answer=current)
        raw = self.generator.generate(
            prompt,
            max_new_tokens=self.config.proposal_max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        patched, n_spans = apply_bounded_patch(
            current,
            raw.strip(),
            max_patches=self.config.revise_max_patches,
            max_span_chars=self.config.revise_max_span_chars,
        )
        return Proposal(
            text=patched,
            raw_text=raw.strip(),
            action_name="revise",
            patched_spans=n_spans,
            truncated=len(raw) >= self.config.proposal_max_new_tokens,
        )
    def switch(self, question: str, current: str) -> Proposal:
        prompt = SWITCH_TEMPLATE.format(question=question)
        candidates: List[str] = []
        for _ in range(max(1, self.config.switch_candidates)):
            text = self.generator.generate(
                prompt,
                max_new_tokens=self.config.proposal_max_new_tokens,
                temperature=self.config.switch_temperature,
                top_p=self.config.top_p,
            ).strip()
            candidates.append(text)
        chosen = current
        for candidate in candidates:
            if candidate and candidate.strip() != current.strip():
                chosen = candidate
                break
        else:
            chosen = candidates[-1] if candidates else current
        return Proposal(
            text=chosen,
            raw_text=chosen,
            action_name="switch",
            patched_spans=0,
        )
    def reflect(self, question: str, current: str) -> Proposal:
        prompt = REFLECT_TEMPLATE.format(question=question, answer=current)
        text = self.generator.generate(
            prompt,
            max_new_tokens=self.config.proposal_max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        ).strip()
        return Proposal(text=text, raw_text=text, action_name="reflect")
def execute_transition(
    kernel: ProposalKernel,
    action_name: str,
    question: str,
    current: str,
) -> Proposal:
    if action_name == "revise":
        return kernel.revise(question, current)
    if action_name == "switch":
        return kernel.switch(question, current)
    raise ValueError(f"no proposal for action {action_name!r}")
