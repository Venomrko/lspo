from __future__ import annotations
import re
import unicodedata
from typing import List, Sequence
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", flags=re.MULTILINE)
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.lstrip(" \t")
        indent = line[: len(line) - len(stripped)]
        lines.append(indent + re.sub(r"[ \t]{2,}", " ", stripped))
    text = "\n".join(lines)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()
_MATH_REWRITES = (
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\bigl?\s*", ""),
    (r"\\bigr?\s*", ""),
    (r"\\Bigl?\s*", ""),
    (r"\\Bigr?\s*", ""),
    (r"\\displaystyle", ""),
    (r"\\limits", ""),
    (r"\\!", ""),
    (r"\\,", ""),
    (r"\\;", ""),
    (r"\\:", ""),
    (r"\\quad", " "),
    (r"\\qquad", " "),
    (r"\\dfrac", "\\frac"),
    (r"\\tfrac", "\\frac"),
    (r"\\cdot", "*"),
    (r"\\times", "*"),
    (r"\\div", "/"),
    (r"\\ast", "*"),
    (r"\\leq?", "<="),
    (r"\\geq?", ">="),
    (r"\\neq", "!="),
    (r"\\approx", "~="),
    (r"\\to", "->"),
    (r"\\rightarrow", "->"),
    (r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1"),
    (r"\\%", "%"),
    (r"\\\$", "$"),
    (r"\\ ", " "),
)
def normalize_math(text: str) -> str:
    for pattern, replacement in _MATH_REWRITES:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\{([A-Za-z0-9])\}", r"\1", text)
    text = re.sub(r"\s*([+\-*/=<>])\s*", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text
def normalize_code(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        expanded = line.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        indent = 4 * round(indent / 4)
        out.append(" " * indent + expanded.lstrip(" "))
    text = "\n".join(out)
    return _MULTI_BLANK.sub("\n\n", text)
def normalize_for_metric(text: str, math: bool = True, code: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = normalize_whitespace(text)
    if math:
        text = normalize_math(text)
    if code:
        text = normalize_code(text)
    return text
_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*"
    r"|\d+(?:\.\d+)?"
    r"|\\[A-Za-z]+"
    r"|==|!=|<=|>=|->|\*\*|//|<<|>>|:=|\+\+|--|\+=|-=|\*=|/=|%=|&&|\|\|"
    r"|[\s\S]",
    re.DOTALL,
)
def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)
_STEP_MARKER = re.compile(
    r"^\s*(?:#{1,6}\s|step\s*\d+|\d+[.)]\s|[-*]\s|\*\*step|```)", re.IGNORECASE
)
def segment_reasoning_spans(text: str, max_spans: int = 64) -> List[str]:
    lines = text.split("\n")
    spans: List[str] = []
    buffer: List[str] = []
    in_fence = False
    def flush() -> None:
        if buffer:
            chunk = "\n".join(buffer).strip()
            if chunk:
                spans.append(chunk)
            buffer.clear()
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            if not in_fence:
                flush()
            continue
        if in_fence:
            buffer.append(line)
            continue
        if not line.strip():
            flush()
            continue
        if _STEP_MARKER.match(line) and buffer:
            flush()
        buffer.append(line)
    flush()
    if len(spans) > max_spans:
        spans = spans[:max_spans]
    return spans or [text]
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_PATTERNS = (
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"\\boxed\{([^{}]*)\}"),
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
_CHOICE_RE = re.compile(r"(?:answer\s*(?:is|:)\s*)?\(?([A-J])\)?(?:\b|$)", re.IGNORECASE)
def extract_final_answer(text: str, task_family: str = "math") -> str:
    if task_family == "code":
        blocks = _FENCE_RE.findall(text)
        return blocks[-1].strip() if blocks else text.strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    for pattern in _ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip()
    if task_family in {"math", "science"}:
        numbers = _NUMBER_RE.findall(text)
        if numbers:
            return numbers[-1]
    if task_family == "logic":
        choices = _CHOICE_RE.findall(text)
        if choices:
            return choices[-1].upper()
    return text.strip()
_PUNCT_STRIP = re.compile(r"[\s$,\\.;:]+$")
_UNIT_STRIP = re.compile(r"\s*(?:units?|cm|m|km|kg|g|seconds?|s|hours?|h|degrees?|°)\s*$",
                         re.IGNORECASE)
def normalize_answer(answer: str) -> str:
    answer = unicodedata.normalize("NFKC", answer).strip().lower()
    answer = _PUNCT_STRIP.sub("", answer)
    answer = _UNIT_STRIP.sub("", answer)
    answer = answer.replace(",", "").replace("$", "").replace("\\%", "%")
    answer = re.sub(r"\s+", " ", answer)
    frac = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", answer)
    if frac:
        answer = f"{frac.group(1)}/{frac.group(2)}"
    answer = re.sub(r"^[a-z]\s*=\s*", "", answer)
    return answer.strip(" .")
def answers_match(pred: str, gold: str, task_family: str = "math") -> bool:
    pred_n, gold_n = normalize_answer(pred), normalize_answer(gold)
    if pred_n == gold_n:
        return True
    def _as_float(value: str):
        try:
            if "/" in value and re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", value):
                num, den = value.split("/")
                return float(num) / float(den)
            return float(value)
        except (ValueError, ZeroDivisionError):
            return None
    a, b = _as_float(pred_n), _as_float(gold_n)
    if a is not None and b is not None:
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    if task_family == "code":
        return pred_n == gold_n
    return False
def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
def common_prefix_len(a: Sequence, b: Sequence) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index
