from __future__ import annotations
import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from ..utils.text import answers_match, extract_final_answer, normalize_answer
@dataclass
class TaskItem:
    item_id: str
    family: str
    question: str
    answer: str = ""
    test_code: Optional[str] = None
    test_inputs: Optional[List[str]] = None
    expected_outputs: Optional[List[str]] = None
    choices: Optional[Dict[str, str]] = None
    reference_solution: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
@dataclass
class VerificationResult:
    verified: bool
    score: float = 0.0
    reason: str = ""
    executed: bool = True
    extracted: str = ""
    def __bool__(self) -> bool:
        return bool(self.verified)
class Checker:
    name = "base"
    def __call__(self, item: TaskItem, generation: str) -> VerificationResult:
        raise NotImplementedError
class ExactMatchChecker(Checker):
    name = "exact_match"
    def __call__(self, item: TaskItem, generation: str) -> VerificationResult:
        extracted = extract_final_answer(generation, task_family="math")
        ok = answers_match(extracted, item.answer, task_family="math")
        return VerificationResult(
            verified=ok, score=1.0 if ok else 0.0, reason="exact_match", extracted=extracted
        )
class MultipleChoiceChecker(Checker):
    name = "mcq"
    _LETTER = re.compile(r"\b([A-J])\b")
    def __call__(self, item: TaskItem, generation: str) -> VerificationResult:
        gold = (item.answer or "").strip().upper()
        tail = generation[-400:]
        marked = re.findall(r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-J])\)?", tail, re.I)
        letters = marked or self._LETTER.findall(tail)
        prediction = letters[-1].upper() if letters else ""
        ok = prediction == gold
        return VerificationResult(
            verified=ok, score=1.0 if ok else 0.0, reason="mcq", extracted=prediction
        )
class CodeChecker(Checker):
    name = "code"
    def __init__(self, allow_exec: bool = False, timeout_s: float = 6.0):
        self.allow_exec = allow_exec
        self.timeout_s = timeout_s
    def __call__(self, item: TaskItem, generation: str) -> VerificationResult:
        code = extract_final_answer(generation, task_family="code")
        expected_name = (item.metadata or {}).get("function_name")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return VerificationResult(
                verified=False, reason=f"syntax_error: {exc.msg}", executed=False
            )
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if expected_name and expected_name not in defined:
            return VerificationResult(
                verified=False,
                reason=f"missing_function:{expected_name}",
                executed=False,
            )
        if not defined:
            return VerificationResult(verified=False, reason="no_function", executed=False)
        if not self.allow_exec:
            return VerificationResult(
                verified=True, score=0.5, reason="static_only", executed=False
            )
        return self._execute(item, code)
    def _execute(self, item: TaskItem, code: str) -> VerificationResult:
        tests = (item.test_code or "").rstrip()
        if not tests:
            return VerificationResult(
                verified=False, reason="no_tests", executed=False
            )
        def indent(block: str, prefix: str = "    ") -> str:
            return "\n".join(
                prefix + line if line.strip() else line for line in block.splitlines()
            )
        if re.search(r"^\s*def\s+check\s*\(", tests, re.MULTILINE):
            runner = (
                f"    check({item.metadata.get('function_name', 'solution')})\n"
                "    return True\n"
            )
            body = indent(tests) + "\n" + runner
        else:
            body = indent(tests) + "\n    return True\n"
        harness = (
            "import sys\n"
            f"{code}\n\n"
            "def _run():\n"
            f"{body}"
            "try:\n"
            "    _run()\n"
            "    print('LSPO_TEST_OK')\n"
            "except AssertionError:\n"
            "    print('LSPO_TEST_FAIL')\n"
            "except Exception as exc:\n"
            "    print('LSPO_TEST_ERROR', type(exc).__name__, exc)\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "candidate.py"
            script.write_text(harness, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    verified=False, reason="timeout", executed=True
                )
        stdout = completed.stdout
        ok = "LSPO_TEST_OK" in stdout
        return VerificationResult(
            verified=ok, score=1.0 if ok else 0.0, reason="executed", executed=True
        )
class LogicChecker(Checker):
    name = "logic"
    _ASSIGN = re.compile(r"^\s*([A-Za-z0-9_ ]+?)\s*(?::|=)\s*([A-Za-z0-9_ ]+?)\s*$", re.M)
    def __call__(self, item: TaskItem, generation: str) -> VerificationResult:
        constraints = (item.metadata or {}).get("constraints") or []
        assignment = self._parse(generation)
        if not assignment:
            return VerificationResult(verified=False, reason="no_assignment", extracted="")
        for constraint in constraints:
            if not evaluate_constraint(constraint, assignment):
                return VerificationResult(
                    verified=False, reason="constraint_violated", extracted=str(assignment)
                )
        solution = (item.metadata or {}).get("solution") or {}
        if solution:
            consistent = all(
                normalize_answer(str(assignment.get(key, ""))) == normalize_answer(value)
                for key, value in solution.items()
                if key in assignment
            )
            if not consistent:
                return VerificationResult(
                    verified=False, reason="wrong_solution", extracted=str(assignment)
                )
        return VerificationResult(
            verified=True, score=1.0, reason="logic", extracted=str(assignment)
        )
    @classmethod
    def _parse(cls, generation: str) -> Dict[str, str]:
        assignment: Dict[str, str] = {}
        for match in cls._ASSIGN.finditer(generation):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            if key and value:
                assignment[key] = value
        return assignment
def evaluate_constraint(constraint: Dict, assignment: Dict[str, str]) -> bool:
    op = constraint.get("op", "eq")
    a = str(constraint.get("a", "")).lower()
    if op == "neq":
        b = constraint.get("b")
        if b is not None:
            b = str(b).lower()
            if a not in assignment or b not in assignment:
                return True
            return normalize_answer(assignment[a]) != normalize_answer(assignment[b])
        value = constraint.get("value")
        if a not in assignment:
            return True
        return normalize_answer(assignment[a]) != normalize_answer(str(value))
    if op == "eq":
        if "b" in constraint:
            b = str(constraint["b"]).lower()
            if a not in assignment or b not in assignment:
                return True
            return normalize_answer(assignment[a]) == normalize_answer(assignment[b])
        if a not in assignment:
            return True
        return normalize_answer(assignment[a]) == normalize_answer(str(constraint.get("value", "")))
    if op in {"lt", "gt"}:
        b = str(constraint.get("b", "")).lower()
        if a not in assignment or b not in assignment:
            return True
        try:
            left = float(normalize_answer(assignment[a]))
            right = float(normalize_answer(assignment[b]))
        except ValueError:
            return True
        return left < right if op == "lt" else left > right
    raise ValueError(f"unknown constraint op {op!r}")
class CompositeChecker(Checker):
    name = "composite"
    def __init__(self, allow_code_exec: bool = False, timeout_s: float = 6.0):
        self.by_family = {
            "math": ExactMatchChecker(),
            "science": MultipleChoiceChecker(),
            "logic": LogicChecker(),
            "code": CodeChecker(allow_exec=allow_code_exec, timeout_s=timeout_s),
        }
        self.aliases = {
            "gsm8k": "math",
            "math500": "math",
            "aime25": "math",
            "arc_challenge": "science",
            "gpqa": "science",
            "mmlu_pro": "science",
            "bbh": "logic",
            "zebralogic": "logic",
            "humaneval": "code",
            "mbpp": "code",
            "livecodebench": "code",
        }
    def checker_for(self, family: str) -> Checker:
        return self.by_family.get(self.aliases.get(family, family), ExactMatchChecker())
    def __call__(self, item: TaskItem, generation: str) -> VerificationResult:
        return self.checker_for(item.family)(item, generation)
def build_checker(config) -> CompositeChecker:
    allow_exec = bool(getattr(config.data, "allow_code_exec", False))
    return CompositeChecker(allow_code_exec=allow_exec)
@dataclass
class StateRecord:
    prompt: str
    question: str
    text: str
    task_id: int
    task_family: str
    verified: bool
    step: int
    edit_cost: float
    energy: float = 0.0
    source: str = "rollout"
def build_preference_pairs(
    records: Sequence[StateRecord],
    margin_rule: str = "verified_then_cost",
) -> List[tuple]:
    by_prompt: Dict[str, List[StateRecord]] = {}
    for record in records:
        by_prompt.setdefault(record.prompt, []).append(record)
    pairs: List[tuple] = []
    def rank_key(record: StateRecord):
        return (
            0 if record.verified else 1,
            record.edit_cost if margin_rule == "verified_then_cost" else 0.0,
            record.step,
        )
    for prompt, group in by_prompt.items():
        winners = sorted(group, key=rank_key)
        verified = [r for r in winners if r.verified]
        unverified = [r for r in winners if not r.verified]
        for winner in verified:
            for loser in unverified:
                pairs.append((winner, loser))
        if not verified:
            ordered = sorted(group, key=lambda r: (r.edit_cost, r.step))
            for i, winner in enumerate(ordered[: max(1, len(ordered) // 2)]):
                for loser in ordered[i + 1:]:
                    if (loser.edit_cost, loser.step) > (winner.edit_cost, winner.step):
                        pairs.append((winner, loser))
    return pairs
