from __future__ import annotations
import hashlib
import random
import re
from typing import Dict, List, Optional, Sequence
from ..env.verifiers import TaskItem
FAMILY_ORDER = ("math", "code", "science", "logic")
class SyntheticCorpus:
    def __init__(self, seed: int = 42, with_answers: bool = True):
        self.rng = random.Random(seed)
        self.with_answers = with_answers
    def math_item(self, index: int) -> TaskItem:
        style = index % 3
        if style == 0:
            a, b = self.rng.randint(11, 89), self.rng.randint(11, 89)
            question = f"Compute {a} + {b}."
            answer = str(a + b)
        elif style == 1:
            a, b = self.rng.randint(3, 19), self.rng.randint(3, 12)
            question = f"Compute {a} * {b}."
            answer = str(a * b)
        else:
            a, b, c = self.rng.randint(20, 60), self.rng.randint(2, 15), self.rng.randint(2, 9)
            question = f"Compute {a} - {b} * {c}."
            answer = str(a - b * c)
        return TaskItem(
            item_id=f"synth-math-{index}",
            family="math",
            question=question,
            answer=answer,
            metadata={"difficulty": "single_step"},
        )
    def code_item(self, index: int) -> TaskItem:
        style = index % 3
        if style == 0:
            name, signature, body = "add", "a, b", "return a + b"
            tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
            description = "returns the sum of two numbers"
        elif style == 1:
            name, signature, body = "square", "x", "return x * x"
            tests = "assert square(4) == 16\nassert square(0) == 0"
            description = "returns its argument squared"
        else:
            name, signature, body = "is_even", "n", "return n % 2 == 0"
            tests = "assert is_even(4)\nassert not is_even(7)"
            description = "returns True if and only if n is even"
        question = (
            f"Write a Python function `{name}({signature})` that {description}. "
            "Reply with the complete function inside a single ```python fenced block."
        )
        return TaskItem(
            item_id=f"synth-code-{index}",
            family="code",
            question=question,
            reference_solution=f"def {name}({signature}):\n    {body}",
            test_code=tests,
            metadata={"function_name": name},
        )
    _SCIENCE_TEMPLATES = (
        ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Earth", "Mars"], "Mercury"),
        ("What gas do plants take in for photosynthesis?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "Carbon dioxide"),
        ("What is the chemical symbol for water?", ["H2O", "CO2", "NaCl", "O2"], "H2O"),
        ("How many bones are in the adult human body?", ["206", "186", "226", "246"], "206"),
    )
    def science_item(self, index: int) -> TaskItem:
        stem, options, correct = self._SCIENCE_TEMPLATES[index % len(self._SCIENCE_TEMPLATES)]
        permutation = list(range(len(options)))
        self.rng.shuffle(permutation)
        shuffled = [options[i] for i in permutation]
        letters = "ABCD"
        choices = {letters[i]: shuffled[i] for i in range(len(shuffled))}
        gold_letter = letters[shuffled.index(correct)]
        question = stem + "\n" + "\n".join(
            f"{letter}. {text}" for letter, text in choices.items()
        )
        question += '\nReply with the letter of the correct option as "Answer: X".'
        return TaskItem(
            item_id=f"synth-science-{index}",
            family="science",
            question=question,
            answer=gold_letter,
            choices=choices,
        )
    def logic_item(self, index: int) -> TaskItem:
        names = ["alice", "bob", "carol", "dave"]
        values = ["red", "blue", "green", "yellow"]
        self.rng.shuffle(values)
        solution = dict(zip(names, values))
        constraints = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                constraints.append({"op": "neq", "a": a, "b": b})
        constraints.append({"op": "eq", "a": "alice", "value": solution["alice"]})
        constraints.append({"op": "neq", "a": "bob", "value": solution["carol"]})
        clue_lines = [
            f"1. Alice's colour is {solution['alice'].capitalize()}.",
            f"2. Bob's colour is not {solution['carol'].capitalize()}.",
            "3. All four people have different colours.",
        ]
        question = (
            "Assign each person a colour from {red, blue, green, yellow}.\n"
            + "\n".join(clue_lines)
            + "\nWrite one line per person in the form \"Name: colour\"."
        )
        return TaskItem(
            item_id=f"synth-logic-{index}",
            family="logic",
            question=question,
            answer="",
            metadata={"constraints": constraints, "solution": solution, "names": names},
        )
    def build(self, count: int, family: str, offset: int = 0) -> List[TaskItem]:
        builder = {
            "math": self.math_item,
            "code": self.code_item,
            "science": self.science_item,
            "logic": self.logic_item,
        }[family]
        return [builder(offset + i) for i in range(count)]
    def mixture(self, sizes: Dict[str, int], offset: int = 0) -> List[TaskItem]:
        items: List[TaskItem] = []
        for family in FAMILY_ORDER:
            count = int(sizes.get(family, 0))
            if count:
                items.extend(self.build(count, family, offset=offset))
        self.rng.shuffle(items)
        return items
    @staticmethod
    def tag_synthetic(items: Sequence[TaskItem]) -> List[TaskItem]:
        for item in items:
            item.metadata = dict(item.metadata or {})
            item.metadata.setdefault("synthetic", True)
        return list(items)
_FIELD_MAPS = {
    "open-r1/OpenR1-Math-220k": {"question": ("problem", "question"), "answer": ("answer", "solution")},
    "BAAI/TACO": {"question": ("question", "description"), "answer": ("solutions", "solution")},
    "allenai/sciq": {
        "question": ("question",),
        "choices": ("choices",),
        "answer": ("correct_answer", "answerKey"),
    },
    "longface/logicLM": {"question": ("question", "prompt"), "answer": ("answer",)},
}
_ANSWER_RE = re.compile(r"\\boxed\{([^{}]*)\}")
def _first_present(row: dict, keys: Sequence[str]):
    for key in keys:
        if key in row and row[key] not in (None, "", []):
            return row[key]
    return None
def _clean_math_answer(raw: str) -> str:
    if not isinstance(raw, str):
        return str(raw)
    boxed = _ANSWER_RE.findall(raw)
    return boxed[-1].strip() if boxed else raw.strip().split("\n")[-1].strip()
def load_hf_family(
    source: str,
    family: str,
    limit: int,
    seed: int = 42,
    split: str = "train",
) -> List[TaskItem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "the 'datasets' package is required for the real Table C2 corpora; "
            "set data.offline_synthetic=true to use the built-in generator."
        ) from exc
    fields = _FIELD_MAPS.get(source, {})
    dataset = load_dataset(source, split=split)
    items: List[TaskItem] = []
    for index, row in enumerate(dataset):
        if len(items) >= limit:
            break
        row = dict(row)
        question = _first_present(row, fields.get("question", ("question", "problem")))
        if question is None:
            continue
        answer = _first_present(
            row, fields.get("answer", ("answer", "solution", "target"))
        )
        item = TaskItem(
            item_id=f"{family}-{index}",
            family=family,
            question=str(question),
            answer=_clean_math_answer(answer) if family == "math" and answer else (
                str(answer) if answer is not None else ""
            ),
            metadata={"source": source, "row": index},
        )
        if family == "science":
            choices = _first_present(row, fields.get("choices", ("choices",)))
            if isinstance(choices, dict):
                item.choices = {str(k): str(v) for k, v in choices.items()}
                item.answer = str(answer)
            else:
                continue
        if family == "code":
            item.metadata["has_tests"] = False
        items.append(item)
    return items
def _stable_seed(seed: int, family: str) -> int:
    digest = hashlib.sha1(f"{seed}:{family}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
def build_training_items(config, limit: Optional[int] = None) -> List[TaskItem]:
    data_cfg = config.data
    if data_cfg.offline_synthetic:
        corpus = SyntheticCorpus(seed=config.seed)
        per_family = max(1, data_cfg.offline_train_size // len(FAMILY_ORDER))
        sizes = {family: per_family for family in FAMILY_ORDER}
        items = corpus.mixture(sizes)
        SyntheticCorpus.tag_synthetic(items)
    else:
        items = []
        for family in FAMILY_ORDER:
            source = data_cfg.sources.get(family)
            count = int(data_cfg.mixture.get(family, 0))
            if not source or count <= 0:
                continue
            items.extend(
                load_hf_family(source, family, count, seed=_stable_seed(config.seed, family))
            )
    if limit is not None:
        items = items[:limit]
    return items
def build_heldout_items(config, per_family: int = 16) -> List[TaskItem]:
    corpus = SyntheticCorpus(seed=config.seed + 99991)
    items: List[TaskItem] = []
    for family in FAMILY_ORDER:
        items.extend(corpus.build(per_family, family, offset=100_000))
    return items
def family_index(family: str, families: Sequence[str] = FAMILY_ORDER) -> int:
    aliases = {
        "gsm8k": "math", "math500": "math", "aime25": "math",
        "arc_challenge": "science", "gpqa": "science", "mmlu_pro": "science",
        "bbh": "logic", "zebralogic": "logic",
        "humaneval": "code", "mbpp": "code", "livecodebench": "code",
    }
    canonical = aliases.get(family, family)
    return families.index(canonical) if canonical in families else 0
