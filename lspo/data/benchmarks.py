from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence
from ..env.verifiers import TaskItem
from .corpus import SyntheticCorpus

@dataclass
class SuiteSpec:
    name: str
    family: str
    repo_id: Optional[str]
    config: Optional[str]
    split: str
    gated: bool = False
    note: str = ''
MAIN_SUITES: Dict[str, SuiteSpec] = {'math500': SuiteSpec('math500', 'math', 'HuggingFaceH4/MATH-500', None, 'test'), 'aime25': SuiteSpec('aime25', 'math', 'MathArena/aime_2025', None, 'train'), 'gpqa': SuiteSpec('gpqa', 'science', 'Idavidrein/gpqa', 'gpqa_main', 'train', gated=True), 'livecodebench': SuiteSpec('livecodebench', 'code', 'livecodebench/code_generation_lite', None, 'test', gated=True, note="release_v5 window per the paper's evaluation date"), 'zebralogic': SuiteSpec('zebralogic', 'logic', 'WildEval/ZebraLogic', None, 'test'), 'mmlu_pro': SuiteSpec('mmlu_pro', 'science', 'TIGER-Lab/MMLU-Pro', None, 'test')}
HOLDOUT_SUITES: Dict[str, SuiteSpec] = {'gsm8k': SuiteSpec('gsm8k', 'math', 'openai/gsm8k', 'main', 'test'), 'arc_challenge': SuiteSpec('arc_challenge', 'science', 'allenai/ai2_arc', 'ARC-Challenge', 'test'), 'bbh': SuiteSpec('bbh', 'logic', 'lukaemon/bbh', None, 'test'), 'humaneval': SuiteSpec('humaneval', 'code', 'openai/openai_humaneval', None, 'test'), 'mbpp': SuiteSpec('mbpp', 'code', 'google-research-datasets/mbpp', 'sanitized', 'test')}
ALL_SUITES = {**MAIN_SUITES, **HOLDOUT_SUITES}

def _adapt_math500(row: dict, index: int) -> TaskItem:
    return TaskItem(item_id=f'math500-{index}', family='math500', question=str(row.get('problem', '')).strip(), answer=str(row.get('answer', '')).strip())

def _adapt_aime(row: dict, index: int) -> TaskItem:
    problem = row.get('problem') or row.get('question') or ''
    return TaskItem(item_id=f'aime25-{index}', family='aime25', question=str(problem).strip(), answer=str(row.get('answer', '')).strip())

def _adapt_gpqa(row: dict, index: int) -> TaskItem:
    correct = str(row.get('Correct Answer', '')).strip()
    wrong = [str(row.get(key, '')).strip() for key in ('Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3')]
    options = [correct] + [w for w in wrong if w]
    letters = 'ABCD'
    choices = {letters[i]: options[i] for i in range(len(options))}
    gold_letter = letters[0]
    question = f"{row.get('Question', '')}\n" + '\n'.join((f'{k}. {v}' for k, v in choices.items())) + '\nReply with the letter of the correct option as "Answer: X".'
    return TaskItem(item_id=f'gpqa-{index}', family='gpqa', question=question, answer=gold_letter, choices=choices, metadata={'domain': row.get('Subdomain', '')})

def _adapt_livecodebench(row: dict, index: int) -> TaskItem:
    tests = row.get('test') or ''
    if not tests and row.get('public_test_cases'):
        import json
        try:
            cases = json.loads(row['public_test_cases'])
            tests = '\n'.join((f"assert solve(*{json.dumps(c.get('input', []))}) == {c.get('output')!r}" for c in cases if isinstance(c, dict)))
        except Exception:
            tests = ''
    return TaskItem(item_id=f'lcb-{index}', family='livecodebench', question=str(row.get('question_content') or row.get('question') or '').strip(), reference_solution=str(row.get('solution') or ''), test_code=tests or None, metadata={'function_name': 'solve', 'has_tests': bool(tests)})

def _adapt_zebralogic(row: dict, index: int) -> TaskItem:
    return TaskItem(item_id=f'zebra-{index}', family='zebralogic', question=str(row.get('puzzle') or row.get('question') or '').strip(), answer=str(row.get('solution') or '').strip(), metadata={'size': row.get('size', '')})

def _adapt_mmlu_pro(row: dict, index: int) -> TaskItem:
    options = row.get('options') or []
    letters = 'ABCDEFGHIJ'
    choices = {letters[i]: str(o) for i, o in enumerate(options)}
    answer = str(row.get('answer', 'A')).strip().upper()[:1]
    question = f"{row.get('question', '')}\n" + '\n'.join((f'{k}. {v}' for k, v in choices.items())) + '\nReply with the letter of the correct option as "Answer: X".'
    return TaskItem(item_id=f'mmlu-pro-{index}', family='mmlu_pro', question=question, answer=answer, choices=choices, metadata={'category': row.get('category', '')})

def _adapt_gsm8k(row: dict, index: int) -> TaskItem:
    from ..utils.text import extract_final_answer
    raw = str(row.get('answer', ''))
    return TaskItem(item_id=f'gsm8k-{index}', family='gsm8k', question=str(row.get('question', '')).strip(), answer=extract_final_answer(raw, task_family='math'))

def _adapt_arc(row: dict, index: int) -> TaskItem:
    labels = row.get('choices', {}).get('label', []) or []
    texts = row.get('choices', {}).get('text', []) or []
    choices = {str(l): str(t) for l, t in zip(labels, texts)}
    return TaskItem(item_id=f'arc-{index}', family='arc_challenge', question=str(row.get('question', '')).strip() + '\n' + '\n'.join((f'{k}. {v}' for k, v in choices.items())), answer=str(row.get('answerKey', '')).strip().upper(), choices=choices)

def _adapt_bbh(row: dict, index: int) -> TaskItem:
    return TaskItem(item_id=f'bbh-{index}', family='bbh', question=str(row.get('input', '')).strip(), answer=str(row.get('target', '')).strip(), metadata={'task': row.get('task', '')})

def _adapt_humaneval(row: dict, index: int) -> TaskItem:
    name = str(row.get('entry_point', 'solution'))
    return TaskItem(item_id=f'humaneval-{index}', family='humaneval', question=str(row.get('prompt', '')).strip() + '\nComplete the function and reply with the full definition in one ```python block.', test_code=str(row.get('test', '')), reference_solution=str(row.get('canonical_solution', '')), metadata={'function_name': name})

def _adapt_mbpp(row: dict, index: int) -> TaskItem:
    name = str(row.get('func_name') or row.get('entry_point') or 'solution')
    tests = '\n'.join((line for line in str(row.get('test_list', '')).split(';') if line.strip()))
    return TaskItem(item_id=f'mbpp-{index}', family='mbpp', question=str(row.get('text') or row.get('prompt') or '').strip() + '\nReply with the complete function in one ```python block.', test_code=tests or None, metadata={'function_name': name})
_ADAPTERS: Dict[str, Callable[[dict, int], TaskItem]] = {'math500': _adapt_math500, 'aime25': _adapt_aime, 'gpqa': _adapt_gpqa, 'livecodebench': _adapt_livecodebench, 'zebralogic': _adapt_zebralogic, 'mmlu_pro': _adapt_mmlu_pro, 'gsm8k': _adapt_gsm8k, 'arc_challenge': _adapt_arc, 'bbh': _adapt_bbh, 'humaneval': _adapt_humaneval, 'mbpp': _adapt_mbpp}

def load_suite(name: str, limit: Optional[int]=None, seed: int=42, allow_download: bool=True) -> List[TaskItem]:
    spec = ALL_SUITES.get(name)
    if spec is None:
        raise KeyError(f'unknown suite {name!r}; known: {sorted(ALL_SUITES)}')
    if allow_download and spec.repo_id:
        try:
            from datasets import load_dataset
            if name == 'bbh':
                from datasets import get_dataset_config_names, concatenate_datasets
                subsets = sorted(get_dataset_config_names(spec.repo_id))
                parts = []
                for subset in subsets:
                    part = load_dataset(spec.repo_id, subset, split=spec.split)
                    if 'task' not in part.column_names:
                        part = part.add_column('task', [subset] * len(part))
                    parts.append(part)
                dataset = concatenate_datasets(parts)
            else:
                dataset = load_dataset(spec.repo_id, spec.config, split=spec.split)
            adapter = _ADAPTERS[name]
            items: List[TaskItem] = []
            for index, row in enumerate(dataset):
                if limit is not None and len(items) >= limit:
                    break
                item = adapter(dict(row), index)
                if item.question:
                    items.append(item)
            if items:
                return items
        except Exception as exc:
            raise RuntimeError(f'Could not load benchmark {name}: {exc}') from exc
        raise ValueError(f'Benchmark {name} contained no usable examples')
    return _synthetic_suite(spec, limit or 32, seed)

def _synthetic_suite(spec: SuiteSpec, count: int, seed: int) -> List[TaskItem]:
    corpus = SyntheticCorpus(seed=seed + hash(spec.name) % 10000)
    canonical = {'math': 'math', 'science': 'science', 'logic': 'logic', 'code': 'code'}[spec.family]
    items = corpus.build(count, canonical, offset=50000)
    for item in items:
        item.family = spec.name
        item.metadata = dict(item.metadata or {})
        item.metadata['synthetic_fallback'] = True
        item.metadata['suite'] = spec.name
    return items

def load_main_suites(config, limit: Optional[int]=None) -> Dict[str, List[TaskItem]]:
    return {name: load_suite(name, limit=limit, seed=config.seed, allow_download=not config.data.offline_synthetic) for name in config.data.main_suites}

def load_holdout_suites(config, limit: Optional[int]=None) -> Dict[str, List[TaskItem]]:
    return {name: load_suite(name, limit=limit, seed=config.seed, allow_download=not config.data.offline_synthetic) for name in config.data.holdout_suites}

def assert_disjoint(train_items: Sequence[TaskItem], eval_items: Sequence[TaskItem]) -> None:
    from ..utils.text import normalize_whitespace

    def is_synthetic(item: TaskItem) -> bool:
        metadata = item.metadata or {}
        return bool(metadata.get('synthetic_fallback') or metadata.get('synthetic'))
    train_questions = {normalize_whitespace(item.question) for item in train_items if not is_synthetic(item)}
    overlap = [item.item_id for item in eval_items if not is_synthetic(item) and normalize_whitespace(item.question) in train_questions]
    if overlap:
        raise ValueError(f"{len(overlap)} evaluation items also appear in the training mixture (first: {overlap[:3]}).  See Appendix E.1 'Data Separation'.")
