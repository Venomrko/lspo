from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ModelConfig:
    backbone: str = 'Qwen/Qwen3.5-27B'
    tokenizer: Optional[str] = None
    dtype: str = 'auto'
    device: str = 'auto'
    trust_remote_code: bool = False
    attn_implementation: str = 'auto'
    amp: bool = True
    compile_model: bool = False
    tiny_d_model: int = 128
    tiny_n_layer: int = 2
    tiny_n_head: int = 4
    tiny_ctx: int = 512
    lift_hidden: int = 1024
    lift_layers: int = 2
    lift_activation: str = 'gelu'
    lift_input_layernorm: bool = True
    d_z: int = 8
    energy_hidden: int = 1024
    energy_layers: int = 3
    energy_activation: str = 'silu'
    energy_margin: float = 0.2
    transition_hidden: int = 1024
    transition_blocks: int = 2
    transition_activation: str = 'silu'
    action_embed_dim: int = 128
    value_hidden: int = 512
    value_layers: int = 2
    value_activation: str = 'silu'
    d_task: int = 64
    task_families: Tuple[str, ...] = ('math', 'code', 'science', 'logic')
    text_pooling: str = 'mean_last'
    policy_state_source: str = 'shared'
    policy_hidden: int = 256
    scoring_encoder_init: str = 'checkpoint'

@dataclass
class ObjectiveConfig:
    beta: float = 0.02
    lambda_edit: float = 0.1
    lambda_step: float = 0.01
    alpha: float = 0.3
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    advantage_normalize: bool = True
    value_loss: str = 'clipped_mse'
    use_progress_reward: bool = True
    terminal_reward: bool = False
    use_internalization: bool = True
    stop_gets_return: bool = False
    energy_noise_std: float = 0.0
    target_selector: str = 'energy'
    si_reduction: str = 'sum'

@dataclass
class RolloutConfig:
    t_max: int = 6
    k_candidates: int = 7
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 512
    proposal_max_new_tokens: int = 256
    revise_min_span_chars: int = 24
    revise_max_span_chars: int = 900
    revise_max_patches: int = 1
    switch_candidates: int = 2
    switch_temperature: float = 0.9
    edit_distance: str = 'span_levenshtein'
    edit_cap: float = 1.0
    edit_normalization: str = 'answer'
    allowed_actions: Tuple[str, ...] = ('revise', 'switch', 'stop')
    switch_resets_coordinate: bool = True
    normalize_math_tokens: bool = True
    normalize_code_tokens: bool = True
    frozen_rollout_pool: bool = False

@dataclass
class OptimConfig:
    optimizer: str = 'muon'
    lr: float = 2e-05
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500
    total_steps: int = 15000
    min_lr: float = 1e-07
    schedule: str = 'cosine'
    prompts_per_update: int = 32
    rollout_batch_size: int = 4
    gradient_accumulation: int = 1
    updates: int = 200
    max_prompt_tokens: int = 512

@dataclass
class DistributedConfig:
    world_size: int = 8
    required_total_memory_gb: float = 640.0
    sharding: str = 'fsdp'
    backend: str = 'auto'
    activation_checkpointing: bool = True
    checkpoint_every: int = 2
    allow_tf32: bool = True
    hardware: str = '8 x NVIDIA A100 80GB'
    target_gpu_hours: float = 2983.2
    target_peak_memory_gb: float = 61.9
    target_elapsed_hours: float = 372.9

@dataclass
class EnergyFitConfig:
    enabled: bool = True
    steps: int = 200
    lr: float = 2e-05
    batch_pairs: int = 32
    margin: float = 0.2
    loss: str = 'margin'
    states_per_prompt: int = 6
    freezes: Tuple[str, ...] = ('text_encoder', 'lift_projection', 'transition_block', 'energy_head', 'action_embedding')
    negatives: Tuple[str, ...] = ('failed', 'verifier_rejected', 'longer_no_gain', 'noisy', 'terminal_only')

@dataclass
class DataConfig:
    mixture: Dict[str, int] = field(default_factory=lambda: {'math': 38000, 'code': 24000, 'science': 18000, 'logic': 12000})
    sources: Dict[str, str] = field(default_factory=lambda: {'math': 'open-r1/OpenR1-Math-220k', 'code': 'BAAI/TACO', 'science': 'allenai/sciq', 'logic': 'longface/logicLM'})
    holdout_suites: Tuple[str, ...] = ('gsm8k', 'arc_challenge', 'bbh', 'humaneval', 'mbpp')
    main_suites: Tuple[str, ...] = ('math500', 'aime25', 'gpqa', 'livecodebench', 'zebralogic', 'mmlu_pro')
    offline_synthetic: bool = True
    offline_train_size: int = 256
    offline_eval_size: int = 64
    max_answer_chars: int = 4000
    allow_code_exec: bool = False

@dataclass
class EvalConfig:
    seeds: Tuple[int, ...] = (42, 1234, 3407)
    pass_k: Tuple[int, ...] = (1, 16)
    reflection_passes: int = 2
    eval_batch_size: int = 1
    max_new_tokens: int = 4096
    warmup_requests: int = 1
    selection_budgets: Tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    bootstrap_samples: int = 1000
    energy_bins: int = 5
    metrics: Tuple[str, ...] = ('accuracy', 'net_gain', 'span_transitions', 'energy', 'latency', 'tokens', 'ordering')

@dataclass
class RunConfig:
    run_name: str = 'lspo'
    output_dir: str = 'runs'
    seed: int = 42
    stage: str = 'all'
    variant: str = 'full'
    log_every: int = 10
    save_every: int = 100
    verbose: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    energy_fit: EnergyFitConfig = field(default_factory=EnergyFitConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and (not isinstance(obj, type)):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def _apply_dict(obj: Any, values: Dict[str, Any]) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key.startswith('_'):
            continue
        if key not in known:
            raise KeyError(f'unknown config key {key!r} for {type(obj).__name__}')
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)

def _coerce_scalar(text: str) -> Any:
    lowered = text.strip()
    if lowered.lower() in {'true', 'false'}:
        return lowered.lower() == 'true'
    if lowered.lower() in {'none', 'null'}:
        return None
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    if lowered.startswith('[') or lowered.startswith('{'):
        return json.loads(lowered)
    return text

def apply_config_dict(config: 'RunConfig', values: Dict[str, Any]) -> 'RunConfig':
    _apply_dict(config, values)
    return config

def load_config(path: Optional[str]=None, overrides: Optional[List[str]]=None) -> RunConfig:
    cfg = RunConfig()
    if path:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        _apply_dict(cfg, raw)
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'override must look like key.path=value, got {item!r}')
        key, _, value = item.partition('=')
        parts = key.strip().split('.')
        target: Any = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f'unknown config path segment {part!r} in {key!r}')
            target = getattr(target, part)
        _apply_dict(target, {parts[-1]: _coerce_scalar(value)})
    return cfg

def save_config(cfg: RunConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_to_jsonable(cfg), indent=2, ensure_ascii=False), encoding='utf-8')

def clone(cfg: RunConfig) -> RunConfig:
    return copy.deepcopy(cfg)
