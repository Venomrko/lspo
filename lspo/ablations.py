from __future__ import annotations
from typing import Any, Dict
VARIANT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_lift": {"variant": "no_lift"},
    "no_lift_mlp": {"variant": "no_lift_mlp"},
    "random_z": {"variant": "random_z"},
    "nuisance_z": {"variant": "nuisance_z"},
    "frozen_projection": {"variant": "frozen_projection"},
    "no_progress_reward": {"objective": {"use_progress_reward": False}},
    "terminal_reward": {"objective": {"terminal_reward": True}},
    "terminal_only": {
        "objective": {"terminal_reward": True, "use_progress_reward": False}
    },
    "outcome_rl": {
        "objective": {
            "terminal_reward": True,
            "use_progress_reward": False,
            "use_internalization": False,
        }
    },
    "no_internalization": {"objective": {"use_internalization": False}},
    "si_only": {"objective": {"target_selector": "energy"}},
    "select_sft": {"objective": {"target_selector": "verifier"}},
    "revise_only": {"rollout": {"allowed_actions": ["revise", "stop"]}},
    "switch_only": {"rollout": {"allowed_actions": ["switch", "stop"]}},
    "no_switch": {"rollout": {"switch_resets_coordinate": False}},
    "no_margin": {"energy_fit": {"margin": 0.0}},
    "noisy_energy": {"objective": {"energy_noise_std": 0.15}},
    "flipped_energy": {
        "objective": {"energy_noise_std": 5.0},
    },
    "no_energy_stop": {"objective": {"terminal_reward": True}},
    "dz64": {"model": {"d_z": 64}},
    "dz2": {"model": {"d_z": 2}},
}
FIXED_POOL_VARIANTS = {"si_only", "select_sft"}
NO_CONTROLLER_VARIANTS = {"si_only", "select_sft"}
def variant_names():
    return sorted(VARIANT_OVERRIDES)
def apply_variant(config, name: str):
    if name not in VARIANT_OVERRIDES:
        raise KeyError(f"unknown variant {name!r}; known: {variant_names()}")
    from .config import apply_config_dict
    overrides = dict(VARIANT_OVERRIDES[name])
    special_variant = overrides.pop("variant", None)
    if special_variant is not None:
        config.variant = special_variant
    if overrides:
        apply_config_dict(config, overrides)
    if name in NO_CONTROLLER_VARIANTS:
        config.rollout.frozen_rollout_pool = True
    return config
