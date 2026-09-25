from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lspo.config import load_config, save_config
from lspo.lightning.runner import memory_budget_report, LARGE_BACKBONES
from lspo.train.compute import estimate_updates_from_paper

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--devices", type=int, default=None,
                        help="GPUs per node (default: config.distributed.world_size)")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"],
                        help="use cpu with --devices 1 to exercise the loop on a small model")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="override config.eval.seeds")
    parser.add_argument("--variant", default=None, help="ablation variant")
    parser.add_argument("--stage", default="all", choices=["all", "energy", "policy"])
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--execute", dest="dry_run", action="store_false")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def preflight(config, devices: int) -> List[str]:

    lines: List[str] = []
    try:
        import torch
        lines.append(f"torch              : {torch.__version__} (cuda {torch.version.cuda})")
        if not torch.cuda.is_available():
            lines.append("  !! no CUDA device visible -- this build cannot train on GPU")
            return lines
        count = torch.cuda.device_count()
        device_gb = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            gpu_gb = properties.total_memory / 2**30
            device_gb.append(gpu_gb)
            lines.append(f"  gpu{index}            : {properties.name} {gpu_gb:.0f} GB")
        total_gb, smallest_gb, notes = memory_budget_report(config, devices)
        lines.append(
            f"gpus visible       : {count} (launching with {devices}), "
            f"{total_gb:.0f} GB total"
        )
        for note in notes:
            lines.append(f"  !! {note}")
    except ImportError:
        lines.append("torch              : NOT INSTALLED")
        return lines
    world = devices
    backend = config.model.backbone
    lines.append(f"backbone           : {backend}")
    if any(tag in backend for tag in LARGE_BACKBONES):
        weights_gb = (2 * 27 * 2) if "27B" in backend else (2 * 35 * 2)
        per_rank = weights_gb / world
        lines.append(
            f"sharded weights    : ~{per_rank:.1f} GB/rank (two bf16 copies of "
            f"{weights_gb:.0f} GB total over {world} ranks)"
        )
        lines.append(
            f"memory per GPU     : {smallest_gb:.0f} GB visible "
            f"(paper peak {config.distributed.target_peak_memory_gb} GB/GPU, Table E7); "
            f"~{max(smallest_gb - per_rank, 0.0):.0f} GB left once weights are sharded"
        )
        if per_rank > 0.85 * smallest_gb:
            lines.append(
                "  !! weights alone exceed 85% of the smallest visible GPU; "
                "increase --devices"
            )
    if world == 1 and "27B" in backend:
        lines.append(
            "  !! sharding='fsdp' needs more than one rank; a single process will "
            "materialise both 27B copies on one GPU"
        )
    lines.append(
        f"optimizer          : {config.optim.optimizer} lr={config.optim.lr} "
        f"momentum={config.optim.momentum} ns_steps={config.optim.ns_steps}"
    )
    lines.append(
        f"objective          : beta={config.objective.beta} "
        f"lambda_edit={config.objective.lambda_edit} "
        f"lambda_step={config.objective.lambda_step} alpha={config.objective.alpha}"
    )
    lines.append(
        f"rollout            : T_max={config.rollout.t_max} "
        f"K={config.rollout.k_candidates} d_z={config.model.d_z} "
        f"temp={config.rollout.temperature} top_p={config.rollout.top_p}"
    )
    lines.append(
        f"global batch       : {config.optim.prompts_per_update} prompts/update "
        f"(micro-batch {config.optim.rollout_batch_size})"
    )
    corpus = sum(config.data.mixture.values())
    lines.append(f"training mixture   : {corpus:,} samples (Table C2)")
    lines.append(
        f"schedule           : {config.optim.updates} updates, "
        f"{config.optim.warmup_steps} warmup, cosine -> {config.optim.min_lr}"
    )
    lines.append(
        f"per-update budget  : {estimate_updates_from_paper():.1f} s "
        f"(from Table E7: {config.distributed.target_elapsed_hours} h / "
        f"{config.optim.total_steps} steps)"
    )
    expected_hours = estimate_updates_from_paper() * config.optim.updates / 3600.0
    lines.append(
        f"projected elapsed  : {expected_hours:.1f} h  "
        f"-> {expected_hours * world:.0f} GPU-hours "
        f"(paper: {config.distributed.target_elapsed_hours} h / "
        f"{config.distributed.target_gpu_hours} GPU-hours)"
    )
    return lines


def build_command(config_path: Path, devices: int, stage: str, seed: int,
                  variant: str, output_dir: str, overrides: List[str],
                  accelerator: str = "gpu") -> List[str]:

    command = [
        sys.executable, str(REPO / "scripts" / "train.py"),
        "--config", str(config_path),
        "--devices", str(devices),
        "--accelerator", accelerator,
        "--stage", stage,
        "--seed", str(seed),
        "--output-dir", output_dir,
    ]
    if variant:
        command += ["--variant", variant]
    for override in overrides:
        command += ["--set", override]
    return command


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(str(config_path), args.overrides)
    if args.variant:
        config.variant = args.variant
    devices = args.devices or config.distributed.world_size
    seeds = args.seeds or list(config.eval.seeds)
    print("=" * 78)
    print(f" LSPO launch plan -- {config.model.backbone} on {devices} GPU(s)")
    print("=" * 78)
    for line in preflight(config, devices):
        print(line)
    print("-" * 78)
    if len(seeds) > 1:
        print(
            f"Table E7 reports the *mean per-run* cost across seeds "
            f"{list(config.eval.seeds)}; launching {len(seeds)} runs sequentially."
        )
    print(f"runs: {len(seeds)}  seeds={seeds}")
    total_gpu_hours = 0.0
    for seed in seeds:
        output_dir = f"{config.output_dir}/seed{seed}"
        resolved = Path(output_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        seed_config = load_config(str(config_path), args.overrides)
        seed_config.seed = seed
        seed_config.output_dir = output_dir
        if args.variant:
            seed_config.variant = args.variant
        save_config(seed_config, resolved / "config_resolved.json")
        command = build_command(
            config_path, devices, args.stage, seed,
            args.variant or "", output_dir, args.overrides, args.accelerator,
        )
        print("-" * 78)
        print(f"seed {seed} -> {output_dir}")
        print("  " + " ".join(command))
        total_gpu_hours += (
            estimate_updates_from_paper() * seed_config.optim.updates / 3600.0 * devices
        )
        if not args.dry_run:
            env = dict(os.environ)
            env["LSPO_SEED"] = str(seed)
            completed = subprocess.run(command, cwd=str(REPO), env=env)
            if completed.returncode != 0:
                print(f"seed {seed} exited with code {completed.returncode}")
                return completed.returncode
    print("=" * 78)
    print(f"projected total: {total_gpu_hours:.0f} GPU-hours across {len(seeds)} run(s)")
    print(f"paper reference: {config.distributed.target_gpu_hours} GPU-hours per run "
          f"on {config.distributed.hardware}")
    if args.dry_run:
        print("dry run -- re-run with --execute to launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
