from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lspo.ablations import variant_names
from lspo.config import clone, load_config, save_config
from lspo.eval.metrics import sample_std

REPO = Path(__file__).resolve().parents[1]


def run_command(command) -> int:
    completed = subprocess.run(command, cwd=str(REPO))
    return completed.returncode


def run_one(base_config, base_config_path: Path, variant: str, seed: int,
            devices: int, accelerator: str, limit, eval_limit) -> dict:

    out_dir = Path(base_config.output_dir) / f"{variant}_s{seed}"
    resolved = clone(base_config)
    resolved.variant = variant
    resolved.seed = seed
    resolved.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = out_dir / "config_resolved.json"
    save_config(resolved, resolved_path)

    train = [sys.executable, str(REPO / "scripts" / "train.py"),
             "--config", str(resolved_path), "--devices", str(devices),
             "--accelerator", accelerator, "--variant", variant, "--seed", str(seed),
             "--output-dir", str(out_dir)]
    if limit is not None:
        train += ["--limit", str(limit)]
    code = run_command(train)
    if code != 0:
        raise RuntimeError(f"training failed for variant={variant} seed={seed} (exit {code})")

    eval_path = out_dir / "evaluation.json"
    evaluate = [sys.executable, str(REPO / "scripts" / "evaluate.py"),
                "--config", str(resolved_path), "--devices", str(devices),
                "--accelerator", accelerator, "--checkpoint", str(out_dir / "final.ckpt"),
                "--output", str(eval_path)]
    if eval_limit is not None:
        evaluate += ["--limit", str(eval_limit)]
    code = run_command(evaluate)
    if code != 0:
        raise RuntimeError(f"evaluation failed for variant={variant} seed={seed} (exit {code})")

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    transitions = payload.get("transitions", {})
    return {
        "variant": variant,
        "seed": seed,
        "aggregate": payload.get("suites", {}).get("average", {}).get("pass@1"),
        "net_gain": transitions.get("net_gain"),
        "repair_pct": transitions.get("repair_pct"),
        "damage_pct": transitions.get("damage_pct"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs" / "paper.json"))
    parser.add_argument("--variants", nargs="*",
                        default=["full", "no_lift", "no_progress_reward", "no_internalization"])
    parser.add_argument("--limit", type=int, default=None, help="training items")
    parser.add_argument("--eval-limit", type=int, default=None, help="items per suite")
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    invalid = [v for v in args.variants if v not in variant_names()]
    if invalid:
        parser.error(f"unknown variants: {invalid}; known: {variant_names()}")
    base_config = load_config(args.config, args.overrides)
    if args.output_dir:
        base_config.output_dir = args.output_dir
    seeds = args.seeds or list(base_config.eval.seeds)
    rows = []
    for variant in args.variants:
        per_seed = [
            run_one(base_config, Path(args.config), variant, seed,
                    args.devices, args.accelerator, args.limit, args.eval_limit)
            for seed in seeds
        ]
        aggregate = [r["aggregate"] for r in per_seed]
        net_gain = [r["net_gain"] for r in per_seed]
        rows.append({
            "variant": variant,
            "aggregate_mean": sum(aggregate) / len(aggregate),
            "aggregate_sd": sample_std(aggregate),
            "net_gain_mean": sum(net_gain) / len(net_gain),
            "per_seed": per_seed,
        })
    table = Path(base_config.output_dir) / "ablation_table.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
