from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lspo.baselines import BASELINES, build_baseline
from lspo.config import load_config, save_config
from lspo.data.corpus import build_training_items
from lspo.env.verifiers import build_checker
from lspo.models.backbone import build_backbone
from lspo.utils.logging import RunLogger
from lspo.utils.seed import resolve_device, set_seed


def resolve_items(config, limit=None):
    items = build_training_items(config, limit=limit)
    if not items:
        raise RuntimeError('the training mixture is empty')
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--method", default="grpo", choices=list(BASELINES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.output_dir:
        config.output_dir = args.output_dir
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    save_config(config, Path(config.output_dir) / f"config_{args.method}.json")
    logger = RunLogger(config.output_dir, f"{config.run_name}_{args.method}",
                       verbose=config.verbose)
    device = resolve_device(config.model.device)
    generator = build_backbone(config.model).to(device)
    checker = build_checker(config)
    items = resolve_items(config, limit=args.limit)
    trainer = build_baseline(args.method, generator, checker, config, logger=logger)
    logger.info(f"method={args.method}  items={len(items)}")
    trainer.train(items)
    checkpoint = Path(config.output_dir) / f"{config.run_name}_{args.method}_checkpoint.pt"
    import torch
    torch.save({"generator": generator.state_dict(), "config": config}, checkpoint)
    logger.info(f"checkpoint: {checkpoint}")
    logger.close()
    print(json.dumps({"method": args.method, "checkpoint": str(checkpoint)}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
