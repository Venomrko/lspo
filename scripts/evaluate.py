from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lspo.config import load_config
from lspo.data.benchmarks import assert_disjoint, load_main_suites, load_holdout_suites
from lspo.data.corpus import build_training_items
from lspo.eval.evaluate import (
    evaluate_energy_ordering,
    evaluate_selection_budget,
    evaluate_transitions,
    summarize_cost,
)
from lspo.eval.metrics import aggregate_accuracy
from lspo.lightning.data import identity
from lspo.lightning.module import LSPOModule
from lspo.lightning.runner import build_trainer
from lspo.models.backbone import format_prompt
from lspo.utils.logging import RunLogger
from lspo.utils.seed import set_seed


def suite_transitions(model, items, config):

    before, after = [], []
    for item in items:
        trajectory = model.collect([item])[0]
        before.append(bool(model.checker(item, trajectory.direct_text).verified))
        after.append(bool(model.checker(item, trajectory.selection.text).verified))
    metrics = evaluate_transitions(before, after)
    return {
        "net_gain": metrics.net_gain,
        "repair_pct": metrics.repair_pct,
        "damage_pct": metrics.damage_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paper.json"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"],
                        help="use cpu with --devices 1 to evaluate on a small model")
    parser.add_argument("--limit", type=int, default=None, help="items per suite")
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-transitions", action="store_true",
                        help="skip the extra direct-versus-selected audit")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.output_dir:
        config.output_dir = args.output_dir
    set_seed(config.seed)
    logger = None

    suites = load_main_suites(config, limit=args.limit)
    holdout = load_holdout_suites(config, limit=args.limit)
    every_eval_item = [i for group in (suites, holdout) for items in group.values() for i in items]
    assert_disjoint(build_training_items(config), every_eval_item)

    model = LSPOModule(config)
    model.eval_with_reflection = not args.no_reflection
    trainer = build_trainer(config, 1, args.devices, args.accelerator, checkpointing=False)
    batches = ([("main", n, i) for n, i in suites.items()]
               + [("held_out", n, i) for n, i in holdout.items()])
    loader = DataLoader(batches, batch_size=None, collate_fn=identity, num_workers=0)
    rows = trainer.predict(model, dataloaders=loader, ckpt_path=args.checkpoint,
                           weights_only=False)

    if trainer.is_global_zero:
        logger = RunLogger(config.output_dir, f"{config.run_name}_eval", verbose=config.verbose)
    model.eval()
    model._seed_update()
    results, heldout_results = {}, {}
    for split, name, payload in rows:


        suite_payload = payload.get(name)
        if suite_payload is None:
            continue
        (results if split == "main" else heldout_results)[name] = suite_payload
    if results:
        names = [n for n in results if n != "average"]
        results["average"] = {
            f"pass@{k}": aggregate_accuracy(
                {n: results[n][f"pass@{k}"] for n in names if f"pass@{k}" in results[n]})
            for k in config.eval.pass_k
        }
    cost = summarize_cost(results)

    ordering = {}
    for split, collection in (("in_domain", suites), ("held_out", holdout)):
        ordering[split] = {
            name: evaluate_energy_ordering(
                model.generator, model.checker, model.scoring, items, config,
                model.family_to_index, collector=model.collector,
                calibration=model.energy_calibration,
            ).as_dict()
            for name, items in collection.items()
        }
    budgets = evaluate_selection_budget(
        model.generator, model.checker, model.scoring, next(iter(suites.values())),
        config, model.family_to_index, logger=logger,
    )
    transitions = {} if args.no_transitions else suite_transitions(
        model, next(iter(suites.values())), config)

    summary = {
        "suites": results,
        "cost": cost,
        "energy_ordering": ordering,
        "heldout_results": heldout_results,
        "code_verification": "execution" if config.data.allow_code_exec else "static_only",
        "evaluation_max_new_tokens": config.eval.max_new_tokens,
        "latency_batch_size": 1,
        "selection_budget": {str(k): v for k, v in budgets.items()},
        "holdout_suites": list(holdout),
        "transitions": transitions,
    }
    if not trainer.is_global_zero:
        return 0
    out = Path(args.output) if args.output else Path(config.output_dir) / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"wrote {out}")
    logger.close()
    print(json.dumps({"average_pass@1": results.get("average", {}).get("pass@1"),
                      "cost": cost, "transitions": transitions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
