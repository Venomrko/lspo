from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lspo.utils.logging import load_records
PAPER_TABLE_E7 = {
    "DPO": {"peak_mem_gb": 57.7, "elapsed_h": 152.2, "gpu_hours": 1217.6,
            "pilot_gpu_hours": 146.1, "latency_s": 10.42, "output_tokens": 1196},
    "Step-DPO": {"peak_mem_gb": 57.7, "elapsed_h": 203.1, "gpu_hours": 1624.8,
                 "pilot_gpu_hours": 195.0, "latency_s": 10.73, "output_tokens": 1242},
    "GRPO": {"peak_mem_gb": 57.7, "elapsed_h": 298.3, "gpu_hours": 2386.4,
             "pilot_gpu_hours": 286.4, "latency_s": 12.04, "output_tokens": 1370},
    "GDPO": {"peak_mem_gb": 57.7, "elapsed_h": 335.6, "gpu_hours": 2684.8,
             "pilot_gpu_hours": 322.2, "latency_s": 11.54, "output_tokens": 1324},
    "SCoRe": {"peak_mem_gb": 57.7, "elapsed_h": 358.2, "gpu_hours": 2865.7,
              "pilot_gpu_hours": 343.9, "latency_s": 13.12, "output_tokens": 1501},
    "LSPO": {"peak_mem_gb": 61.9, "elapsed_h": 372.9, "gpu_hours": 2983.2,
             "pilot_gpu_hours": 358.0, "latency_s": 10.93, "output_tokens": 1265},
}
def collect(run_dir: Path) -> Optional[Dict[str, float]]:
    logs = sorted(run_dir.glob("*.jsonl"))
    if not logs:
        return None
    records: List[Dict] = []
    for log in logs:
        records.extend(load_records(log))
    if not records:
        return None
    updates = [r for r in records if r.get("stage") == "lspo_update"]
    peak = max((r.get("peak_memory_gb", 0.0) for r in records), default=0.0)
    wall_clock = statistics.fmean(
        [r["wall_clock_update_s"] for r in updates if "wall_clock_update_s" in r]
    ) if updates else 0.0
    totals = [r["elapsed_total_s"] for r in records if "elapsed_total_s" in r]
    elapsed = max(totals) if totals else max((r.get("wall_clock_s",0.0) for r in records),default=0.0)
    count = max((r.get("policy_updates",r.get("step",0)) for r in records
                 if r.get("stage") in {"lspo_update","training_summary"}),default=0)
    world = max((r.get("compute_world_size", 1.0) for r in records), default=1.0)
    return {
        "peak_mem_gb": peak,
        "elapsed_h": elapsed / 3600.0,
        "gpu_hours": elapsed / 3600.0 * world,
        "updates": float(count),
        "mean_update_s": wall_clock,
        "world_size": world,
    }
def print_table(rows: Dict[str, Dict[str, float]],
                latency: Optional[float], tokens: Optional[float]) -> None:
    header = (f"{'Run':<16}{'Peak mem':>10}{'Elapsed h':>11}{'GPU hrs':>11}"
              f"{'Updates':>9}{'s/update':>10}")
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        print(
            f"{name:<16}{row['peak_mem_gb']:>10.1f}{row['elapsed_h']:>11.2f}"
            f"{row['gpu_hours']:>11.1f}{row['updates']:>9.0f}"
            f"{row['mean_update_s']:>10.1f}"
        )
    print()
    reference = PAPER_TABLE_E7["LSPO"]
    print("Table E7 reference (LSPO, Qwen3.5-27B on 8 x A100 80GB):")
    print(f"  peak mem {reference['peak_mem_gb']} GB | elapsed "
          f"{reference['elapsed_h']} h | {reference['gpu_hours']} GPU-hours | "
          f"latency {reference['latency_s']} s | {reference['output_tokens']} tokens")
    if latency is not None or tokens is not None:
        print(f"  measured deployment: latency {latency} s | tokens {tokens}")
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="run directories containing *.jsonl logs")
    parser.add_argument("--latency", type=float, default=None,
                        help="measured mean deployment latency in seconds")
    parser.add_argument("--tokens", type=float, default=None,
                        help="measured mean output tokens")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()
    rows: Dict[str, Dict[str, float]] = {}
    for run in args.runs:
        metrics = collect(Path(run))
        if metrics is None:
            print(f"[warn] no usable log found in {run}")
            continue
        rows[Path(run).name] = metrics
    if not rows:
        print("nothing to report")
        return 1
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows, args.latency, args.tokens)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
