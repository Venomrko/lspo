from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lspo.config import load_config
from lspo.lightning.runner import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs"/"paper.json"))
    parser.add_argument("--stage",choices=["all","energy","policy"],default="all")
    parser.add_argument("--variant",default=None,help="ablation variant")
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--output-dir",default=None)
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument("--devices",type=int,default=8)
    parser.add_argument("--accelerator",default="gpu",choices=["gpu","cpu"],
                        help="use cpu with --devices 1 to exercise the loop on a small model")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--set",action="append",default=[])
    args = parser.parse_args()
    config = load_config(args.config,args.set)
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.variant is not None:
        config.variant = args.variant
    print(run(config,args.stage,args.resume,args.limit,args.devices,
              accelerator=args.accelerator,initialize_from=args.initialize_from))

if __name__ == "__main__":
    main()
