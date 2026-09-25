# LSPO — Lightning Implementation

This directory implements LSPO with PyTorch Lightning. Lightning owns the training loop, process launch, optimizer stepping, logging, and checkpoints. The energy-ranking loss, controller objective, lifted transitions, and answer internalization reuse the existing implementation.

## Contributions

1. Internalize selected low-energy answers into a generator used for direct generation.
2. Assign transition credit through energy decrease minus edit and step costs.
3. Carry refinement context through a learned continuous coordinate.

## Hardware Requirement

The minimum release configuration is 8 NVIDIA A100 80GB GPUs, or equivalent or greater GPU memory and throughput. The paper results were obtained with 8 NVIDIA A100 80GB GPUs.

## Installation

Use Python 3.12 in a separate environment. Install a CUDA-enabled PyTorch build (2.10.0 or later) appropriate for the host, then install the dependencies:

```bash
pip install -r requirements.txt
```

The minimum versions are PyTorch 2.10.0, Lightning and PyTorch Lightning 2.6.6, and Transformers 5.2.0. Transformers 5.2.0 includes the Qwen3.5 text-generation mapping used by `AutoModelForCausalLM` for the default `Qwen/Qwen3.5-27B` backbone. See the [upstream model mapping](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/auto/modeling_auto.py). Other dependencies are resolved under these packages' declared constraints. Model weights and datasets must be accessible separately.

## Training

```bash
python scripts/train.py --config configs/paper.json --devices 8
```

A single Lightning run first fits the scoring branch, freezes it, and then runs controller RL with answer internalization. Every parameter uses Muon. Scalars and vectors use a single-row matrix view; higher-dimensional tensors flatten their trailing dimensions. The optimizer has no AdamW fallback.

```bash
python scripts/train.py --config configs/paper.json --resume runs/paper_27b/checkpoints/last.ckpt
python scripts/train.py --config configs/paper.json --stage energy
python scripts/train.py --config configs/paper.json --stage policy --initialize-from runs/paper_27b/final.ckpt
```

`--resume` restores the same training stage and configuration, including optimizer state and update counters. `--initialize-from` transfers model weights and calibration from a completed energy checkpoint into a new policy-only run. Keep separate output directories for independent runs. Use `--set seed=1234 --set output_dir=runs/seed1234` for another seed.

## Evaluation

```bash
python scripts/evaluate.py --config configs/paper.json --checkpoint runs/paper_27b/final.ckpt --devices 8 --output evaluation.json
```

Evaluation loads all six primary suites and five held-out suites. Data-loading failures stop evaluation. Code verification remains static by default and is identified in the output. Each input receives 16 samples for pass@16. The first sample determines pass@1 and generation cost. Latency is measured per request after warmup with device synchronization. The 4096-token evaluation cap is an implementation default, not a recovered experiment setting; truncation rates are reported.

## Distributed Execution

The GPU strategy uses FSDP2 for the two backbone copies and synchronized replicated heads. Tensor parallelism is disabled. Online trajectories can have different action counts and generation lengths, so this version deliberately uses identical prompt batches and sampling seeds on all ranks. This keeps FSDP forward collectives in the same order. `prompts_per_update` counts unique prompts, not duplicated rank-local copies. This correctness-first mode does not provide independent per-rank rollout throughput and should not be used to reproduce the paper's wall-clock figure without further measurement.

## Checkpoints

Checkpoints contain model and reference weights, both Muon states, stage counters, the training-derived energy calibration, and Lightning loop state. Learning-rate positions are reconstructed from the saved counters. Energy states are regenerated deterministically from the fixed initial generator when resuming energy fitting. No evaluation labels are used for calibration.

## Repository Layout

```text
lspo_repro_lightning/
|-- configs/
|   |-- paper.json
|-- scripts/
|   |-- train.py
|   |-- evaluate.py
|-- lspo/
|   |-- lightning/
|   |   |-- module.py       Two-stage LightningModule and manual optimization
|   |   |-- data.py         Prompt batches and deterministic update ordering
|   |   |-- runner.py       Trainer, strategy, logging, and checkpoints
|   |-- models/            Generator, scoring branch, controller, and value head
|   |-- env/               Rollouts, proposals, candidates, and static verifiers
|   |-- train/             Muon, losses, GAE, and state-sampling helpers
|   |-- data/              Training and benchmark adapters
|   |-- eval/              Accuracy, ordering, and generation-cost metrics
|   |-- baselines/         Reusable baseline components
|   |-- utils/             Text and device helpers
|-- requirements.txt
|-- LICENSE
|-- README.md
```

## Validation Boundary

The CPU checks exercise the real Lightning Trainer on a small internal model. They do not reproduce paper accuracy or validate eight-GPU FSDP2 execution. No low-resource training configuration or smoke-test entry point is included in this release.

Validated locally with Lightning 2.6.6 and PyTorch 2.10.0 (CPU): two-stage training, loss and gradient equality with the reused reference functions, checkpoint-based prediction including pass@16, and interrupted energy-stage recovery with zero final parameter difference. Eight-GPU execution remains unverified.

Resume validates the seed, variant, training limit, and an ordered fingerprint of the training records. Checkpoints created before these checks were introduced cannot provide verified exact resume. Resource reports use the saved policy-update counter and cumulative active training time across resumed sessions. Ablation evaluation uses complete suites unless `--eval-limit` is explicitly supplied.

## License

The LSPO code is released under the [MIT License](LICENSE), copyright 2026 LSPO authors. Model weights, datasets, and third-party dependencies retain their respective licenses.
