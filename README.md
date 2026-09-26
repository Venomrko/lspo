# Self-Correction as Transition Geometry: Internalizing Reasoning via Lifted State Policy Optimization

**NeurIPS 2026** · Ren Zhuang, Ben Wang, Shuifa Sun

Code for **Lifted State Policy Optimization (LSPO)**.

## Overview

LSPO trains language models to internalize answer correction. Each answer is paired with a continuous auxiliary coordinate that carries refinement context. A controller learns local revisions from energy decreases, balanced against edit and step costs. Selected answers then become training targets for the generator. At deployment, the model produces answers through direct generation.

![Qwen3.5-27B benchmark results, rollout-to-direct alignment, and energy separation](assets/figure1.png)

Benchmark performance on Qwen3.5-27B, alignment between rollout-selected answers and direct generation, and final-energy distributions for correct and incorrect answers.

### Method

![Comparison of answer-level optimization, trajectory RL, and LSPO](assets/method.png)

**Comparison with answer-level optimization and trajectory RL.** LSPO assigns credit to transitions in the lifted state space and internalizes the selected answers.

<img src="assets/formulas_combined.png" alt="LSPO reward and optimization objective" width="80%" />

**Training objective.** Energy decrease supplies transition credit, with edit and step penalties. KL regularization constrains policy updates, while answer internalization trains the generator on selected answers.

<p align="center">
  <img src="assets/fig1_drawio.png" alt="LSPO framework: lifted states and answer internalization" width="80%" />
</p>

**LSPO framework.** Correction proceeds through lifted states, and selected answers provide targets for direct generation.

### Benchmark Results

The paper evaluates Qwen3.5-9B and Qwen3.5-27B dense models, together with the Qwen3.5-35B-A3B mixture-of-experts model. Each figure reports results across six benchmarks. Hatched bars show pass@1; full bar heights show pass@16.

**Qwen3.5-9B**

<p align="center">
  <img src="assets/figure6_pass16_benchmarks_qwen35_9b.png" alt="Qwen3.5-9B benchmark results" width="80%" />
</p>

**Qwen3.5-27B**

<p align="center">
  <img src="assets/figure6_pass16_benchmarks_qwen35_27b.png" alt="Qwen3.5-27B benchmark results" width="80%" />
</p>

**Qwen3.5-35B-A3B**

<p align="center">
  <img src="assets/figure6_pass16_benchmarks_qwen35_35b_a3b.png" alt="Qwen3.5-35B-A3B benchmark results" width="80%" />
</p>

### Quality and Internalization

![Quality, sampling, correction transitions, and output cost on Qwen3.5-27B](assets/results.png)

**Quality and cost on Qwen3.5-27B.** LSPO achieves an aggregate pass@1 score of 71.8%, with an average latency of 10.93 seconds and 1,265 output tokens in the paper's evaluation.

### Training Dynamics

<p align="center">
  <img src="assets/figure5_training_dynamics.png" alt="Transition stability, net gain, and direct performance during training" width="50%" />
</p>

**Training dynamics.** Transition stability, net correction gain, and direct pass@1 over 15,000 training steps.

<p align="center">
  <img src="assets/geometric_descent.png" alt="Energy during policy training" width="80%" />
</p>

**Energy during policy training.** Highlighted points mark improvements in the lowest energy observed so far; the line tracks that running minimum across policy updates.

Component ablations on Qwen3.5-27B:

| Variant | Aggregate pass@1 (%) | Average latency (s) | Average output tokens |
| --- | ---: | ---: | ---: |
| **LSPO** | **71.8** | **10.93** | **1,265** |
| Without lifting | 70.9 | 11.70 | 1,316 |
| Without progress reward | 70.0 | 12.35 | 1,366 |
| Without internalization | 70.3 | 12.02 | 1,341 |

## Initialization

Use Python 3.12 and a CUDA-enabled PyTorch installation. Minimum package versions are listed in [requirements.txt](requirements.txt).

```bash
git clone https://github.com/Venomrko/lspo.git
cd lspo
pip install -r requirements.txt
```

Minimum hardware: **8 × NVIDIA A100 80GB GPUs**, or GPUs with equivalent or greater memory and compute capacity. The paper experiments used 8 × NVIDIA A100 80GB GPUs.

## Data Preparation

Training sources are specified in `data.sources` in [configs/paper.json](configs/paper.json). The loader downloads their training splits through Hugging Face Datasets. Ensure that the datasets and the configured model weights are accessible before training.

| Domain | Dataset | Configured prompt budget |
| --- | --- | ---: |
| Mathematics | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 38,000 |
| Code | [TACO](https://huggingface.co/datasets/BAAI/TACO) | 24,000 |
| Science | [SciQ](https://huggingface.co/datasets/allenai/sciq) | 18,000 |
| Logic | [LogicLM](https://huggingface.co/datasets/longface/logicLM) | 12,000 |

Set the dataset identifiers in `data.sources` and the per-domain budgets in `data.mixture`. The default backbone is [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), configured through `model.backbone`.

## Training

Train with the default configuration:

```bash
python scripts/train.py --config configs/paper.json --devices 8
```

For a separate seed and output directory:

```bash
python scripts/train.py --config configs/paper.json --devices 8 --seed 1234 --output-dir runs/seed1234
```

Resume a run:

```bash
python scripts/train.py --config configs/paper.json --devices 8 --resume runs/paper_27b/checkpoints/last.ckpt
```

Evaluate the trained checkpoint:

```bash
python scripts/evaluate.py --config configs/paper.json --checkpoint runs/paper_27b/final.ckpt --devices 8 --output evaluation.json
```

## License

This code is released under the [MIT License](LICENSE). Model weights and datasets retain their respective licenses.

## Citation

```bibtex
@inproceedings{zhuang2026lspo,
  title = {Self-Correction as Transition Geometry: Internalizing Reasoning via Lifted State Policy Optimization},
  author = {Zhuang, Ren and Wang, Ben and Sun, Shuifa},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
