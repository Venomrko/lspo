# Self-Correction as Transition Geometry: Internalizing Reasoning via Lifted State Policy Optimization

**NeurIPS 2026** · Ren Zhuang, Ben Wang, Shuifa Sun

Code for **Lifted State Policy Optimization (LSPO)**.

## Overview

LSPO trains language models to internalize answer correction. Each answer is paired with a continuous auxiliary coordinate that carries refinement context. A controller learns local revisions from energy decreases, balanced against edit and step costs. Selected answers then become training targets for the generator. At deployment, the model produces answers through direct generation.

![Qwen3.5-27B benchmark results, rollout-to-direct alignment, and energy separation](assets/figure1.png)

**Figure 1.** Benchmark performance on Qwen3.5-27B, alignment between rollout-selected answers and direct generation, and final-energy distributions for correct and incorrect answers.

![Comparison of answer-level optimization, trajectory RL, and LSPO](assets/method.png)

**Method overview.** LSPO assigns credit to transitions in the lifted state space and internalizes the selected answers.

The paper evaluates Qwen3.5-9B and Qwen3.5-27B dense models, together with the Qwen3.5-35B-A3B mixture-of-experts model. The following results are reported in the paper across MATH-500, AIME25, GPQA, LiveCodeBench, ZebraLogic, and MMLU-Pro. Scores are percentages; each cell gives **pass@1 / pass@16**.

| Method | Qwen3.5-9B | Qwen3.5-27B | Qwen3.5-35B-A3B |
| --- | ---: | ---: | ---: |
| Base | 54.8 / 67.2 | 65.9 / 76.2 | 67.8 / 77.7 |
| DPO | 55.8 / 68.1 | 66.9 / 77.1 | 68.8 / 78.7 |
| Step-DPO | 56.9 / 69.5 | 68.0 / 78.5 | 70.0 / 80.0 |
| GRPO | 58.9 / 71.9 | 70.0 / 80.9 | 71.6 / 82.0 |
| GDPO | 59.6 / 72.7 | 70.8 / 81.7 | 72.3 / 82.9 |
| SCoRe | 59.4 / 72.8 | 70.7 / 82.0 | 71.9 / 82.9 |
| **LSPO** | **60.8 / 73.9** | **71.8 / 82.7** | **72.6 / 83.1** |

![Quality, sampling, correction transitions, and output cost on Qwen3.5-27B](assets/results.png)

**Quality and cost on Qwen3.5-27B.** LSPO achieves an aggregate pass@1 score of 71.8%, with an average latency of 10.93 seconds and 1,265 output tokens in the paper's evaluation.

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
