# MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning

PyTorch implementation of **MC-GRPO** from [MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning](https://arxiv.org/abs/2601.22582).

## Overview

Group-relative policy optimization methods (GRPO) train language models by generating multiple rollouts per prompt and normalizing rewards with a shared mean baseline. Under small rollout budgets, the mean baseline becomes noisy and induces **advantage sign flips** — reversing the update direction for some rollouts.

**MC-GRPO** addresses this by:
1. Replacing the **mean** baseline with a **median** baseline, which is far less sensitive to outlier rewards.
2. Sampling **G+1** rollouts (one extra) to form an odd-sized group, so exactly one completion is the median.
3. **Excluding the median rollout** (zero advantage) from backpropagation, preserving the effective update cost of G rollouts.
4. Using **MAD** (Median Absolute Deviation) instead of standard deviation for robust normalization.

MC-GRPO is algorithm-agnostic within the GRPO family: the same median-centered baseline yields consistent improvements across **GRPO**, **DAPO**, and **DR-GRPO**.

## Project Structure

```
MC-GRPO/
├── src/
│   └── open_r1/
│       ├── grpo_gsm.py              # GSM8K training entry point
│       ├── grpo_math.py             # MATH training entry point
│       ├── grpo_trainer_gsm.py      # MC-GRPO trainer for GSM8K
│       ├── grpo_trainer_math.py     # MC-GRPO trainer for MATH
│       ├── rewards_gsm.py           # Reward functions for GSM8K
│       ├── rewards_math.py          # Reward functions for MATH
│       ├── configs.py               # Training configurations
│       ├── eval_math.py             # MATH evaluation script
│       └── utils/
├── scripts/
│   ├── GRPO_gsm.sh                  # GSM8K training launch script
│   └── GRPO_math.sh                 # MATH training launch script
├── recipes/
│   ├── accelerate_configs/          # DeepSpeed / DDP configs
│   ├── gsm8k/                       # GSM8K model-specific configs
│   └── math/                        # MATH model-specific configs
└── setup_env.sh                     # Environment setup script
```

## Setup

### Environment

```bash
bash setup_env.sh
conda activate mc_grpo_env
```


### Python Path

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python -c "import open_r1; print('open_r1 loaded successfully')"
```

## Training

### GSM8K

```bash
# MC-GRPO on GSM8K with Qwen3-1.7B
bash scripts/GRPO_gsm.sh
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=2 src/open_r1/grpo_gsm.py \
    --config recipes/gsm8k/Qwen3-1.7B.yaml \
    --output_dir=results/gsm8k \
    --num_generations=2 \
    --loss_type grpo \
    --use_vllm=True \
    --mc=True
```

### MATH

```bash
# MC-GRPO on MATH with Qwen2.5-Math-1.5B
bash scripts/GRPO_math.sh
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 src/open_r1/grpo_math.py \
    --config recipes/math/Qwen2.5-Math-1.5B.yaml \
    --output_dir=results/math \
    --num_generations=2 \
    --loss_type grpo \
    --use_vllm=True \
    --mc=True
```

### Key Arguments

| Argument | Description |
|---|---|
| `--mc=True` | Enable **Median centered (ours)** — median baseline and MAD normalization (MC-GRPO). Set `False` for standard GRPO (mean baseline). |
| `--num_generations` | Number of rollouts per prompt (G). MC-GRPO internally generates G+1. |
| `--loss_type` | Loss formulation: `grpo`, `dapo`, or `dr_grpo`. |
| `--use_vllm=True` | Use vLLM for fast rollout generation. |
| `--config` | Path to model-specific YAML config. |

### Available Model Configs

**GSM8K** (`recipes/gsm8k/`):
- `Qwen3-1.7B.yaml`
- `Qwen2.5-1.5B-Instruct.yaml`
- `Llama-3.2-3B-instruct.yaml`

**MATH** (`recipes/math/`):
- `Qwen2.5-Math-1.5B.yaml`
- `Qwen3-4B-instruct.yaml`
- `Qwen2.5-7B-Instruct.yaml`
- `Llama-3.2-3B-instruct.yaml`

## Citation

```bibtex
@article{kim2026mcgrpo,
    title={MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning},
    author={Youngeun Kim},
    year={2026},
    eprint={2601.22582},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Acknowledgements

This codebase builds on [TRL](https://github.com/huggingface/trl) and [Open-R1](https://github.com/huggingface/open-r1).
