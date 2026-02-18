SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

python -c "import open_r1; print('open_r1 has been added to python path', open_r1.__file__)"

# Example: GSM8K training with MC-GRPO
# Adjust CUDA_VISIBLE_DEVICES, num_processes, num_generations, and mc as needed.
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=2 src/open_r1/grpo_gsm.py \
    --config recipes/gsm8k/Qwen3-1.7B.yaml \
    --output_dir=results/gsm8k \
    --save_strategy='best' \
    --dataset_name=openai/gsm8k \
    --num_generations=8 \
    --run_name='MC-GRPO-gsm8k-Qwen3-1.7B' \
    --loss_type grpo \
    --use_vllm=True \
    --mc=True