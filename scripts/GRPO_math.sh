SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

python -c "import open_r1; print('open_r1 has been added to python path', open_r1.__file__)"

# Example: MATH training with MC-GRPO
# Adjust CUDA_VISIBLE_DEVICES, num_processes, num_generations, and mc as needed.
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 src/open_r1/grpo_math.py \
    --config recipes/math/Qwen2.5-Math-1.5B.yaml \
    --output_dir=results/math \
    --save_strategy='best' \
    --eval_steps=100 --max_completion_length=3072 \
    --dataset_name=DigitalLearningGmbH/MATH-lighteval \
    --num_generations=8 \
    --run_name='MC-GRPO-math-Qwen2.5-Math-1.5B' \
    --loss_type grpo \
    --use_vllm=True \
    --mc=True
