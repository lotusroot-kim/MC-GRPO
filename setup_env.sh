#!/bin/bash
set -e

##############################################
# DeepResearch Environment Setup Script
#
# Sets up a conda environment for RL training
##############################################

ENV_NAME="mc_grpo_env"
PYTHON_VERSION="3.12"

echo "========================================================="
echo " 🔧 Starting DeepResearch environment setup"
echo "========================================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed."
    echo "   Please install Miniconda or Anaconda first."
    exit 1
fi

echo "📦 Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"

# Remove environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "   Existing environment found. Removing and recreating it..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo ""
echo "📥 Installing packages..."

# Note: 'conda activate' may not work in non-interactive bash scripts,
# so we use 'conda run' to run commands inside the environment.
conda run -n $ENV_NAME pip install vllm==0.11.2
# Flash-attn: try prebuilt wheel first; fallback to build from source
if ! conda run -n $ENV_NAME pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" 2>/dev/null; then
  echo "   Prebuilt flash-attn wheel not compatible; building from source..."
  conda run -n $ENV_NAME pip install flash-attn --no-build-isolation
fi
conda run -n $ENV_NAME pip install trl==0.26.1
conda run -n $ENV_NAME pip install peft
conda run -n $ENV_NAME pip install liger-kernel
conda run -n $ENV_NAME pip install wandb
conda run -n $ENV_NAME pip3 install deepspeed
conda run -n $ENV_NAME pip install bitsandbytes
conda run -n $ENV_NAME pip install math_verify==0.8.0
conda run -n $ENV_NAME pip install "latex2sympy2-extended[antlr4-13-2]"

echo ""
echo "========================================================="
echo " ✅ Environment setup complete!"
echo "========================================================="
echo ""
echo "Activate the environment with:"
echo "  conda activate $ENV_NAME"
echo ""
