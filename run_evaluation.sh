#!/bin/bash
#SBATCH --job-name=lcb-eval-qwen2.5-7b
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=students
#SBATCH --time=02:00:00
#SBATCH --mem=64GB

# IMPORTANT: Before submitting this job, make sure you've pre-downloaded the dataset
# on the head node (which has internet access) by running:
#   source .venv/bin/activate
#   python3 -c "from datasets import load_dataset; load_dataset('livecodebench/code_generation_lite', split='test', version_tag='release_v5')"
# This caches the dataset so offline compute nodes can access it.

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load required modules
echo "Loading Python and CUDA modules..."
module load python/3.10.7
module load nvidia/cuda-11.8

# Activate virtual environment using absolute path
VENV_PATH="$(pwd)/.venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtual environment at: $VENV_PATH"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create the virtual environment first"
    exit 1
fi

# Verify Python is available
which python
python --version

# Print GPU information
echo "GPU Information:"
nvidia-smi

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Set HuggingFace cache directory (datasets will be cached here)
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p logs

# Run LiveCodeBench evaluation
echo "Starting LiveCodeBench evaluation..."
echo "Model path: ~/GSD-finetune/lora/qwen2.5-7b-instruct-merged"
echo "Scenario: codegeneration"
echo "Release version: release_v5"

python -m lcb_runner.runner.main \
    --model Qwen2.5-7B-Finetuned \
    --local_model_path ~/GSD-finetune/lora/qwen2.5-7b-instruct-merged \
    --scenario codegeneration \
    --evaluate \
    --release_version release_v5 \
    --n 10 \
    --temperature 0.2 \
    --tensor_parallel_size 1 \
    --use_cache \
    --num_process_evaluate 12 \
    --timeout 10 \
    --dtype bfloat16

echo "Evaluation completed at: $(date)"

# Print location of results
echo ""
echo "Results saved in:"
echo "  - output/Qwen2.5-7B-FT/codegeneration_10_0.2.json"
echo "  - output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval.json"
echo "  - output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json"
