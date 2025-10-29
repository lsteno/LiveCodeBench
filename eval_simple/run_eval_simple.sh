#!/bin/bash
# Minimal LiveCodeBench evaluation helper.
# Usage:
#   sbatch eval_simple/run_eval_simple.sh \
#     --model Qwen2.5-1.5B-QLoRA \
#     --local-path ~/GSD-finetune/qlora_simple/runs/qwen2.5-1.5b-qlora-merged

set -euo pipefail

MODEL=""
LOCAL_PATH=""
RELEASE="v6"
SCENARIO="codegeneration"
N=10
TEMPERATURE=0.2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --local-path) LOCAL_PATH="$2"; shift 2 ;;
    --release) RELEASE="$2"; shift 2 ;;
    --scenario) SCENARIO="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --model NAME --local-path DIR [--release v6] [--scenario codegeneration] [--n 10] [--temperature 0.2]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$LOCAL_PATH" ]]; then
  echo "Both --model and --local-path are required." >&2
  exit 1
fi

# Offline by default (override HF_HOME/HF_DATASETS_CACHE before calling if desired)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

source ~/LiveCodeBench/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log and error filenames with timestamp and model name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/${MODEL}_${SCENARIO}_${RELEASE}_${TIMESTAMP}.log"
ERR_FILE="logs/${MODEL}_${SCENARIO}_${RELEASE}_${TIMESTAMP}.err"

echo "Logging to: $LOG_FILE"
echo "Errors to: $ERR_FILE"

# Run python with stdout to log file and stderr to err file
# Using explicit redirection that works in Slurm non-interactive environments
python -m lcb_runner.runner.main \
  --model "$MODEL" \
  --local_model_path "$LOCAL_PATH" \
  --scenario "$SCENARIO" \
  --evaluate \
  --release_version "$RELEASE" \
  --n "$N" \
  --temperature "$TEMPERATURE" \
  2>&1 | tee "$LOG_FILE"
