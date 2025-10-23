#!/bin/bash
#SBATCH --job-name=lcb-simple
#SBATCH --output=logs/simple_eval_%j.out
#SBATCH --error=logs/simple_eval_%j.err
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=02:00:00

module load python/3.10.7

cd ~/LiveCodeBench
source .venv/bin/activate

bash eval_simple/run_eval_simple.sh "$@"
