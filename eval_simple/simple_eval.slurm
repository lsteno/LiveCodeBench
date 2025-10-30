#!/bin/bash
#SBATCH --job-name=lcb-simple
#SBATCH --output=logs/simple_eval_%j.out
#SBATCH --error=logs/simple_eval_%j.err
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00

module load python/3.10.7

cd ~/LiveCodeBench
source ~/LiveCodeBench/.venv/bin/activate

mkdir -p logs

bash eval_simple/run_eval_simple.sh "$@"
