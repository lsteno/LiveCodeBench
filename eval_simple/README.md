# LiveCodeBench Eval Simple

A trimmed-down entry point for running LiveCodeBench on locally fine-tuned models without touching the full HPC scripts

## Files

```
eval_simple/
├── README.md
├── run_eval_simple.sh   # direct CLI helper
└── simple_eval.slurm    # optional SLURM wrapper
```

## 1. Environment (one-time)

```bash
module load python/3.10.7
cd ~/LiveCodeBench
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install 'datasets<3.2.0'

# Cache evaluation dataset once (needs internet)
python -c "from datasets import load_dataset; load_dataset('livecodebench/code_generation_lite', split='test', version_tag='v6', trust_remote_code=True)"
```

## 2. Run evaluation (interactive)

```bash
cd ~/LiveCodeBench
source .venv/bin/activate
bash eval_simple/run_eval_simple.sh \
  --model Qwen2.5-3B-Finetuned \
  --local-path ~/GSD-finetune/qlora_simple/runs/qwen2.5-1.5b-qlora-5k-merged
```

To run multiple repetitions for statistical analysis:

```bash
bash eval_simple/run_eval_simple.sh \
  --model Qwen2.5-3B-Finetuned \
  --local-path ~/GSD-finetune/qlora_simple/runs/qwen2.5-1.5b-qlora-5k-merged \
  --repetitions 5
```

The script passes through `--release v6`, `--n 10`, `--temperature 0.2`, `--repetitions 1`, but you can override them with extra flags (see file header).

## 3. Run evaluation (SLURM)

```bash
cd ~/LiveCodeBench
sbatch eval_simple/simple_eval.slurm \
  --model Qwen/Qwen2.5-7B-Instruct\
  --local-path ~/GSD-finetune/bitfit_simple/runs/1.5b-bitfit-5k-merged \
  --repetitions 5
```

To run multiple repetitions for statistical analysis:

```bash
sbatch eval_simple/simple_eval.slurm \
  --model Qwen/Qwen2.5-7B-Instruct\
  --local-path ~/GSD-finetune/lora_simple/runs/1.5b-lora-50k-r16-merged \
  --repetitions 5
```

The job loads Python, activates `.venv`, and executes the same `run_eval_simple.sh`. Logs live in `logs/simple_eval_<JOB_ID>.(out|err)`. Each repetition creates separate log files with `_rep{N}` suffix for easy tracking.

## Output & Analysis

Evaluation artefacts land under `output/<model_repr>/Scenario.codegeneration_10_0.2*.json`. 

When running multiple repetitions, results are organized in a batch folder:
```
output/<model_repr>_batch_<timestamp>/rep1/
output/<model_repr>_batch_<timestamp>/rep2/
...
```

Use the standard tooling to analyze results, e.g.:

```bash
# Single run
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file output/Qwen2.5-Ins-7B/Scenario.codegeneration_10_0.2_eval_all.json

# Multiple repetitions - analyze each repetition
python -m lcb_runner.evaluation.compute_scores \
  --eval_all_file output/Qwen2.5-Ins-7B_batch_20251030_143022/rep1/Scenario.codegeneration_10_0.2_eval_all.json
```

## Notes

- The helper just wraps `python -m lcb_runner.runner.main`; feel free to add flags (e.g. `--continue_existing`) inside the shell script.
- Keep datasets cached on the head node; compute nodes run offline.
- Works for any model registered in `lcb_runner/lm_styles.py`.
