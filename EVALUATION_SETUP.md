# Running LiveCodeBench Evaluation on Finetuned Qwen 2.5 7B

## Prerequisites

1. **Merged model** at: `~/GSD-finetune/lora/qwen2.5-7b-instruct-merged`
2. **SLURM cluster** access with the `students` partition
3. **Internet access** on the head/login node for setup

## Setup Steps

### 1. Model Registration

The finetuned model has been registered in `lcb_runner/lm_styles.py` as:
- **Model Name**: `Qwen2.5-7B-Finetuned`
- **Model Repr**: `Qwen2.5-7B-FT` 
- **Style**: `CodeQwenInstruct` (uses Qwen chat template)

### 2. Environment Setup

On the login node (which has internet access):

```bash
# Navigate to LiveCodeBench directory
cd ~/LiveCodeBench

# Load Python module
module load python/3.10.7

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Create logs directory
mkdir -p logs
```

### 3. Submit the Evaluation Job

```bash
sbatch run_evaluation.sh
```

### 4. Monitor the Job

```bash
# Check job status
squeue -u $USER

# Watch the output log in real-time
tail -f logs/eval_<JOB_ID>.out

# Check for errors
tail -f logs/eval_<JOB_ID>.err
```

## Evaluation Configuration

The `run_evaluation.sh` script is configured with:

- **Model**: `Qwen2.5-7B-Finetuned` (from `~/GSD-finetune/lora/qwen2.5-7b-instruct-merged`)
- **Scenario**: `codegeneration` (code generation and evaluation)
- **Dataset Version**: `v5_v6` (includes v5 and v6 problems only)
- **Samples per problem**: 10 (`--n 10`)
- **Temperature**: 0.2
- **Precision**: `bfloat16`
- **Resources**: 1 GPU, 16 CPUs, 64GB RAM
- **Time Limit**: 2 hours
- **Evaluation**: 12 parallel processes, 10s timeout per test case
- **Caching**: Enabled for faster reruns

## Expected Runtime

- **Generation + Evaluation**: 1.5-2 hours (actual runtime depends on model speed)
- The 2-hour time limit is sufficient for the configured workload

## Output Files

Results will be saved in `output/Qwen2.5-7B-FT/`:

1. **`codegeneration_10_0.2.json`** - Raw generated code (all 10 samples per problem)
2. **`codegeneration_10_0.2_eval.json`** - Summary metrics (pass@1, pass@5, pass@10)
3. **`codegeneration_10_0.2_eval_all.json`** - Detailed per-problem results

## Analyzing Results

After the job completes, analyze results by time window:

```bash
# Activate environment
source .venv/bin/activate

# All problems
python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json

# Problems after Sep 2023 (avoid potential contamination)
python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json \
    --start_date 2023-09-01

# Only 2024+ problems
python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json \
    --start_date 2024-01-01
```

## Troubleshooting

### Python version mismatch

**Error**: `AssertionError: SRE module mismatch`

**Solution**: Ensure the Python module in `run_evaluation.sh` matches your venv:
```bash
# Check your venv Python version
source .venv/bin/activate
python --version  # Should be 3.10.7
```

### Continuing from partial run

Add `--continue_existing` flag to reuse existing completions:
```bash
# In run_evaluation.sh, add to the python command:
--continue_existing
```

### Memory issues

Add memory configuration before the python command:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Slow evaluation

Adjust these parameters in `run_evaluation.sh`:
- `--num_process_evaluate 8` (reduce from 12)
- `--timeout 15` (increase timeout if tests are failing unfairly)

### Model loading fails

Verify:
1. Model path exists: `~/GSD-finetune/lora/qwen2.5-7b-instruct-merged`
2. Contains: `config.json`, model weights (`.safetensors` or `.bin`), tokenizer files
3. You have read permissions

## Additional Scenarios

After code generation evaluation, you can test other scenarios:

### Self-Repair
Requires existing code generation results (`--codegen_n` must match original run):
```bash
python -m lcb_runner.runner.main \
    --model Qwen2.5-7B-Finetuned \
    --local_model_path ~/GSD-finetune/lora/qwen2.5-7b-instruct-merged \
    --scenario selfrepair \
    --codegen_n 10 \
    --n 1 \
    --evaluate
```

### Test Output Prediction
```bash
python -m lcb_runner.runner.main \
    --model Qwen2.5-7B-Finetuned \
    --local_model_path ~/GSD-finetune/lora/qwen2.5-7b-instruct-merged \
    --scenario testoutputprediction \
    --evaluate
```

### Code Execution
```bash
python -m lcb_runner.runner.main \
    --model Qwen2.5-7B-Finetuned \
    --local_model_path ~/GSD-finetune/lora/qwen2.5-7b-instruct-merged \
    --scenario codeexecution \
    --evaluate
```

## Notes

- Uses **VLLM** for efficient inference
- Results auto-save; resume with `--continue_existing`
- Uses **pruned test cases** by default (faster). Use `--not_fast` for full test suite
- `--debug` flag runs only 15 problems for quick testing
