# Running LiveCodeBench Evaluation on Your Finetuned Qwen 2.5 7B

## Prerequisites

1. **Merged model** at: `~/GSD-finetune/lora/qwen2.5-7b-instruct-merged`
2. **Virtual environment** set up in this directory (`.venv/`)
3. **SLURM cluster** access with the `students` partition

## Setup Steps

### 1. Register The Model (Done)

The finetuned model has been registered in `lcb_runner/lm_styles.py` as:
- **Model Name**: `Qwen2.5-7B-Finetuned`
- **Model Repr**: `Qwen2.5-7B-FT` 
- **Style**: `CodeQwenInstruct` (uses Qwen chat template)

### 2. Prepare the Environment on the Cluster

On the login node (head node with internet access):

```bash
# Navigate to LiveCodeBench directory
cd ~/LiveCodeBench

# Create virtual environment matching the cluster's Python version
# Check available versions with: module avail python
module load python/3.10.7
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Create logs directory
mkdir -p logs
```

### 3. Pre-download the Dataset (IMPORTANT - Must do on head node)

Since compute nodes don't have internet access, download the dataset first:

```bash
# Make sure you're on the head node with internet and venv activated
source .venv/bin/activate

# Download the benchmark dataset
python3 -c "
from datasets import load_dataset
print('Downloading LiveCodeBench dataset...')
dataset = load_dataset('livecodebench/code_generation_lite', split='test', version_tag='v5_v6')
print(f'Downloaded {len(dataset)} problems successfully!')
"
```

This will cache the dataset in `~/.cache/huggingface/datasets/` so it's available offline on compute nodes.

### 4. Submit the Evaluation Job

```bash
# Make the script executable
chmod +x run_evaluation.sh

# Submit to SLURM
sbatch run_evaluation.sh
```

### 5. Monitor the Job

```bash
# Check job status
squeue -u $USER

# Watch the output log in real-time
tail -f logs/eval_<JOB_ID>.out

# Check for errors
tail -f logs/eval_<JOB_ID>.err
```

## Understanding the Evaluation Script

The `run_evaluation.sh` script runs with these parameters:

- **Model**: Your finetuned Qwen 2.5 7B from `~/GSD-finetune/lora/qwen2.5-7b-instruct-merged`
- **Scenario**: `codegeneration` (generates and evaluates Python code)
- **Dataset**: `release_v5` (880 problems from May 2023 to Jan 2025)
- **Samples per problem**: 10 (`--n 10`)
- **Temperature**: 0.2 (for diverse but focused generation)
- **GPU**: 1 GPU with tensor parallelism disabled
- **Evaluation**: Automated with 12 parallel processes
- **Timeout**: 10 seconds per test case
- **Caching**: Enabled to save time on repeated runs

## Expected Runtime

- **Generation**: ~4-8 hours for 880 problems × 10 samples (depends on model speed)
- **Evaluation**: ~2-4 hours for running all test cases
- **Total**: Expect 6-12 hours for full run (48 hour limit is conservative)

## Output Files

Results will be saved in `output/Qwen2.5-7B-FT/`:

1. **`codegeneration_10_0.2.json`**
   - Raw generated code for each problem
   - Contains all 10 samples per problem

2. **`codegeneration_10_0.2_eval.json`**
   - Summary metrics (pass@1, pass@5, pass@10)
   - Overall performance statistics

3. **`codegeneration_10_0.2_eval_all.json`**
   - Detailed results for each problem
   - Shows which test cases passed/failed
   - Useful for debugging and analysis

## Analyzing Results

After the job completes, you can analyze results by time window:

```bash
# Get scores for all problems
python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json

# Get scores for problems after Sep 2023 (to avoid contamination)
python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json \
    --start_date 2023-09-01

# Get scores for recent problems only (2024 onwards)
python -m lcb_runner.evaluation.compute_scores \
    --eval_all_file output/Qwen2.5-7B-FT/codegeneration_10_0.2_eval_all.json \
    --start_date 2024-01-01
```

## Troubleshooting

### Python version mismatch error:

If you see `AssertionError: SRE module mismatch`, your venv Python version doesn't match the loaded module.

**Solution**: Make sure the Python version in the SLURM script matches your venv:
```bash
# Check your venv Python version
source .venv/bin/activate
python --version  # e.g., Python 3.10.7

# Make sure run_evaluation.sh loads the same version:
# module load python/3.10.7  (line 19 in run_evaluation.sh)
```

### Dataset not found error (offline nodes):

If you see `ConnectionError: Couldn't reach 'livecodebench/code_generation_lite' on the Hub (OfflineModeIsEnabled)`:

**Solution**: Pre-download the dataset on the head node (Step 3 above) before submitting the job.

### If the job fails due to memory:

Reduce batch size or use gradient checkpointing by editing the script to add:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### If you need to continue from a partial run:

Edit `run_evaluation.sh` and add the `--continue_existing` flag:
```bash
python -m lcb_runner.runner.main \
    --model Qwen2.5-7B-Finetuned \
    --local_model_path ~/GSD-finetune/lora/qwen2.5-7b-instruct-merged \
    --scenario codegeneration \
    --evaluate \
    --continue_existing \
    ... # rest of flags
```

### If evaluation is too slow:

Adjust these parameters in the script:
- `--num_process_evaluate 8` (reduce from 12)
- `--timeout 15` (increase if tests are timing out unfairly)

### If model loading fails:

Check that:
1. Model path is correct: `~/GSD-finetune/lora/qwen2.5-7b-instruct-merged`
2. Model files include: `config.json`, `pytorch_model.bin` or `.safetensors`, `tokenizer.json`
3. You have proper permissions to access the model directory

## Running Other Scenarios

Once code generation works, you can try:

### Self-Repair
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

- The evaluation uses **VLLM** for efficient inference
- Results are automatically saved and can be resumed with `--continue_existing`
- The benchmark uses **test cases pruned** for faster evaluation (use `--not_fast` for full tests)
- All 880 problems will be evaluated unless you use `--debug` (which runs only 15)
