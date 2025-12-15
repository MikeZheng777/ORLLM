# Quick Start Guide - Running Experiments

This guide will walk you through running the SFT experiments step by step.

## Prerequisites

### 1. Prepare Evaluation Datasets

First, create the evaluation datasets:

```bash
python prepare_eval_datasets.py
```

This creates 3 files in `train_test_data/eval_datasets/`:
- `industryor_eval_30.json`
- `mamo_easy_eval_30.json`
- `optmath_eval_30.json`

### 2. Verify Training Data

Make sure you have training data files:
- `train_test_data/converted_data_messages_new_100.json`
- `train_test_data/converted_data_messages_new_200.json` (for data_size experiments)
- `train_test_data/converted_data_messages_new_300.json` (for data_size experiments)
- `train_test_data/converted_data_messages_new_400.json` (for data_size experiments)

### 3. Configure Paths (Optional)

Edit `run_experiments.sh` if needed:
- `MODEL_NAME_OR_PATH`: Path to your base model
- `BASE_SAVE_PATH`: Where to save trained models
- `NUM_GPUS`: Number of GPUs to use
- `NUM_TRAIN_EPOCHS` or `MAX_TRAIN_STEPS`: Training duration

## Running Experiments

### Option 1: Run All Experiments (Not Recommended Initially)

```bash
./run_experiments.sh all
```


### Option 2: Run Individual Experiment Groups (Recommended)

#### A. LR Schedule Experiments

Tests different learning rate schedules:

```bash
./run_experiments.sh lr_schedule
```

Runs 3 experiments:
- Linear LR schedule
- Exponential LR schedule  
- Constant LR (no schedule)


#### B. LoRA Rank Experiments

Tests different LoRA ranks:

```bash
./run_experiments.sh lora
```

Runs 6 experiments:
- Full parameter tuning (no LoRA)
- LoRA rank 1
- LoRA rank 4
- LoRA rank 8
- LoRA rank 16
- LoRA rank 32

**Expected time**: ~6-12 hours

#### C. Data Size Experiments

Tests different training data sizes:

```bash
./run_experiments.sh data_size
```

Runs 4 experiments:
- 100 examples
- 200 examples
- 300 examples
- 400 examples

**Expected time**: ~4-8 hours

## What Happens During Training

1. **Model Loading**: Loads the base model (Qwen3-8B)
2. **Data Loading**: Loads training and evaluation datasets
3. **Training**: Trains the model for specified epochs/steps
4. **Evaluation**: Every 50 steps, evaluates on 3 datasets:
   - Generates code for each evaluation example
   - Executes the code
   - Reports metrics for each dataset separately
5. **Checkpointing**: Saves model checkpoints every 100 steps
6. **Logging**: Logs all metrics to WandB

## Monitoring Progress

### WandB Dashboard

All experiments automatically log to WandB. You can:

1. **View in browser**: Go to https://wandb.ai
2. **Find your project**: `orlm_sft_experiments`
3. **Monitor metrics**:
   - Training loss
   - Evaluation metrics for each dataset:
     - `industryor_execution_success_rate`
     - `industryor_mathematical_accuracy`
     - `mamo_easy_execution_success_rate`
     - `mamo_easy_mathematical_accuracy`
     - `optmath_execution_success_rate`
     - `optmath_mathematical_accuracy`

### Console Output

The script prints:
- Experiment start/end messages
- Evaluation results after each evaluation step
- Progress indicators

### Log Files

Checkpoints and logs are saved in:
```
{BASE_SAVE_PATH}/{experiment_name}/
```

For example:
- `/hanzheng/orcd/scratch/orlm_finetune/experiments/lr_schedule_linear/`
- `/hanzheng/orcd/scratch/orlm_finetune/experiments/lora_rank_16/`

## Example Output

When you run an experiment, you'll see:

```
🚀 Starting experiment: lr_schedule_linear
===============================================
Data: train_test_data/converted_data_messages_new_100.json
LoRA: false (rank: 16)
LR Scheduler: linear
Output: /hanzheng/orcd/scratch/orlm_finetune/experiments/lr_schedule_linear
Training for 3 epochs
===============================================

[Training logs...]

Running code generation evaluation at step 50
Evaluating industryor: 100%|████████| 30/30
industryor - Execution Success: 0.733, Mathematical Accuracy: 0.667
Evaluating mamo_easy: 100%|████████| 30/30
mamo_easy - Execution Success: 0.800, Mathematical Accuracy: 0.733
Evaluating optmath: 100%|████████| 30/30
optmath - Execution Success: 0.700, Mathematical Accuracy: 0.633

✅ Completed experiment: lr_schedule_linear
```

## Troubleshooting

### Error: Evaluation datasets not found

```bash
python prepare_eval_datasets.py
```

### Error: Training data not found

Create missing data files using `convert_data_for_finetuning.py`

### Out of Memory

Reduce in `run_experiments.sh`:
- `BATCH_SIZE_PER_GPU` (e.g., from 4 to 2)
- `MAX_SEQ_LENGTH` (e.g., from 8192 to 4096)

### WandB Not Logging

Make sure WandB is installed and logged in:
```bash
pip install wandb
wandb login
```

## Recommended Workflow

1. **Start small**: Run one experiment group first (e.g., `lr_schedule`)
2. **Monitor**: Check WandB dashboard to see if training is progressing
3. **Verify**: Check that evaluation metrics are being logged
4. **Scale up**: Once confident, run other experiment groups
5. **Analyze**: Compare results across different configurations

## Next Steps

After experiments complete:
1. Analyze results in WandB
2. Compare metrics across configurations
3. Select best configuration
4. Optionally run additional experiments with refined hyperparameters

