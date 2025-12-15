#!/bin/bash

# Base directory for experiment models
BASE_DIR="/home/hanzheng/orcd/scratch/orlm_finetune/experiments"

# List of all experiments to evaluate
experiments=(
    # "data_size_100"
    # "data_size_200"
    # "data_size_300"
    # "Qwen3-8B"
    # "data_size_400"
    "lora_rank_4"
    "lora_rank_8"
    "lora_rank_16"
    "lora_rank_32"
    "lora_rank_64"
    # "lr_schedule_constant"
    # "lr_schedule_cosine"
    # "lr_schedule_linear"
)

# Log file for tracking progress
LOG_FILE="eval_all_experiments.log"
echo "Starting evaluation for all experiments at $(date)" | tee "$LOG_FILE"

# Counter for tracking progress
total=${#experiments[@]}
current=0

for exp in "${experiments[@]}"; do
    current=$((current + 1))
    MODEL_PATH="${BASE_DIR}/${exp}"
    
    echo "" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
    echo "[$current/$total] Evaluating experiment: $exp" | tee -a "$LOG_FILE"
    echo "Model path: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "===============================================" | tee -a "$LOG_FILE"
    
    # Check if model directory exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  WARNING: Model directory not found: $MODEL_PATH" | tee -a "$LOG_FILE"
        echo "Skipping $exp..." | tee -a "$LOG_FILE"
        continue
    fi
    
    # Run evaluation
    if ./run_eval.sh "$MODEL_PATH" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✅ Successfully completed evaluation for $exp" | tee -a "$LOG_FILE"
    else
        echo "❌ ERROR: Evaluation failed for $exp" | tee -a "$LOG_FILE"
    fi
    
    echo "Completed at: $(date)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "===============================================" | tee -a "$LOG_FILE"
echo "All evaluations completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved in: results/" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

