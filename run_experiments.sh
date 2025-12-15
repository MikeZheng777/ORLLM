#!/bin/bash

# Comprehensive experiment runner for SFT experiments
# This script runs experiments for:
# 1. LR schedule experiments (linear, cosine, constant)
# 2. LoRA rank experiments (full param, 1, 4, 8, 16, 32)
# 3. Data size experiments (100, 200, 300, 400)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model and Base Paths
MODEL_NAME_OR_PATH="/home/hanzheng/orcd/scratch/models/Qwen3-8B"
BASE_SAVE_PATH="/home/hanzheng/orcd/scratch/orlm_finetune/experiments"

# Evaluation Datasets (3 separate datasets from baseline_test_data)
# Each dataset will be sampled to 10 fixed samples during training (via max_eval_samples=10)
EVAL_DATASETS_DIR="baseline_test_data"
INDUSTRYOR_EVAL="${EVAL_DATASETS_DIR}/industryor_test.json"
MAMO_COMPLEX_EVAL="${EVAL_DATASETS_DIR}/mamo_complex_test.json"
NLP4LP_EVAL="${EVAL_DATASETS_DIR}/nlp4lp_test.json"

# Combined eval datasets string for training script
EVAL_DATASETS_STRING="industryor:${INDUSTRYOR_EVAL},mamo_complex:${MAMO_COMPLEX_EVAL},nlp4lp:${NLP4LP_EVAL}"

# Training Configuration
NUM_GPUS=2
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=16
PREPROCESSING_NUM_WORKERS=0
MAX_SEQ_LENGTH=8192
LEARNING_RATE=2e-5
WARMUP_RATIO=0.03

# Training Duration - Choose ONE of the following:
# Option 1: Train for a specific number of epochs
NUM_TRAIN_EPOCHS=4
# Option 2: Train for a specific number of steps (uncomment to use instead of epochs)
# MAX_TRAIN_STEPS=1000
# Note: If MAX_TRAIN_STEPS is set, it will override NUM_TRAIN_EPOCHS

# Evaluation Configuration
# Disable evaluation during training - we'll evaluate final models separately
EVAL_STRATEGY="no"  # No evaluation during training
EVAL_STEPS=10000  # Not used (evaluation disabled)
SAVE_STRATEGY="no"  # Don't save intermediate checkpoints during training
SAVE_STEPS=10000  # Not used (save_strategy is "no")
SAVE_TOTAL_LIMIT=1  # Not used (save_strategy is "no")

# WandB Configuration
WANDB_PROJECT="orlm_sft_experiments"
WANDB_ENTITY=""  # Set your wandb entity if needed
WANDB_API_KEY="838fc2c3fd1b86329f9a0789a1508e44ec3abfda"  # Optional: Set API key here or use 'wandb login' command

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

run_training() {
    local EXP_NAME=$1
    local DATA_PATH=$2
    local USE_LORA=$3
    local LORA_RANK=$4
    local LR_SCHEDULER=$5
    local SAVE_DIR="${BASE_SAVE_PATH}/${EXP_NAME}"
    
    echo "🚀 Starting experiment: $EXP_NAME"
    echo "==============================================="
    echo "Data: $DATA_PATH"
    echo "LoRA: $USE_LORA (rank: $LORA_RANK)"
    echo "LR Scheduler: $LR_SCHEDULER"
    echo "Output: $SAVE_DIR"
    echo "==============================================="
    
    # Calculate gradient accumulation steps
    GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
    
    # Set CUDA devices
    export CUDA_VISIBLE_DEVICES=0,1
    
    # Set WandB configuration via environment variables
    export WANDB_RUN_NAME="$EXP_NAME"
    export WANDB_PROJECT="$WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        export WANDB_ENTITY="$WANDB_ENTITY"
    fi
    # Ensure WandB uses the project name from environment
    export WANDB_PROJECT_NAME="$WANDB_PROJECT"
    
    # Set WandB API key if provided
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export WANDB_API_KEY="$WANDB_API_KEY"
    fi
    
    # Don't save intermediate checkpoints - only save final model at end of training
    # The final model will be saved by trainer.save_model() in finetune.py
    CURRENT_SAVE_STRATEGY="no"  # Don't save during training
    CURRENT_SAVE_STEPS=10000  # Not used when strategy is "no"
    CURRENT_SAVE_TOTAL_LIMIT=0  # Not used when strategy is "no"
    echo "ℹ️  Intermediate checkpoints disabled (final model will be saved at end)"
    
    # For LoRA, use ZeRO Stage 2 instead of Stage 3 for better compatibility with PEFT
    # ZeRO Stage 3 + gradient checkpointing + PEFT can cause issues where checkpointed layers
    # don't have gradients (frozen base model). Stage 2 only partitions optimizer/gradients.
    if [ "$USE_LORA" == "true" ]; then
        if [ "$LR_SCHEDULER" == "constant" ]; then
            WARMUP_RATIO_FOR_EXP=0
            # For LoRA + constant, we'd need a no_scheduler version, but constant with LoRA is unlikely
            DEEPSPEED_CONFIG="train/configs/h200_optimized_bf16_lora.json"
            echo "ℹ️  LoRA + Constant scheduler: Using ZeRO Stage 2 config for LoRA compatibility"
        elif [ "$LR_SCHEDULER" == "cosine" ]; then
            WARMUP_RATIO_FOR_EXP=$WARMUP_RATIO
            # For LoRA + cosine, we'd need a no_scheduler version
            DEEPSPEED_CONFIG="train/configs/h200_optimized_bf16_lora.json"
            echo "ℹ️  LoRA + Cosine scheduler: Using ZeRO Stage 2 config for LoRA compatibility (note: will use linear decay)"
        else
            WARMUP_RATIO_FOR_EXP=$WARMUP_RATIO
            DEEPSPEED_CONFIG="train/configs/h200_optimized_bf16_lora.json"
            echo "ℹ️  LoRA: Using ZeRO Stage 2 config (model-level gradient checkpointing only, no DeepSpeed activation checkpointing)"
        fi
    # For constant and cosine schedulers, use DeepSpeed config without scheduler
    # (so transformers handles the scheduler instead of DeepSpeed's WarmupDecayLR)
    # DeepSpeed's WarmupDecayLR only supports linear decay, so it overrides cosine/constant
    elif [ "$LR_SCHEDULER" == "constant" ]; then
        WARMUP_RATIO_FOR_EXP=0
        DEEPSPEED_CONFIG="train/configs/h200_optimized_bf16_no_scheduler.json"
        echo "ℹ️  Constant scheduler: warmup disabled (warmup_ratio=0)"
        echo "ℹ️  Using DeepSpeed config without scheduler (transformers will handle constant LR)"
    elif [ "$LR_SCHEDULER" == "cosine" ]; then
        WARMUP_RATIO_FOR_EXP=$WARMUP_RATIO
        DEEPSPEED_CONFIG="train/configs/h200_optimized_bf16_no_scheduler.json"
        echo "ℹ️  Cosine scheduler: Using DeepSpeed config without scheduler (transformers will handle cosine LR)"
    else
        WARMUP_RATIO_FOR_EXP=$WARMUP_RATIO
        DEEPSPEED_CONFIG="train/configs/h200_optimized_bf16.json"
    fi
    
    # Build training command arguments
    TRAIN_CMD_ARGS=(
        --model_name_or_path $MODEL_NAME_OR_PATH
        --train_dataset_name_or_path $DATA_PATH
        # --eval_datasets removed - no evaluation during training
        --output_dir $SAVE_DIR
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU
        --per_device_eval_batch_size $BATCH_SIZE_PER_GPU
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS
        --eval_strategy $EVAL_STRATEGY
        --eval_steps $EVAL_STEPS
        --save_strategy "no"
        --save_steps 999999
        --save_total_limit 0
        --save_on_each_node False
        --load_best_model_at_end False
        --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS
        --ddp_timeout 14400
        --max_seq_length $MAX_SEQ_LENGTH
    )
    
    # Set learning rate - use 10x for LoRA experiments (common practice: LoRA adapters need higher LR)
    if [ "$USE_LORA" == "true" ]; then
        # Calculate 10x learning rate using python for proper scientific notation handling
        LORA_LEARNING_RATE=$(python3 -c "print($LEARNING_RATE * 10)")
        TRAIN_CMD_ARGS+=(
            --learning_rate $LORA_LEARNING_RATE
        )
        echo "ℹ️  LoRA: Using 10x learning rate ($LORA_LEARNING_RATE instead of $LEARNING_RATE)"
    else
        TRAIN_CMD_ARGS+=(
            --learning_rate $LEARNING_RATE
        )
    fi
    
    TRAIN_CMD_ARGS+=(
        --lr_scheduler_type $LR_SCHEDULER
        --warmup_ratio $WARMUP_RATIO_FOR_EXP
        --logging_steps 5
        --report_to "wandb"
        --run_name "$EXP_NAME"
    )
    
    # Disable gradient checkpointing for LoRA (conflicts with PEFT)
    # For full parameter tuning, enable gradient checkpointing for memory savings
    if [ "$USE_LORA" == "true" ]; then
        TRAIN_CMD_ARGS+=(
            --gradient_checkpointing False
        )
        echo "ℹ️  LoRA: Gradient checkpointing disabled (incompatible with PEFT/LoRA)"
    else
        TRAIN_CMD_ARGS+=(
            --gradient_checkpointing True
        )
    fi
    
    # Add DeepSpeed config
    TRAIN_CMD_ARGS+=(
        --deepspeed $DEEPSPEED_CONFIG
        --overwrite_output_dir
        --bf16 True
        --use_lora $USE_LORA
        --lora_rank $LORA_RANK
        --lora_alpha $LORA_RANK
        --use_auth_token True
        --remove_unused_columns False
    )
    
    # Add training duration (epochs or steps)
    if [ -n "${MAX_TRAIN_STEPS:-}" ]; then
        TRAIN_CMD_ARGS+=(--max_steps $MAX_TRAIN_STEPS)
        echo "Training for $MAX_TRAIN_STEPS steps"
    else
        TRAIN_CMD_ARGS+=(--num_train_epochs $NUM_TRAIN_EPOCHS)
        echo "Training for $NUM_TRAIN_EPOCHS epochs"
    fi
    
    # Run training
    torchrun \
        --nproc_per_node $NUM_GPUS \
        -m train.finetune \
        "${TRAIN_CMD_ARGS[@]}"
    
    echo "✅ Completed experiment: $EXP_NAME"
    echo ""
}

# =============================================================================
# EXPERIMENT 1: LR SCHEDULE EXPERIMENTS
# =============================================================================

run_lr_schedule_experiments() {
    echo "📊 Running LR Schedule Experiments"
    echo "==============================================="
    
    # Use 400 dataset and full parameter tuning
    DATA_PATH="train_test_data/converted_data_completion_new_400.json"
    USE_LORA="false"
    LORA_RANK=16  # Not used when USE_LORA=false
    
    # Linear scheduler (default)
    # run_training "lr_schedule_linear" "$DATA_PATH" "$USE_LORA" "$LORA_RANK" "linear"
    
    # Cosine scheduler
    run_training "lr_schedule_cosine" "$DATA_PATH" "$USE_LORA" "$LORA_RANK" "cosine"
    
    # Constant scheduler (no schedule)
    # run_training "lr_schedule_constant" "$DATA_PATH" "$USE_LORA" "$LORA_RANK" "constant"
    
    echo "✅ LR Schedule experiments completed"
}

# =============================================================================
# EXPERIMENT 2: LORA RANK EXPERIMENTS
# =============================================================================

run_lora_experiments() {
    echo "📊 Running LoRA Rank Experiments"
    echo "==============================================="
    
    # Use 400 dataset
    DATA_PATH="train_test_data/converted_data_completion_new_400.json"
    
    # Full parameter tuning
    # run_training "lora_full_param" "$DATA_PATH" "false" 16 "linear"
    
    # LoRA with different ranks
    for RANK in 4 8 16 32 64; do
        run_training "lora_rank_${RANK}" "$DATA_PATH" "true" "$RANK" "linear"
    done
    
    echo "✅ LoRA experiments completed"
}

# =============================================================================
# EXPERIMENT 3: DATA SIZE EXPERIMENTS
# =============================================================================

run_data_size_experiments() {
    echo "📊 Running Data Size Experiments"
    echo "==============================================="
    
    # Use full parameter tuning
    USE_LORA="false"
    LORA_RANK=16  # Not used when USE_LORA=false
    
    # Different data sizes
    for SIZE in 100 200 300 400; do
        DATA_PATH="train_test_data/converted_data_completion_new_${SIZE}.json"
        
        # Check if data file exists
        if [ ! -f "$DATA_PATH" ]; then
            echo "⚠️  Warning: $DATA_PATH not found, skipping size $SIZE"
            continue
        fi
        
        run_training "data_size_${SIZE}" "$DATA_PATH" "$USE_LORA" "$LORA_RANK" "linear"
    done
    
    echo "✅ Data size experiments completed"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Check if evaluation datasets exist
    MISSING_DATASETS=()
    [ ! -f "$INDUSTRYOR_EVAL" ] && MISSING_DATASETS+=("$INDUSTRYOR_EVAL")
    [ ! -f "$MAMO_COMPLEX_EVAL" ] && MISSING_DATASETS+=("$MAMO_COMPLEX_EVAL")
    [ ! -f "$NLP4LP_EVAL" ] && MISSING_DATASETS+=("$NLP4LP_EVAL")
    
    if [ ${#MISSING_DATASETS[@]} -gt 0 ]; then
        echo "❌ Error: Evaluation datasets not found:"
        for ds in "${MISSING_DATASETS[@]}"; do
            echo "   - $ds"
        done
        echo "Please ensure the datasets exist in baseline_test_data/ directory"
        exit 1
    fi
    
    # Check if DeepSpeed config exists
    if [ ! -f "train/configs/h200_optimized_bf16.json" ]; then
        echo "❌ Error: DeepSpeed config not found"
        exit 1
    fi
    
    # Parse command line arguments
    if [ "$1" == "lr_schedule" ]; then
        run_lr_schedule_experiments
    elif [ "$1" == "lora" ]; then
        run_lora_experiments
    elif [ "$1" == "data_size" ]; then
        run_data_size_experiments
    elif [ "$1" == "all" ]; then
        echo "🚀 Running ALL experiments"
        echo "This will take a long time..."
        echo ""
        run_lr_schedule_experiments
        run_lora_experiments
        run_data_size_experiments
        echo "🎉 All experiments completed!"
    else
        echo "Usage: $0 [lr_schedule|lora|data_size|all]"
        echo ""
        echo "Examples:"
        echo "  $0 lr_schedule    # Run LR schedule experiments"
        echo "  $0 lora          # Run LoRA rank experiments"
        echo "  $0 data_size     # Run data size experiments"
        echo "  $0 all           # Run all experiments"
        exit 1
    fi
}

main "$@"

