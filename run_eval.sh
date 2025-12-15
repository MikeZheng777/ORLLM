#!/bin/bash

datasets=(
   # "complexor.json"
   "industryor_test.json"
   "mamo_complex_test.json"
   # "mamo_easy.json"
# #    "nl4opt_test.json"
   "nlp4lp_test.json"
#    "optibench.json"
   # "optmath_bench.json"
# #    "logior_test.json"
#      "logior.json"
)

# Base model path for LoRA merging
BASE_MODEL_PATH="/home/hanzheng/orcd/scratch/models/Qwen3-8B"

# Model path - can be passed as argument or environment variable
# Usage: ./run_eval.sh [MODEL_PATH]
# Or: MODEL_PATH=/path/to/model ./run_eval.sh
if [ -n "$1" ]; then
    MODEL_PATH="$1"
elif [ -n "$MODEL_PATH" ]; then
    # Use environment variable if set
    MODEL_PATH="$MODEL_PATH"
else
    # Default fallback
    MODEL_PATH="/home/hanzheng/orcd/scratch/orlm_finetune/experiments/lora_rank_1"
fi

# Check if this is a LoRA model (has adapter_config.json)
if [ -f "${MODEL_PATH}/adapter_config.json" ]; then
    echo "🔍 Detected LoRA model: $MODEL_PATH"
    MERGED_MODEL_PATH="${MODEL_PATH}_merged"
    
    # Check if merged model already exists
    if [ -d "$MERGED_MODEL_PATH" ] && [ -f "${MERGED_MODEL_PATH}/config.json" ]; then
        echo "✅ Using existing merged model: $MERGED_MODEL_PATH"
        MODEL_PATH="$MERGED_MODEL_PATH"
    else
        echo "🔄 Merging LoRA adapter with base model..."
        echo "   LoRA adapter: $MODEL_PATH"
        echo "   Base model: $BASE_MODEL_PATH"
        echo "   Output: $MERGED_MODEL_PATH"
        
        python merge_lora_model.py \
            --lora_model_path "$MODEL_PATH" \
            --base_model_path "$BASE_MODEL_PATH" \
            --output_path "$MERGED_MODEL_PATH" \
            --overwrite
        
        if [ $? -eq 0 ]; then
            echo "✅ LoRA model merged successfully"
            MODEL_PATH="$MERGED_MODEL_PATH"
        else
            echo "❌ Failed to merge LoRA model. Exiting."
            exit 1
        fi
    fi
fi

# Output directory based on original model name (before merging)
ORIGINAL_MODEL_NAME=$(basename "$MODEL_PATH" | sed 's/_merged$//')
OUTPUT_DIR="results/${ORIGINAL_MODEL_NAME}"

for dataset in "${datasets[@]}"; do
    echo "Evaluating $dataset with model: $MODEL_PATH"
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --model_name_or_path "$MODEL_PATH" \
        --timeout 60 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.8 \
        --test_file baseline_test_data/$dataset \
        --output_file ${OUTPUT_DIR}/$dataset \
        --verbose
    # CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --test_file ../baseline_test_data/$dataset --output_file results/Meta-Llama-3-8B-Instruct/$dataset --verbose
      #   CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path /orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/models--AlphaOpt_ORLM_Qwen2-7B-Instruct_ft_new_200 --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --test_file /baseline_test_data/$dataset --output_file results/AlphaOpt_ORLM_Qwen2-7B-Instruct_ft_new_200_v2/$dataset --verbose

done

