#!/bin/bash

# Evaluation script for the base Qwen3-8B model (before fine-tuning)

# Base model path
BASE_MODEL_PATH="/home/hanzheng/orcd/scratch/models/Qwen3-8B"

# Same datasets as run_eval.sh for consistency
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

# Output directory for base model
OUTPUT_DIR="results/Qwen3-8B-base"

echo "Evaluating base model: $BASE_MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "==============================================="

# Check if base model directory exists
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ ERROR: Base model directory not found: $BASE_MODEL_PATH"
    exit 1
fi

for dataset in "${datasets[@]}"; do
    echo ""
    echo "Evaluating $dataset with base model: $BASE_MODEL_PATH"
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --timeout 60 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.8 \
        --test_file baseline_test_data/$dataset \
        --output_file ${OUTPUT_DIR}/$dataset \
        --verbose
done

echo ""
echo "✅ Completed evaluation for base model"
echo "Results saved in: $OUTPUT_DIR"

