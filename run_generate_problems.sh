#!/usr/bin/env bash
set -euo pipefail

# Update this to your model path or Hugging Face model ID
MODEL="/home/hanzheng/orcd/scratch/models/Qwen3-30B-A3B-Instruct-2507"
# MODEL="/home/hanzheng/orcd/scratch/models/gpt-oss-20b"

## get the model name from the model path
MODEL_NAME=$(basename "$MODEL")

# Output directory for generated JSON files
OUT_DIR="problems_generated/$MODEL_NAME"
mkdir -p "$OUT_DIR"

# Use two H200 GPUs (adjust IDs if needed)
export CUDA_VISIBLE_DEVICES=0,1

# Fix MoE (Mixture of Experts) issues with Qwen3-30B-A3B model
# NOTE: run_eval.sh works because it uses Qwen3-8B (dense model, no MoE)
#       This script uses Qwen3-30B-A3B (MoE model) which has broken CUDA extension
# Use Triton MoE backend instead of broken CUDA extension
# export VLLM_USE_TRITON_MOE=1
# # Disable torch.compile which triggers MoE extension errors
# # (Empty string means disable all torch.compile)
# export VLLM_TORCH_COMPILE_LAYERS=""

# 1) Generate problems using batched generation (FAST - recommended)
# Batching and prefix caching are enabled by default for better performance
# This will generate 1 problem per application (only "Facility location" is currently enabled)
python generate_problem.py \
  --model_name_or_path "$MODEL" \
  --tensor_parallel_size 2 \
  --gpu_memory_utilization 0.9 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_tokens 2048 \
  --enable_triton_moe \
  --problem_gen_output "$OUT_DIR/generated_milp_problems_facility_location_v1.json" \
  --problem_gen_apps "Facility location" \
  --problem_gen_num_per_app 10

# 2) Generate multiple problems per application (batched - FAST!)
# This will generate 10 problems for "Facility location" in one batched call
# All prompts share the same prefix, so prefix caching makes this very efficient
# python generate_problem.py \
#   --model_name_or_path "$MODEL" \
#   --tensor_parallel_size 2 \
#   --gpu_memory_utilization 0.9 \
#   --temperature 0.7 \
#   --top_p 0.95 \
#   --max_tokens 2048 \
#   --enable_triton_moe \
#   --problem_gen_output "$OUT_DIR/generated_facility_location_10.json" \
#   --problem_gen_apps "Facility location" \
#   --problem_gen_num_per_app 10

# 3) Disable batching (slower, one-by-one generation - for debugging)
# Use this only if you need to debug individual generations
# python generate_problem.py \
#   --model_name_or_path "$MODEL" \
#   --tensor_parallel_size 2 \
#   --gpu_memory_utilization 0.9 \
#   --temperature 0.7 \
#   --top_p 0.95 \
#   --max_tokens 2048 \
#   --enable_triton_moe \
#   --disable_batching \
#   --problem_gen_output "$OUT_DIR/generated_slow.json" \
#   --problem_gen_apps "Facility location" \
#   --problem_gen_num_per_app 1

echo "All generation commands finished. Outputs in: $OUT_DIR"


