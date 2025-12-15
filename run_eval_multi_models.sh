#!/bin/bash

# Evaluation script for AlphaOpt ORLM models
# Tests 4 models: ft_new_100, ft_new_200, ft_new_300, ft_new_400

datasets=(
   "complexor.json"
   "mamo_easy.json"
   "optibench.json"
   "optmath_bench.json"
   "logior.json"
)

models=(
   "100"
   "200"
   "300"
   "400"
)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python eval.py --model_name_or_path /orcd/scratch/seedfund/001/multimodal/qua/huggingface/hub/models--AlphaOpt_ORLM_Qwen2-7B-Instruct_ft_new_$model --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --test_file ../baseline_test_data/$dataset --output_file results/AlphaOpt_ORLM_Qwen2-7B-Instruct_ft_new_$model/$dataset --verbose
    done
done
