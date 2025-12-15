#!/usr/bin/env python
"""
Utility script to merge LoRA adapter with base model for evaluation.
This is needed because vLLM requires a full model, not just LoRA adapters.
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: PEFT library is required. Install with: pip install peft")
    sys.exit(1)


def is_lora_model(model_path):
    """Check if the model path contains LoRA adapter files."""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    return os.path.exists(adapter_config_path)


def merge_lora_model(lora_model_path, base_model_path, output_path):
    """
    Merge LoRA adapter with base model and save to output path.
    
    Args:
        lora_model_path: Path to LoRA adapter directory
        base_model_path: Path to base model
        output_path: Path where merged model will be saved
    """
    if not os.path.exists(lora_model_path):
        raise ValueError(f"LoRA model path does not exist: {lora_model_path}")
    
    if not is_lora_model(lora_model_path):
        raise ValueError(f"Not a LoRA model (no adapter_config.json found): {lora_model_path}")
    
    if not os.path.exists(base_model_path):
        raise ValueError(f"Base model path does not exist: {base_model_path}")
    
    print(f"Loading base model from: {base_model_path}")
    print(f"Loading LoRA adapter from: {lora_model_path}")
    
    # Load base model and tokenizer
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge adapter into base model
    print("Merging LoRA adapter into base model...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ Merged model saved successfully to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/home/hanzheng/orcd/scratch/models/Qwen3-8B",
        help="Path to base model (default: /home/hanzheng/orcd/scratch/models/Qwen3-8B)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where merged model will be saved (default: lora_model_path + '_merged')"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        args.output_path = args.lora_model_path.rstrip('/') + "_merged"
    
    # Check if output exists
    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Output path already exists: {args.output_path}")
        print("Use --overwrite to overwrite, or specify a different --output_path")
        sys.exit(1)
    
    # Remove existing output if overwriting
    if os.path.exists(args.output_path) and args.overwrite:
        print(f"Removing existing directory: {args.output_path}")
        shutil.rmtree(args.output_path)
    
    try:
        merge_lora_model(args.lora_model_path, args.base_model_path, args.output_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

